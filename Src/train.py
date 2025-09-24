#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our model (assuming it's in the same directory)
from rl_model import RLChatModel, OASST2Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOTrainer:
    def __init__(self, model, optimizer, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'rewards': [],
            'accuracy': [],
            'perplexity': []
        }
    
    def compute_returns(self, rewards, values, gamma=0.99, lam=0.95):
        """Compute GAE returns"""
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value - values[i]
            gae = delta + gamma * lam * gae
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        return torch.tensor(returns), torch.tensor(advantages)
    
    def compute_metrics(self, batch):
        """Compute accuracy and perplexity metrics"""
        prompt_ids = batch['prompt_ids']
        prompt_mask = batch['prompt_mask']
        response_ids = batch['response_ids']
        response_mask = batch['response_mask']
        
        batch_size = prompt_ids.shape[0]
        total_accuracy = 0
        total_perplexity = 0
        valid_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for i in range(batch_size):
                try:
                    # Get model predictions for prompt (which should predict next token)
                    policy_logits, _ = self.model.forward(prompt_ids[i:i+1], prompt_mask[i:i+1])
                    
                    # Get the first non-pad token from response as target
                    response_tokens = response_ids[i]
                    response_mask_i = response_mask[i]
                    
                    # Find first non-pad token in response
                    non_pad_indices = (response_tokens != self.model.pad_token_id).nonzero()
                    
                    if len(non_pad_indices) > 0:
                        target_token = response_tokens[non_pad_indices[0]]
                        
                        # Calculate accuracy (check if predicted token matches target)
                        predicted_token = torch.argmax(policy_logits, dim=-1).squeeze()
                        correct = (predicted_token == target_token).float().item()
                        total_accuracy += correct
                        
                        # Calculate perplexity using negative log probability of target token
                        log_probs = F.log_softmax(policy_logits, dim=-1)
                        target_log_prob = log_probs[0, target_token].item()
                        perplexity = torch.exp(-torch.tensor(target_log_prob)).item()
                        
                        total_perplexity += perplexity
                        valid_samples += 1
                        
                except Exception as e:
                    logger.warning(f"Error computing metrics for sample {i}: {e}")
                    continue
        
        accuracy = total_accuracy / max(valid_samples, 1)
        avg_perplexity = total_perplexity / max(valid_samples, 1)
        
        return accuracy, avg_perplexity
        
    def compute_reward(self, prompt_text, response_text, quality_score):
        """Compute reward for a prompt-response pair"""
        # Simple reward function - you can make this more sophisticated
        base_reward = quality_score
        
        # Penalty for very short responses
        if len(response_text.split()) < 3:
            base_reward -= 0.3
        
        # Bonus for appropriate length responses
        if 5 <= len(response_text.split()) <= 50:
            base_reward += 0.1
        
        # Penalty for repetitive responses
        words = response_text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            base_reward += 0.2 * unique_ratio - 0.1
        
        return max(base_reward, 0.0)  # Ensure non-negative reward
    
    def train_step(self, batch):
        """Single PPO training step"""
        prompt_ids = batch['prompt_ids']
        prompt_mask = batch['prompt_mask']
        response_ids = batch['response_ids']
        quality_scores = batch['quality_score']
        prompt_texts = batch['prompt_text']
        response_texts = batch['response_text']
        
        batch_size = prompt_ids.shape[0]
        
        # Collect experiences
        old_log_probs = []
        values = []
        rewards = []
        actions = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(batch_size):
                try:
                    # Compute reward
                    reward = self.compute_reward(prompt_texts[i], response_texts[i], quality_scores[i].item())
                    rewards.append(reward)
                    
                    # Get action (next token to generate)
                    action, log_prob, value = self.model.act(
                        prompt_ids[i:i+1], 
                        prompt_mask[i:i+1]
                    )
                    
                    old_log_probs.append(log_prob.item())
                    values.append(value.item())
                    actions.append(action.item())
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    # Use default values for failed samples
                    rewards.append(0.0)
                    old_log_probs.append(-1.0)
                    values.append(0.0)
                    actions.append(0)
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns(rewards, values)
        
        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        self.model.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for ppo_epoch in range(4):  # PPO epochs
            for i in range(batch_size):
                try:
                    # Evaluate current policy
                    new_log_probs, new_values, entropy = self.model.evaluate_actions(
                        prompt_ids[i:i+1],
                        prompt_mask[i:i+1], 
                        actions[i:i+1]
                    )
                    
                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - old_log_probs[i])
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    
                    policy_loss = -torch.min(
                        ratio * advantages[i],
                        clipped_ratio * advantages[i]
                    )
                    
                    # Value loss - fix the dimension mismatch
                    value_loss = F.mse_loss(new_values.squeeze(), returns[i])
                    
                    # Total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                except Exception as e:
                    logger.warning(f"Error in training step for sample {i}, epoch {ppo_epoch}: {e}")
                    continue
        
        # Update statistics
        avg_policy_loss = total_policy_loss / (4 * batch_size)
        avg_value_loss = total_value_loss / (4 * batch_size)
        avg_entropy = total_entropy / (4 * batch_size)
        
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy'].append(avg_entropy)
        self.training_stats['total_loss'].append(avg_policy_loss + avg_value_loss - avg_entropy)
        self.training_stats['rewards'].append(rewards.mean().item())
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'avg_reward': rewards.mean().item()
        }


def train_model(args):
    """Main training function"""
    logger.info("Starting training...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize model
    model = RLChatModel(
        base_model_name=args.base_model,
        tiktoken_encoding=args.tiktoken_encoding,
        max_length=args.max_length
    )
    
    # Load dataset (note: dataset now needs the model for tokenization)
    dataset = OASST2Dataset(
        data_path=args.data_path,
        model=model,  # Pass the model for tokenization
        max_length=args.max_length
    )
    
    logger.info(f"Loaded {len(dataset)} conversations")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Initialize trainer
    trainer = PPOTrainer(model, optimizer)
    
    # Training loop
    best_reward = -float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_metrics = []
        train_accuracies = []
        train_perplexities = []
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            metrics = trainer.train_step(batch)
            train_metrics.append(metrics)
            
            # Compute accuracy and perplexity
            accuracy, perplexity = trainer.compute_metrics(batch)
            train_accuracies.append(accuracy)
            train_perplexities.append(perplexity)
            
            train_pbar.set_postfix({
                'Policy Loss': f"{metrics['policy_loss']:.4f}",
                'Value Loss': f"{metrics['value_loss']:.4f}",
                'Reward': f"{metrics['avg_reward']:.4f}",
                'Acc': f"{accuracy:.3f}",
                'PPL': f"{perplexity:.2f}"
            })
        
        # Validation
        model.eval()
        val_rewards = []
        val_accuracies = []
        val_perplexities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                prompt_texts = batch['prompt_text']
                response_texts = batch['response_text']
                quality_scores = batch['quality_score']
                
                # Compute rewards
                batch_rewards = []
                for i in range(len(prompt_texts)):
                    reward = trainer.compute_reward(
                        prompt_texts[i], response_texts[i], quality_scores[i].item()
                    )
                    batch_rewards.append(reward)
                val_rewards.extend(batch_rewards)
                
                # Compute accuracy and perplexity
                accuracy, perplexity = trainer.compute_metrics(batch)
                val_accuracies.append(accuracy)
                val_perplexities.append(perplexity)
        
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0.0
        avg_train_reward = np.mean([m['avg_reward'] for m in train_metrics]) if train_metrics else 0.0
        
        # Calculate epoch averages
        avg_train_accuracy = np.mean(train_accuracies) if train_accuracies else 0.0
        avg_train_perplexity = np.mean(train_perplexities) if train_perplexities else 0.0
        avg_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0.0
        avg_val_perplexity = np.mean(val_perplexities) if val_perplexities else 0.0
        
        # Update epoch-level statistics
        trainer.training_stats['accuracy'].append(avg_train_accuracy)
        trainer.training_stats['perplexity'].append(avg_train_perplexity)
        
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train - Reward: {avg_train_reward:.4f}, Accuracy: {avg_train_accuracy:.4f}, PPL: {avg_train_perplexity:.2f}")
        logger.info(f"  Val   - Reward: {avg_val_reward:.4f}, Accuracy: {avg_val_accuracy:.4f}, PPL: {avg_val_perplexity:.2f}")
        logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_reward > best_reward:
            best_reward = avg_val_reward
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'training_stats': trainer.training_stats,
                'tokenizer_encoding': args.tiktoken_encoding,
                'vocab_size': model.vocab_size
            }, args.save_path)
            logger.info(f"Saved new best model with validation reward: {best_reward:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Generate sample response every few epochs
        if (epoch + 1) % 5 == 0:
            try:
                sample_prompt = "Hello, how are you doing today?"
                response = model.generate_response(sample_prompt, max_new_tokens=30)
                logger.info(f"Sample generation - Prompt: '{sample_prompt}' -> Response: '{response}'")
            except Exception as e:
                logger.warning(f"Error generating sample response: {e}")
    
    logger.info("Training completed!")
    
    # Plot training statistics
    if args.plot_stats:
        try:
            plot_training_stats(trainer.training_stats, args.save_path.replace('.pt', '_stats.png'))
        except Exception as e:
            logger.warning(f"Error plotting training statistics: {e}")


def plot_training_stats(stats, save_path):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if stats['policy_loss']:
        axes[0, 0].plot(stats['policy_loss'])
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    if stats['value_loss']:
        axes[0, 1].plot(stats['value_loss'])
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    if stats['entropy']:
        axes[0, 2].plot(stats['entropy'])
        axes[0, 2].set_title('Entropy')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Entropy')
        axes[0, 2].grid(True)
    
    if stats['rewards']:
        axes[1, 0].plot(stats['rewards'])
        axes[1, 0].set_title('Average Reward')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
    
    if stats['accuracy']:
        axes[1, 1].plot(stats['accuracy'])
        axes[1, 1].set_title('Training Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True)
    
    if stats['perplexity']:
        axes[1, 2].plot(stats['perplexity'])
        axes[1, 2].set_title('Training Perplexity')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Perplexity')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training statistics plot saved to {save_path}")


def main():
    # Hardcoded configuration
    class Config:
        data_path = 'oasst1_train.jsonl'
        save_path = 'rl_chat_model_tiktoken.pt'
        base_model = 'distilbert-base-uncased'
        tiktoken_encoding = 'cl100k_base'  # GPT-4 encoding
        max_length = 512
        batch_size = 8
        learning_rate = 1e-4
        num_epochs = 10
        seed = 42
        plot_stats = True
    
    args = Config()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    
    # Train model
    train_model(args)


if __name__ == "__main__":
    main()