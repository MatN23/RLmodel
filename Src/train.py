#!/usr/bin/env python3

import torch
import torch.nn as nn
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
            'rewards': []
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
                
                # Value loss
                value_loss = nn.MSELoss()(new_values, returns[i])
                
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
        
        # Update statistics
        self.training_stats['policy_loss'].append(total_policy_loss / (4 * batch_size))
        self.training_stats['value_loss'].append(total_value_loss / (4 * batch_size))
        self.training_stats['entropy'].append(total_entropy / (4 * batch_size))
        self.training_stats['total_loss'].append(
            (total_policy_loss + total_value_loss - total_entropy) / (4 * batch_size)
        )
        self.training_stats['rewards'].append(rewards.mean().item())
        
        return {
            'policy_loss': total_policy_loss / (4 * batch_size),
            'value_loss': total_value_loss / (4 * batch_size),
            'entropy': total_entropy / (4 * batch_size),
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
        max_length=args.max_length
    )
    
    # Load dataset
    dataset = OASST2Dataset(
        data_path=args.data_path,
        tokenizer=model.tokenizer,
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
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            metrics = trainer.train_step(batch)
            train_metrics.append(metrics)
            
            train_pbar.set_postfix({
                'Policy Loss': f"{metrics['policy_loss']:.4f}",
                'Value Loss': f"{metrics['value_loss']:.4f}",
                'Reward': f"{metrics['avg_reward']:.4f}"
            })
        
        # Validation
        model.eval()
        val_rewards = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                prompt_texts = batch['prompt_text']
                response_texts = batch['response_text']
                quality_scores = batch['quality_score']
                
                batch_rewards = []
                for i in range(len(prompt_texts)):
                    reward = trainer.compute_reward(
                        prompt_texts[i], response_texts[i], quality_scores[i].item()
                    )
                    batch_rewards.append(reward)
                
                val_rewards.extend(batch_rewards)
        
        avg_val_reward = np.mean(val_rewards)
        avg_train_reward = np.mean([m['avg_reward'] for m in train_metrics])
        
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Reward: {avg_train_reward:.4f}")
        logger.info(f"  Val Reward: {avg_val_reward:.4f}")
        logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_reward > best_reward:
            best_reward = avg_val_reward
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'training_stats': trainer.training_stats
            }, args.save_path)
            logger.info(f"Saved new best model with validation reward: {best_reward:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Generate sample response every few epochs
        if (epoch + 1) % 5 == 0:
            sample_prompt = "Hello, how are you doing today?"
            response = model.generate_response(sample_prompt, max_new_tokens=30)
            logger.info(f"Sample generation - Prompt: '{sample_prompt}' -> Response: '{response}'")
    
    logger.info("Training completed!")
    
    # Plot training statistics
    if args.plot_stats:
        plot_training_stats(trainer.training_stats, args.save_path.replace('.pt', '_stats.png'))


def plot_training_stats(stats, save_path):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(stats['policy_loss'])
    axes[0, 0].set_title('Policy Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(stats['value_loss'])
    axes[0, 1].set_title('Value Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    
    axes[1, 0].plot(stats['entropy'])
    axes[1, 0].set_title('Entropy')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entropy')
    
    axes[1, 1].plot(stats['rewards'])
    axes[1, 1].set_title('Average Reward')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Hardcoded configuration
    class Config:
        data_path = 'data.jsonl'  # Path to your OASST2 format data file
        save_path = 'rl_chat_model.pt'
        base_model = 'distilbert-base-uncased'
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