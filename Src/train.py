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
from collections import deque
import random
from typing import Dict, List, Tuple

# Import our model
from rl_model import RLChatModel, ConversationDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PPOTrainer:
    def __init__(self, model, optimizer, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, gamma=0.99, lam=0.95):
        self.model = model
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'rewards': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        # Experience buffer for collecting episodes
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': []
        }
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
            else:
                next_value = values[i + 1] if i + 1 < len(values) else 0
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * gae * (1 - dones[i])
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantages, dtype=torch.float32)
    
    def compute_reward(self, prompt: str, response: str, quality_score: float) -> float:
        """Compute reward for a prompt-response pair"""
        # Base reward from quality score
        reward = quality_score
        
        # Response length penalty/bonus
        response_len = len(response.split())
        if response_len < 3:
            reward -= 0.5  # Too short
        elif response_len > 100:
            reward -= 0.3  # Too long
        elif 5 <= response_len <= 50:
            reward += 0.2  # Good length
        
        # Repetition penalty
        words = response.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:
                reward -= 0.4  # Very repetitive
            else:
                reward += 0.1 * repetition_ratio
        
        # Coherence bonus (simple heuristic)
        if any(word in response.lower() for word in ["sorry", "i don't know", "i can't"]):
            reward -= 0.1
        
        # Engagement bonus
        if any(word in response.lower() for word in ["how", "what", "would", "could", "let"]):
            reward += 0.1
        
        return max(reward, 0.1)  # Minimum reward to avoid negative reinforcement
    
    def collect_episode(self, prompt: str, quality_score: float, max_response_tokens: int = 50) -> Dict:
        """Collect a complete episode (full response generation)"""
        self.model.eval()
        
        response, log_probs, values = self.model.generate_sequence(
            prompt, 
            max_new_tokens=max_response_tokens,
            temperature=0.8
        )
        
        # Compute final reward
        final_reward = self.compute_reward(prompt, response, quality_score)
        
        # Create reward signal - only reward at the end
        rewards = [0.0] * (len(log_probs) - 1) + [final_reward] if log_probs else [final_reward]
        dones = [False] * (len(log_probs) - 1) + [True] if log_probs else [True]
        
        # Pad lists to same length
        min_len = min(len(log_probs), len(values), len(rewards))
        log_probs = log_probs[:min_len]
        values = values[:min_len]
        rewards = rewards[:min_len]
        dones = dones[:min_len]
        
        return {
            'prompt': prompt,
            'response': response,
            'log_probs': log_probs,
            'values': values,
            'rewards': rewards,
            'dones': dones,
            'final_reward': final_reward
        }
    
    def update_policy(self, episodes: List[Dict]) -> Dict:
        """Update policy using collected episodes"""
        if not episodes:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_div': 0, 'clip_frac': 0}
        
        # Flatten all episodes into training data
        all_log_probs = []
        all_values = []
        all_returns = []
        all_advantages = []
        
        for episode in episodes:
            if not episode['log_probs'] or not episode['values']:
                continue
                
            returns, advantages = self.compute_gae(
                episode['rewards'], 
                episode['values'], 
                episode['dones']
            )
            
            all_log_probs.extend(episode['log_probs'])
            all_values.extend(episode['values'])
            all_returns.extend(returns.tolist())
            all_advantages.extend(advantages.tolist())
        
        if not all_log_probs:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_div': 0, 'clip_frac': 0}
        
        # Convert to tensors
        old_log_probs = torch.tensor(all_log_probs, dtype=torch.float32)
        old_values = torch.tensor(all_values, dtype=torch.float32)
        returns = torch.tensor(all_returns, dtype=torch.float32)
        advantages = torch.tensor(all_advantages, dtype=torch.float32)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        self.model.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        total_clip_frac = 0
        num_updates = 0
        
        # Mini-batch training
        batch_size = 64
        indices = list(range(len(old_log_probs)))
        
        for epoch in range(4):  # PPO epochs
            random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                if len(batch_indices) < 4:  # Skip tiny batches
                    continue
                
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Re-evaluate policy (this is simplified - in practice you'd need to store states)
                # For now, we'll use a surrogate approach
                
                # Compute policy loss using importance sampling
                # Note: This is a simplified version - full implementation would re-evaluate the policy
                ratio = torch.ones_like(batch_old_log_probs)  # Simplified
                
                policy_loss_1 = ratio * batch_advantages
                policy_loss_2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                value_loss = F.mse_loss(old_values[batch_indices], batch_returns)
                
                # Entropy loss (encourage exploration)
                entropy_loss = 0.01  # Simplified
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss
                
                # KL divergence and clipping fraction (simplified)
                kl_div = 0.0  # Would compute actual KL div in full implementation
                clip_frac = 0.0  # Would compute actual clipping fraction
                
                total_kl_div += kl_div
                total_clip_frac += clip_frac
                num_updates += 1
        
        # Average metrics
        if num_updates > 0:
            metrics = {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy': total_entropy / num_updates,
                'kl_div': total_kl_div / num_updates,
                'clip_frac': total_clip_frac / num_updates
            }
        else:
            metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'kl_div': 0, 'clip_frac': 0}
        
        # Update training stats
        for key, value in metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return metrics
    
    def train_epoch(self, dataloader: DataLoader, episodes_per_batch: int = 4) -> Dict:
        """Train for one epoch"""
        epoch_episodes = []
        epoch_rewards = []
        
        for batch in tqdm(dataloader, desc="Collecting Episodes"):
            batch_episodes = []
            
            for i in range(min(len(batch['prompt_text']), episodes_per_batch)):
                prompt = batch['prompt_text'][i]
                quality = batch['quality_score'][i].item()
                
                episode = self.collect_episode(prompt, quality)
                batch_episodes.append(episode)
                epoch_rewards.append(episode['final_reward'])
            
            epoch_episodes.extend(batch_episodes)
            
            # Update policy every few batches
            if len(epoch_episodes) >= 16:  # Update when we have enough episodes
                metrics = self.update_policy(epoch_episodes[-16:])
                if metrics['policy_loss'] != 0:  # Only log if we actually updated
                    logger.debug(f"Policy update - Loss: {metrics['policy_loss']:.4f}, Value: {metrics['value_loss']:.4f}")
        
        # Final policy update with all episodes
        final_metrics = self.update_policy(epoch_episodes)
        
        # Update reward stats
        if epoch_rewards:
            self.training_stats['rewards'].append(np.mean(epoch_rewards))
        
        return {
            'avg_reward': np.mean(epoch_rewards) if epoch_rewards else 0,
            'num_episodes': len(epoch_episodes),
            **final_metrics
        }


def train_model(config):
    """Main training function"""
    logger.info("Starting RL training...")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Initialize model
    model = RLChatModel(
        base_model_name=config.base_model,
        max_length=config.max_length
    )
    
    # Load dataset
    dataset = ConversationDataset(
        data_path=config.data_path,
        tokenizer=model.tokenizer,
        max_length=config.max_length
    )
    
    logger.info(f"Loaded {len(dataset)} conversations")
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize optimizer and trainer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    trainer = PPOTrainer(model, optimizer)
    
    # Training loop
    best_reward = -float('inf')
    
    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_rewards = []
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                for i in range(min(len(batch['prompt_text']), 2)):  # Sample validation
                    prompt = batch['prompt_text'][i]
                    quality = batch['quality_score'][i].item()
                    
                    episode = trainer.collect_episode(prompt, quality)
                    val_rewards.append(episode['final_reward'])
        
        avg_val_reward = np.mean(val_rewards) if val_rewards else 0
        
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Train Episodes: {train_metrics['num_episodes']}")
        logger.info(f"  Train Reward: {train_metrics['avg_reward']:.4f}")
        logger.info(f"  Val Reward: {avg_val_reward:.4f}")
        logger.info(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
        logger.info(f"  Value Loss: {train_metrics['value_loss']:.4f}")
        
        # Save best model
        if avg_val_reward > best_reward:
            best_reward = avg_val_reward
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainer_stats': trainer.training_stats,
                'epoch': epoch,
                'best_reward': best_reward,
                'vocab_size': model.vocab_size,
                'config': config.__dict__
            }, config.save_path)
            logger.info(f"Saved new best model with reward: {best_reward:.4f}")
        
        scheduler.step()
        
        # Generate sample
        if (epoch + 1) % 2 == 0:
            try:
                sample_prompt = "Hello! How can I help you today?"
                response = model.generate_response(sample_prompt, max_new_tokens=30)
                logger.info(f"Sample - Prompt: '{sample_prompt}' -> Response: '{response}'")
            except Exception as e:
                logger.warning(f"Error generating sample: {e}")
    
    logger.info("Training completed!")
    
    # Plot training statistics
    if config.plot_stats:
        try:
            plot_training_stats(trainer.training_stats, config.save_path.replace('.pt', '_stats.png'))
        except Exception as e:
            logger.warning(f"Error plotting stats: {e}")


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    # Extract all fields
    prompt_texts = [item['prompt_text'] for item in batch]
    response_texts = [item['response_text'] for item in batch]
    quality_scores = torch.stack([item['quality_score'] for item in batch])
    
    return {
        'prompt_text': prompt_texts,
        'response_text': response_texts,
        'quality_score': quality_scores
    }


def plot_training_stats(stats, save_path):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if stats['policy_loss']:
        axes[0, 0].plot(stats['policy_loss'])
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Update')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    if stats['value_loss']:
        axes[0, 1].plot(stats['value_loss'])
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    if stats['entropy']:
        axes[0, 2].plot(stats['entropy'])
        axes[0, 2].set_title('Entropy')
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Entropy')
        axes[0, 2].grid(True)
    
    if stats['rewards']:
        axes[1, 0].plot(stats['rewards'])
        axes[1, 0].set_title('Average Reward')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
    
    if stats['kl_divergence']:
        axes[1, 1].plot(stats['kl_divergence'])
        axes[1, 1].set_title('KL Divergence')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('KL Div')
        axes[1, 1].grid(True)
    
    if stats['clip_fraction']:
        axes[1, 2].plot(stats['clip_fraction'])
        axes[1, 2].set_title('Clip Fraction')
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Clip Frac')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training stats plot saved to {save_path}")


def main():
    """Main function"""
    class Config:
        # Data and model
        data_path = 'conversations.jsonl'  # Your conversation data
        save_path = 'rl_chat_model_fixed.pt'
        base_model = 'microsoft/DialoGPT-small'  # Better choice for chat
        max_length = 512
        
        # Training hyperparameters
        batch_size = 4  # Small batch size for RL
        learning_rate = 5e-5  # Lower learning rate for stability
        num_epochs = 20
        seed = 42
        
        # Other options
        plot_stats = True
    
    config = Config()
    
    # Create output directory
    os.makedirs(os.path.dirname(config.save_path) if os.path.dirname(config.save_path) else '.', exist_ok=True)
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()