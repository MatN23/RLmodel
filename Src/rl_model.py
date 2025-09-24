#!/usr/bin/env python3
"""
RL Chat Model - Reinforcement Learning Model for Conversational AI

This module contains the main model definition and dataset loader for
training a conversational AI using reinforcement learning with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.distributions import Categorical
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional

class RLChatModel(nn.Module):
    def __init__(self, base_model_name="microsoft/DialoGPT-small", hidden_dim=768, max_length=512):
        super(RLChatModel, self).__init__()
        
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.base_model_name = base_model_name
        
        # Use proper transformers tokenizer instead of tiktoken
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Add special tokens if they don't exist
        special_tokens = {"pad_token": "<|pad|>", "eos_token": "<|endoftext|>"}
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"Vocab size: {self.vocab_size}, EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")
        
        # Load base transformer model
        config = AutoConfig.from_pretrained(base_model_name)
        config.vocab_size = self.vocab_size  # Update vocab size
        
        self.base_model = AutoModel.from_pretrained(base_model_name, config=config)
        
        # Resize embeddings to match new vocab size
        self.base_model.resize_token_embeddings(self.vocab_size)
        
        # Actor head - outputs logits over vocabulary
        self.actor_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, self.vocab_size)
        )
        
        # Critic head - estimates state values
        self.critic_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Encode text using the tokenizer"""
        if max_length is None:
            max_length = self.max_length
            
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return encoded
    
    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens using the tokenizer"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model"""
        # Get outputs from base transformer
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden states
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # For generation, we want logits for each position
        actor_logits = self.actor_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        # For value estimation, use pooled representation
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden = hidden_states * mask_expanded
            pooled = masked_hidden.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)
            
        values = self.critic_head(pooled).squeeze(-1)  # [batch_size]
        
        return actor_logits, values
    
    def generate_sequence(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7, 
                         top_k: int = 50, top_p: float = 0.9) -> Tuple[str, List[float], List[float]]:
        """Generate a complete response sequence with action probabilities and values"""
        self.eval()
        
        # Encode prompt
        inputs = self.encode_text(prompt)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        generated_tokens = []
        log_probs = []
        values = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model outputs
                actor_logits, value = self.forward(input_ids, attention_mask)
                
                # Get logits for next token (last position)
                next_token_logits = actor_logits[0, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Store log probability and value
                log_prob = torch.log(probs[next_token] + 1e-8)
                log_probs.append(log_prob.item())
                values.append(value.item())
                
                # Check for end token
                if next_token.item() == self.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                
                # Update input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
                
                # Truncate if too long
                if input_ids.shape[1] > self.max_length:
                    input_ids = input_ids[:, -self.max_length:]
                    attention_mask = attention_mask[:, -self.max_length:]
        
        # Decode response
        if generated_tokens:
            response = self.decode_tokens(generated_tokens)
        else:
            response = ""
            
        return response, log_probs, values
    
    def generate_response(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Simple response generation (wrapper for backward compatibility)"""
        response, _, _ = self.generate_sequence(prompt, max_new_tokens, temperature)
        return response
    
    def evaluate_actions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO training"""
        actor_logits, values = self.forward(input_ids, attention_mask)
        
        # Get logits for the positions where we took actions
        batch_size, seq_len = actions.shape
        
        # Reshape for easier indexing
        flat_logits = actor_logits.view(-1, self.vocab_size)  # [batch*seq, vocab]
        flat_actions = actions.view(-1)  # [batch*seq]
        
        # Create position indices
        position_indices = torch.arange(batch_size * seq_len, device=actions.device)
        
        # Get log probabilities for taken actions
        log_probs = F.log_softmax(flat_logits, dim=-1)
        action_log_probs = log_probs[position_indices, flat_actions].view(batch_size, seq_len)
        
        # Calculate entropy
        probs = F.softmax(flat_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).view(batch_size, seq_len)
        
        return action_log_probs, values, entropy


class ConversationDataset:
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        self.load_data()
    
    def load_data(self):
        """Load conversation data"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'conversations' in data:
                            # Multi-turn conversation format
                            conversations = data['conversations']
                            if len(conversations) >= 2:
                                for i in range(0, len(conversations) - 1, 2):
                                    if i + 1 < len(conversations):
                                        human_msg = conversations[i].get('value', '')
                                        assistant_msg = conversations[i + 1].get('value', '')
                                        if human_msg and assistant_msg:
                                            self.conversations.append({
                                                'prompt': human_msg,
                                                'response': assistant_msg,
                                                'quality': data.get('quality', 1.0)
                                            })
                        elif 'prompt' in data and 'response' in data:
                            # Simple prompt-response format
                            self.conversations.append({
                                'prompt': data['prompt'],
                                'response': data['response'],
                                'quality': data.get('quality', 1.0)
                            })
                    except (json.JSONDecodeError, KeyError):
                        continue
        except FileNotFoundError:
            print(f"Warning: Data file {self.data_path} not found. Using dummy data.")
            # Create more realistic dummy conversations
            self.conversations = [
                {'prompt': 'Hello! How are you today?', 'response': 'I am doing well, thank you for asking! How can I help you today?', 'quality': 1.0},
                {'prompt': 'What is machine learning?', 'response': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.', 'quality': 1.0},
                {'prompt': 'Can you help me write a poem?', 'response': 'Of course! I would be happy to help you write a poem. What topic or theme would you like the poem to be about?', 'quality': 0.9},
                {'prompt': 'Tell me a fun fact', 'response': 'Here is a fun fact: Octopuses have three hearts and blue blood! Two hearts pump blood to the gills, while the third pumps blood to the rest of the body.', 'quality': 0.95},
                {'prompt': 'How do I cook pasta?', 'response': 'To cook pasta: 1) Boil water with salt, 2) Add pasta and stir occasionally, 3) Cook for the time specified on the package, 4) Drain and serve with your favorite sauce!', 'quality': 1.0},
                {'prompt': 'What is the weather like?', 'response': 'I do not have access to real-time weather information. I recommend checking a weather app or website for current conditions in your area.', 'quality': 0.8},
            ]
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Create conversation format: "Human: {prompt} Assistant: {response}"
        full_text = f"Human: {conv['prompt']} Assistant: {conv['response']}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Find the split point between prompt and response
        prompt_encoded = self.tokenizer(
            f"Human: {conv['prompt']} Assistant:",
            max_length=self.max_length // 2,
            truncation=True,
            return_tensors='pt'
        )
        
        prompt_length = prompt_encoded['input_ids'].shape[1]
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'prompt_length': prompt_length,
            'quality_score': torch.tensor(conv['quality'], dtype=torch.float32),
            'prompt_text': conv['prompt'],
            'response_text': conv['response'],
            'full_text': full_text
        }