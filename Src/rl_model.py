#!/usr/bin/env python3
"""
RL Chat Model - Reinforcement Learning Model for Conversational AI

This module contains the main model definition and dataset loader for
training a conversational AI using reinforcement learning with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch.distributions import Categorical
import tiktoken
import json
import random
import numpy as np

class RLChatModel(nn.Module):
    def __init__(self, base_model_name="distilbert-base-uncased", tiktoken_encoding="cl100k_base", hidden_dim=768, max_length=512):
        super(RLChatModel, self).__init__()
        
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        
        # Load tiktoken tokenizer
        self.tokenizer = tiktoken.get_encoding(tiktoken_encoding)
        self.vocab_size = self.tokenizer.n_vocab
        
        # Special tokens
        self.pad_token_id = self.tokenizer.encode("<|pad|>")[0] if "<|pad|>" in self.tokenizer.decode_tokens_bytes([i for i in range(min(1000, self.vocab_size))]) else 0
        self.eos_token_id = self.tokenizer.encode("<|endoftext|>")[0] if "<|endoftext|>" in str(self.tokenizer.decode([i for i in range(min(100, self.vocab_size))])) else 50256
        
        # Alternative: use common token IDs for GPT tokenizers
        try:
            # Try to find endoftext token
            test_tokens = self.tokenizer.encode("<|endoftext|>")
            if test_tokens:
                self.eos_token_id = test_tokens[0]
            else:
                self.eos_token_id = 50256  # Common endoftext token ID for GPT tokenizers
        except:
            self.eos_token_id = 50256
            
        try:
            # For padding, we'll use a less common token or create a special handling
            self.pad_token_id = self.eos_token_id  # Use same as EOS for simplicity
        except:
            self.pad_token_id = 0
            
        print(f"Vocab size: {self.vocab_size}, EOS token ID: {self.eos_token_id}, PAD token ID: {self.pad_token_id}")
        
        # Load base transformer (we'll use it for embeddings and basic architecture)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Create embedding layer that matches our tokenizer vocab size
        self.embeddings = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=self.pad_token_id)
        
        # Initialize embeddings from base model if possible
        if hasattr(self.base_model, 'embeddings') and hasattr(self.base_model.embeddings, 'word_embeddings'):
            base_vocab_size = self.base_model.embeddings.word_embeddings.num_embeddings
            with torch.no_grad():
                # Copy what we can from the base model embeddings
                copy_size = min(base_vocab_size, self.vocab_size)
                self.embeddings.weight[:copy_size] = self.base_model.embeddings.word_embeddings.weight[:copy_size]
        
        # Transformer layers (we'll use the base model's transformer layers)
        self.transformer = self.base_model
        
        # Policy head (actor) - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.vocab_size)
        )
        
        # Value head (critic) - estimates state values
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_text(self, text, max_length=None):
        """Encode text using tiktoken"""
        if max_length is None:
            max_length = self.max_length
            
        tokens = self.tokenizer.encode(text)
        
        # Truncate if too long
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad if too short
        while len(tokens) < max_length:
            tokens.append(self.pad_token_id)
        
        return {
            'input_ids': torch.tensor([tokens]),
            'attention_mask': torch.tensor([[1 if token != self.pad_token_id else 0 for token in tokens]])
        }
    
    def decode_tokens(self, tokens):
        """Decode tokens using tiktoken"""
        # Remove padding tokens
        tokens = [t for t in tokens if t != self.pad_token_id]
        try:
            return self.tokenizer.decode(tokens)
        except:
            # Fallback for invalid tokens
            valid_tokens = [t for t in tokens if 0 <= t < self.vocab_size]
            return self.tokenizer.decode(valid_tokens)
        
    def forward(self, input_ids, attention_mask=None, return_dict=False):
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        embeddings = self.embeddings(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Pass through transformer layers (we need to adapt this for our embeddings)
        # For simplicity, we'll use a basic approach
        hidden_states = embeddings
        
        # Apply a simple transformer-like processing
        # Note: This is simplified - you might want to use actual transformer layers
        hidden_states = hidden_states + self.positional_encoding(seq_len, hidden_states.device)
        
        # Mean pooling for sequence representation
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        masked_hidden_states = hidden_states * mask_expanded
        pooled_output = masked_hidden_states.sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Get policy logits and values
        policy_logits = self.policy_head(pooled_output)
        values = self.value_head(pooled_output).squeeze(-1)
        
        if return_dict:
            return {
                'policy_logits': policy_logits,
                'values': values,
                'hidden_states': hidden_states
            }
        
        return policy_logits, values
    
    def positional_encoding(self, seq_len, device):
        """Simple positional encoding"""
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2, dtype=torch.float, device=device) *
                           -(np.log(10000.0) / self.hidden_dim))
        
        pe = torch.zeros(seq_len, self.hidden_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:self.hidden_dim//2])
        
        return pe.unsqueeze(0)
    
    def act(self, input_ids, attention_mask=None, deterministic=False):
        """Sample action from policy"""
        with torch.no_grad():
            policy_logits, values = self.forward(input_ids, attention_mask)
            
            if deterministic:
                actions = torch.argmax(policy_logits, dim=-1)
                log_probs = F.log_softmax(policy_logits, dim=-1)
                action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            else:
                dist = Categorical(logits=policy_logits)
                actions = dist.sample()
                action_log_probs = dist.log_prob(actions)
            
            return actions, action_log_probs, values
    
    def evaluate_actions(self, input_ids, attention_mask, actions):
        """Evaluate actions for PPO training"""
        policy_logits, values = self.forward(input_ids, attention_mask)
        
        dist = Categorical(logits=policy_logits)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_probs, values, entropy
    
    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7):
        """Generate a response given a prompt"""
        self.eval()
        
        # Tokenize prompt
        inputs = self.encode_text(prompt, max_length=self.max_length//2)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get next token probabilities
                policy_logits, _ = self.forward(input_ids, attention_mask)
                
                # Apply temperature
                logits = policy_logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Check for end token
                if next_token.item() == self.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                
                # Update input for next iteration
                if input_ids.shape[1] < self.max_length:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
                else:
                    # Shift the sequence
                    input_ids = torch.cat([input_ids[:, 1:], next_token], dim=1)
        
        # Decode response
        if generated_tokens:
            response = self.decode_tokens(generated_tokens)
        else:
            response = ""
            
        return response


class OASST2Dataset:
    def __init__(self, data_path, model, max_length=512):
        self.data_path = data_path
        self.model = model  # We need the model to access the tokenizer
        self.max_length = max_length
        self.conversations = []
        self.load_data()
    
    def load_data(self):
        """Load OASST2 format data"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data:
                            # Simple format: just text
                            self.conversations.append({
                                'prompt': '',
                                'response': data['text'],
                                'quality_score': data.get('quality', 1.0)
                            })
                        elif 'messages' in data:
                            # Conversation format
                            messages = data['messages']
                            if len(messages) >= 2:
                                prompt = messages[-2].get('content', '')
                                response = messages[-1].get('content', '')
                                self.conversations.append({
                                    'prompt': prompt,
                                    'response': response,
                                    'quality_score': data.get('quality', 1.0)
                                })
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Warning: Data file {self.data_path} not found. Using dummy data.")
            # Create some dummy conversations for testing
            self.conversations = [
                {'prompt': 'Hello', 'response': 'Hi there! How can I help you today?', 'quality_score': 1.0},
                {'prompt': 'What is AI?', 'response': 'AI stands for Artificial Intelligence, which refers to computer systems that can perform tasks typically requiring human intelligence.', 'quality_score': 1.0},
                {'prompt': 'How are you?', 'response': 'I am doing well, thank you for asking! How are you doing?', 'quality_score': 0.8},
                {'prompt': 'Tell me a joke', 'response': 'Why did the computer go to the doctor? Because it had a virus!', 'quality_score': 0.9},
                {'prompt': 'Explain quantum physics', 'response': 'Quantum physics studies the behavior of matter and energy at the smallest scales, where particles can exist in multiple states simultaneously.', 'quality_score': 1.0},
            ]
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Tokenize prompt and response using our model's tokenizer
        prompt_encoded = self.model.encode_text(conv['prompt'], max_length=self.max_length//2)
        response_encoded = self.model.encode_text(conv['response'], max_length=self.max_length//2)
        
        return {
            'prompt_ids': prompt_encoded['input_ids'].squeeze(0),
            'prompt_mask': prompt_encoded['attention_mask'].squeeze(0),
            'response_ids': response_encoded['input_ids'].squeeze(0),
            'response_mask': response_encoded['attention_mask'].squeeze(0),
            'quality_score': torch.tensor(conv['quality_score'], dtype=torch.float32),
            'prompt_text': conv['prompt'],
            'response_text': conv['response']
        }