#!/usr/bin/env python3
"""
RL Chat Model - Reinforcement Learning Model for Conversational AI

This module contains the main model definition and dataset loader for
training a conversational AI using reinforcement learning with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.distributions import Categorical
import json
import random
import numpy as np

class RLChatModel(nn.Module):
    def __init__(self, base_model_name="distilbert-base-uncased", vocab_size=30522, hidden_dim=768, max_length=512):
        super(RLChatModel, self).__init__()
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Load base transformer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModel.from_pretrained(base_model_name)
        
        # Policy head (actor) - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Value head (critic) - estimates state values
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, input_ids, attention_mask=None, return_dict=False):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use [CLS] token representation or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Mean pooling over sequence length
            pooled_output = hidden_states.mean(dim=1)
        
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
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length//2)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get next token probabilities
                policy_logits, _ = self.forward(input_ids, attention_mask)
                
                # Apply temperature
                logits = policy_logits[:, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Check for end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                generated_tokens.append(next_token.item())
                
                # Update input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1)], dim=1)
                
                # Prevent infinite generation
                if input_ids.shape[1] >= self.max_length:
                    break
        
        # Decode response
        if generated_tokens:
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = ""
            
        return response


class OASST2Dataset:
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
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
            ]
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Tokenize prompt and response
        prompt_tokens = self.tokenizer(
            conv['prompt'], 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length//2
        )
        
        response_tokens = self.tokenizer(
            conv['response'], 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length//2
        )
        
        return {
            'prompt_ids': prompt_tokens['input_ids'].squeeze(0),
            'prompt_mask': prompt_tokens['attention_mask'].squeeze(0),
            'response_ids': response_tokens['input_ids'].squeeze(0),
            'response_mask': response_tokens['attention_mask'].squeeze(0),
            'quality_score': torch.tensor(conv['quality_score'], dtype=torch.float32),
            'prompt_text': conv['prompt'],
            'response_text': conv['response']
        }