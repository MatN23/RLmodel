#!/usr/bin/env python3

import torch
import json
import os
from datetime import datetime
import logging

# Import our model (assuming it's in the same directory)
from rl_model import RLChatModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatInterface:
    def __init__(self, model_path, base_model='distilbert-base-uncased', tiktoken_encoding='cl100k_base', max_length=512):
        """Initialize chat interface with trained model"""
        self.model = RLChatModel(
            base_model_name=base_model,
            tiktoken_encoding=tiktoken_encoding,
            max_length=max_length
        )
        
        # Load trained model
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Print training info if available
            if 'epoch' in checkpoint:
                logger.info(f"Model trained for {checkpoint['epoch'] + 1} epochs")
            if 'best_reward' in checkpoint:
                logger.info(f"Best validation reward: {checkpoint['best_reward']:.4f}")
            if 'vocab_size' in checkpoint:
                logger.info(f"Model vocab size: {checkpoint['vocab_size']}")
            if 'tokenizer_encoding' in checkpoint:
                logger.info(f"Tokenizer encoding: {checkpoint['tokenizer_encoding']}")
        else:
            logger.warning(f"Model file {model_path} not found. Using untrained model.")
        
        self.model.eval()
        self.conversation_history = []
    
    def generate_response(self, user_input, max_new_tokens=100, temperature=0.7, use_history=True):
        """Generate response to user input"""
        # Prepare context with conversation history
        if use_history and self.conversation_history:
            # Include recent conversation context
            context_parts = []
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                context_parts.append(f"Human: {entry['user']}")
                context_parts.append(f"Assistant: {entry['assistant']}")
            context_parts.append(f"Human: {user_input}")
            context_parts.append("Assistant:")
            
            prompt = " ".join(context_parts)
        else:
            prompt = f"Human: {user_input}\nAssistant:"
        
        # Generate response
        try:
            response = self.model.generate_response(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = "I'm sorry, I encountered an error generating a response. Could you try again?"
        
        # Clean up response
        response = response.strip()
        if not response:
            response = "I'm not sure how to respond to that. Could you try rephrasing?"
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def interactive_chat(self, max_new_tokens=100, temperature=0.7, use_history=True):
        """Start interactive chat session"""
        print("=== RL Chat Model (Tiktoken) - Interactive Session ===")
        print(f"Tokenizer: {self.model.tokenizer}")
        print(f"Vocab size: {self.model.vocab_size}")
        print(f"EOS token ID: {self.model.eos_token_id}")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'clear' to clear conversation history")
        print("Type 'save <filename>' to save conversation history")
        print("Type 'test' to run a tokenization test")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Thanks for chatting.")
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared.")
                    continue
                elif user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    self.save_conversation(filename)
                    continue
                elif user_input.lower() == 'test':
                    self.run_tokenization_test()
                    continue
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  quit/exit/bye - End session")
                    print("  clear - Clear conversation history") 
                    print("  save <filename> - Save conversation")
                    print("  test - Run tokenization test")
                    print("  help - Show this help")
                    continue
                
                # Generate and display response
                print("\nAssistant: ", end="", flush=True)
                
                response = self.generate_response(
                    user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_history=use_history
                )
                
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError generating response: {e}")
                logger.error(f"Error in interactive chat: {e}")
    
    def run_tokenization_test(self):
        """Test tokenization with sample text"""
        test_texts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "The quick brown fox jumps over the lazy dog.",
            "🤖 AI is fascinating! 🚀"
        ]
        
        print("\n--- Tokenization Test ---")
        for text in test_texts:
            try:
                encoded = self.model.encode_text(text)
                tokens = encoded['input_ids'].squeeze().tolist()
                decoded = self.model.decode_tokens(tokens)
                
                print(f"Original: {text}")
                print(f"Tokens: {tokens}")
                print(f"Decoded: {decoded}")
                print(f"Token count: {len([t for t in tokens if t != self.model.pad_token_id])}")
                print("-" * 40)
            except Exception as e:
                print(f"Error tokenizing '{text}': {e}")
                print("-" * 40)
    
    def batch_chat(self, prompts, max_new_tokens=100, temperature=0.7):
        """Process multiple prompts in batch"""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
            
            response = self.generate_response(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                use_history=False  # Don't use history for batch processing
            )
            
            results.append({
                'prompt': prompt,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Response: {response}")
        
        return results
    
    def save_conversation(self, filename):
        """Save conversation history to file"""
        try:
            conversation_data = {
                'conversation_history': self.conversation_history,
                'model_info': {
                    'vocab_size': self.model.vocab_size,
                    'eos_token_id': self.model.eos_token_id,
                    'pad_token_id': self.model.pad_token_id,
                    'max_length': self.model.max_length
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            print(f"Conversation saved to {filename}")
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def load_conversation(self, filename):
        """Load conversation history from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'conversation_history' in data:
                self.conversation_history = data['conversation_history']
                print(f"Conversation loaded from {filename}")
                print(f"Loaded {len(self.conversation_history)} exchanges")
                
                if 'model_info' in data:
                    print("Model info from file:")
                    for key, value in data['model_info'].items():
                        print(f"  {key}: {value}")
            else:
                # Legacy format - compatible with original chat.py saves
                self.conversation_history = data
                print(f"Conversation loaded from {filename} (legacy format)")
                print(f"Loaded {len(self.conversation_history)} exchanges")
                
        except Exception as e:
            print(f"Error loading conversation: {e}")


def main():
    # Hardcoded configuration - updated for tiktoken
    model_path = 'rl_chat_model_tiktoken.pt'  # Updated model path
    base_model = 'distilbert-base-uncased'
    tiktoken_encoding = 'cl100k_base'  # GPT-4 encoding
    max_length = 512
    max_new_tokens = 100
    temperature = 0.7
    use_history = True
    mode = 'interactive'  # or 'batch'
    
    # Initialize chat interface with tiktoken support
    chat = ChatInterface(model_path, base_model, tiktoken_encoding, max_length)
    
    if mode == 'interactive':
        # Start interactive chat
        chat.interactive_chat(max_new_tokens, temperature, use_history)
        
    elif mode == 'batch':
        # Example batch prompts - modify as needed
        prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Can you help me with a problem?",
            "Tell me a joke",
            "What's the weather like?"
        ]
        
        results = chat.batch_chat(prompts, max_new_tokens, temperature)
        
        # Save results with tiktoken suffix
        output_file = 'batch_results_tiktoken.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nBatch results saved to {output_file}")
    
    else:
        print(f"Unknown mode: {mode}. Use 'interactive' or 'batch'")


if __name__ == "__main__":
    main()