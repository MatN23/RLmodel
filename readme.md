# RL Chat Model with Tiktoken

A reinforcement learning-based conversational AI model that uses Proximal Policy Optimization (PPO) to train a chatbot capable of generating contextually appropriate responses. This implementation leverages tiktoken tokenization for improved token handling and supports both training and interactive chat functionality.

## Overview

This project implements a complete pipeline for training and deploying a conversational AI model using reinforcement learning techniques. The model combines a transformer-based architecture with PPO training to learn from conversation data and generate human-like responses.

### Key Features

- **Reinforcement Learning Training**: Uses PPO (Proximal Policy Optimization) for training
- **Tiktoken Integration**: Modern tokenization using OpenAI's tiktoken library
- **Interactive Chat Interface**: Real-time conversation capabilities
- **Comprehensive Metrics**: Training monitoring with accuracy, perplexity, and reward tracking
- **Flexible Architecture**: Modular design supporting different base models
- **Conversation Management**: History tracking and context-aware responses

## Architecture

The system consists of three main components:

### 1. RL Model (`rl_model.py`)
- **RLChatModel**: Main neural network combining transformer layers with policy and value heads
- **Policy Head**: Generates action probabilities for next token prediction
- **Value Head**: Estimates state values for PPO training
- **Tiktoken Integration**: Modern tokenization with vocabulary management
- **OASST2Dataset**: Data loader for conversation datasets

### 2. Training Pipeline (`train.py`)
- **PPOTrainer**: Implements Proximal Policy Optimization algorithm
- **Reward Function**: Multi-factor reward calculation based on response quality
- **Metrics Tracking**: Comprehensive monitoring of training progress
- **Model Checkpointing**: Automatic saving of best-performing models
- **Training Visualization**: Automatic generation of training statistics plots

### 3. Chat Interface (`chat.py`)
- **Interactive Chat**: Real-time conversation interface
- **Batch Processing**: Multi-prompt processing capabilities
- **Conversation History**: Context-aware response generation
- **Model Testing**: Built-in tokenization and model testing tools
- **Data Persistence**: Save and load conversation histories

## Installation

### Requirements

```bash
pip install torch transformers tiktoken numpy matplotlib tqdm
```

### Dependencies
- Python 3.7+
- PyTorch
- Transformers
- Tiktoken
- NumPy
- Matplotlib
- tqdm

## Usage

### Training a New Model

1. **Prepare your dataset**: Format your training data as JSONL with conversation pairs
2. **Configure training parameters** in `train.py`:
   ```python
   data_path = 'your_dataset.jsonl'
   base_model = 'distilbert-base-uncased'
   tiktoken_encoding = 'cl100k_base'
   max_length = 512
   ```
3. **Run training**:
   ```bash
   python train.py
   ```

### Using the Trained Model

**Interactive Chat**:
```bash
python chat.py
```

**Programmatic Usage**:
```python
from chat import ChatInterface

chat = ChatInterface('rl_chat_model_tiktoken.pt')
response = chat.generate_response("Hello, how are you?")
print(response)
```

## Configuration Options

### Model Parameters
- `base_model`: Base transformer model (default: 'distilbert-base-uncased')
- `tiktoken_encoding`: Tokenizer encoding (default: 'cl100k_base')
- `max_length`: Maximum sequence length (default: 512)
- `hidden_dim`: Hidden layer dimensions (default: 768)

### Training Parameters
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 1e-4)
- `num_epochs`: Number of training epochs (default: 10)
- `clip_eps`: PPO clipping parameter (default: 0.2)

### Generation Parameters
- `max_new_tokens`: Maximum tokens to generate (default: 100)
- `temperature`: Sampling temperature (default: 0.7)
- `use_history`: Enable conversation history (default: True)

## Dataset Format

The model expects conversation data in JSONL format:

```json
{"messages": [{"content": "Hello"}, {"content": "Hi there! How can I help?"}], "quality": 1.0}
{"text": "Single response text", "quality": 0.8}
```

## Training Process

1. **Data Loading**: Loads conversation pairs from JSONL files
2. **Tokenization**: Converts text to tokens using tiktoken
3. **Experience Collection**: Generates responses and computes rewards
4. **PPO Training**: Updates policy using clipped objective function
5. **Validation**: Evaluates model performance on held-out data
6. **Checkpointing**: Saves best-performing models automatically

## Reward Function

The reward system evaluates responses based on multiple factors:
- **Base Quality**: Human-annotated quality scores
- **Length Penalties**: Discourages overly short responses
- **Length Bonuses**: Rewards appropriate response lengths
- **Diversity Rewards**: Encourages varied vocabulary usage

## Interactive Commands

When using the chat interface:
- `quit`/`exit`/`bye`: End the session
- `clear`: Clear conversation history
- `save <filename>`: Save conversation to file
- `test`: Run tokenization diagnostics
- `help`: Display available commands

## Model Output

The trained model generates:
- **Saved Model**: `rl_chat_model_tiktoken.pt`
- **Training Stats**: `*_stats.png` visualization plots
- **Conversation Logs**: JSON files with chat history

## Performance Metrics

The system tracks multiple metrics during training:
- **Policy Loss**: PPO policy optimization loss
- **Value Loss**: Value function approximation error
- **Entropy**: Policy exploration measure
- **Reward**: Average reward per conversation
- **Accuracy**: Token prediction accuracy
- **Perplexity**: Language model quality measure

## Limitations and Considerations

- **Memory Usage**: Large vocabulary sizes require significant RAM
- **Training Time**: RL training can be computationally intensive
- **Data Requirements**: Benefits from high-quality conversation datasets
- **Hyperparameter Sensitivity**: PPO parameters may need tuning for optimal performance

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Consider using GPU acceleration
3. **Poor Responses**: Check reward function and training data quality
4. **Tokenization Errors**: Verify tiktoken encoding compatibility

### Debug Features
- Built-in tokenization testing (`test` command)
- Comprehensive logging throughout training
- Sample response generation during training
- Training statistics visualization

## Future Enhancements

Potential improvements and extensions:
- Multi-turn conversation optimization
- Advanced reward shaping techniques
- Integration with larger language models
- Real-time learning capabilities
- Multi-modal conversation support

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with the licenses of all dependencies, particularly the base transformer models and tokenization libraries.