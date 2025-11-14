import torch
import torch.nn as nn
import math
import numpy as np
import fasttext
# from tokenization import *


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length=512, fasttext_embeddings=None):
        """
        Input Embedding Layer with Positional Encoding - Everything from scratch
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of model embeddings
            max_seq_length: Maximum sequence length for positional encoding
            fasttext_embeddings: Pre-trained FastText embeddings (optional)
        """
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Initialize embedding layer
        if fasttext_embeddings is not None:
            print("Using pre-trained FastText embeddings for initialization.")
            # Convert to tensor if numpy array
            if isinstance(fasttext_embeddings, np.ndarray):
                fasttext_embeddings = torch.FloatTensor(fasttext_embeddings)
            
            # Ensure vocabulary size matches
            assert fasttext_embeddings.shape[0] >= vocab_size, "FastText embeddings don't cover full vocabulary"
            
            # Use pre-trained FastText embeddings
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            # Initialize with FastText embeddings
            with torch.no_grad():
                self.token_embedding.weight[:vocab_size] = fasttext_embeddings[:vocab_size]
        else:
            # Random initialization (fallback)
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            # Initialize with normal distribution as in original transformer
            self._init_weights()
        
        # Create positional encoding matrix
        self.register_buffer('positional_encoding', self.create_positional_encoding(max_seq_length, d_model))
        self.dropout = nn.Dropout(0.1)
        
    def _init_weights(self):
        """Initialize embedding weights with normal distribution"""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
    def create_positional_encoding(self, max_seq_length, d_model):
        """
        Create sinusoidal positional encoding as in Vaswani et al. 2017 - from scratch
        """
        positional_encoding = torch.zeros(max_seq_length, d_model)
        
        positions = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        # Apply cosine to odd indices  
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)
        
        # Add batch dimension for broadcasting: (1, max_seq_length, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)
        
        return positional_encoding
    
    def forward(self, input_tokens):
        """
        Forward pass for input embedding
        
        Args:
            input_tokens: Tensor of shape (batch_size, seq_length) containing token indices
            
        Returns:
            embedded_output: Tensor of shape (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = input_tokens.shape
        
        # Ensure sequence length doesn't exceed maximum
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds maximum {self.max_seq_length}")
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_tokens)  # (batch_size, seq_length, d_model)
        
        # Scale embeddings by sqrt(d_model) as in original paper
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Add positional encoding
        # positional_encoding: (1, max_seq_length, d_model) -> take first seq_length positions
        positional_embeds = self.positional_encoding[:, :seq_length, :]
        
        # Combine token embeddings and positional encoding
        embedded_output = token_embeds + positional_embeds
        
        # Apply dropout
        embedded_output = self.dropout(embedded_output)
        
        return embedded_output

# Example usage function
def create_input_embedding_layer(vocab_size, d_model, fasttext_embeddings):
    """
    Helper function to create the input embedding layer
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        fasttext_embeddings: Pre-trained FastText embeddings (optional)
    
    Returns:
        embedding_layer: InputEmbedding instance
    """
    return InputEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_length=512,
        fasttext_embeddings=fasttext_embeddings
    )

def build_fasttext_embedding_matrix(tokenizer, ft_model, d_model):
    vocab_size = tokenizer.vocab_size
    embedding_matrix = np.zeros((vocab_size, d_model))

    # For each word in tokenizer vocab
    for word, idx in tokenizer.word_to_idx.items():
        if word in tokenizer.special_tokens:  # Handle special tokens
            embedding_matrix[idx] = np.zeros(d_model)
        else:
            embedding_matrix[idx] = ft_model.get_word_vector(word)
    return embedding_matrix

# Example of how to use:
if __name__ == "__main__":
    # Create and train a simple tokenizer
    tokenizer = WordLevelTokenizer()
    # sample_texts = [
    #     "Hello, how are you?",
    #     "I am learning about transformers",
    #     "This is a decoder only model",
    #     "We are implementing everything from scratch"
    # ]
    # tokenizer.build_vocab(sample_texts, max_vocab_size=1000)

    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    print(ds.column_names)
    # print(len(ds['train']))
    # print(ds['train'][:10])
    tokenizer.build_vocab(ds['train']['text'][:10000], max_vocab_size=10000)
    
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)
    # Match FastText dimension
    d_model = 300
    batch_size = 4
    seq_length = 64

    # Test with actual text
    # sample_text = "Hello, I am Animesh Lohar"
    # token_ids = get_tokens(sample_text, tokenizer, max_length=seq_length)
    # print("Token IDs:", token_ids)
    # print("Decoded:", tokenizer.decode(token_ids[0]))
    input_tokens = torch.zeros((batch_size, seq_length), dtype=torch.long)
    for i in range(batch_size):
        token_ids = get_tokens(ds['validation']['text'][i], tokenizer, max_length=seq_length)
        input_tokens[i] = token_ids

    # Create batch of tokens
    # input_tokens = token_ids.repeat(batch_size, 1)
    print(f"Input tokens: {input_tokens}")

    # Load pre-trained English FastText word vectors (300-dim)
    ft_model = fasttext.load_model("./cc.en.300.bin")
    fasttext_embeddings = build_fasttext_embedding_matrix(tokenizer, ft_model, d_model)

    # Create embedding layer
    embedding_layer = create_input_embedding_layer(vocab_size, d_model, fasttext_embeddings)

    # Get embeddings
    embedded_output = embedding_layer(input_tokens)
    print("Embedded output:", embedded_output)
    # embedded_plus_positional_output = embedding_layer.forward(input_tokens)
    # print("Embedded + Positional output:", embedded_plus_positional_output)
    # if(embedded_output == embedded_plus_positional_output).all():
    #     print("Forward method works correctly!")
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Embedded output shape: {embedded_output.shape}")
    # print(f"Embedded + Positional output shape: {embedded_plus_positional_output.shape}")