import torch
import torch.nn as nn
import math
import fasttext
from datasets import load_dataset
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from layers.transformer_block import *
from layers.embedding import *
from layers.tokenization import *

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length=512, dropout=0.1, fasttext_embeddings=None):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        # Input embedding with FastText
        self.input_embedding = InputEmbedding(vocab_size, d_model, max_seq_length, fasttext_embeddings)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = LayerNorm(d_model)
        
        # Output projection
        self.output_linear = Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
        return mask
    
    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len)
        
        # Input embedding
        x = self.input_embedding(input_tokens)
        
        # Store attention weights for visualization
        attention_weights = []
        
        # Pass through each transformer block
        for layer in self.layers:
            x, attn_weights = layer(x, causal_mask)
            attention_weights.append(attn_weights)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_linear(x)
        
        return logits, attention_weights
    
    def generate(self, prompt, tokenizer, max_length=64, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            # Encode prompt
            input_ids = tokenizer.encode(prompt, max_length=max_length).unsqueeze(0)
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == tokenizer.word_to_idx['<eos>']:
                    break
            
            return generated

class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, self.max_length)
        return tokens

def train_model():
    # Load dataset
    ds = load_dataset("roneneldan/TinyStories")
    
    # Initialize tokenizer
    tokenizer = WordLevelTokenizer()
    tokenizer.build_vocab(ds['train']['text'][:10000], max_vocab_size=10000)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Model configuration (as per assignment)
    d_model = 300  # FastText dimension
    num_layers = 3
    num_heads = 8
    d_ff = 1024
    max_seq_length = 64
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 3
    
    # Load FastText embeddings
    try:
        ft_model = fasttext.load_model("cc.en.300.bin")
        fasttext_embeddings = build_fasttext_embedding_matrix(tokenizer, ft_model, d_model)
        print("FastText embeddings loaded successfully")
    except:
        print("FastText model not found, using random embeddings")
        fasttext_embeddings = None
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        fasttext_embeddings=fasttext_embeddings
    )
    
    # Create datasets
    train_dataset = TinyStoriesDataset(ds['train']['text'][:1000], tokenizer, max_seq_length)
    val_dataset = TinyStoriesDataset(ds['validation']['text'][:200], tokenizer, max_seq_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_idx['<pad>'])
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Teacher forcing: input is all tokens except last, target is all tokens except first
            input_tokens = batch[:, :-1]
            target_tokens = batch[:, 1:]
            
            logits, _ = model(input_tokens)
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_tokens = batch[:, :-1]
                target_tokens = batch[:, 1:]
                
                logits, _ = model(input_tokens)
                loss = criterion(logits.reshape(-1, vocab_size), target_tokens.reshape(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Generate sample text
        if epoch % 1 == 0:
            prompt = "Once upon a time"
            generated = model.generate(prompt, tokenizer, max_length=50, temperature=0.8)
            generated_text = tokenizer.decode(generated[0])
            print(f"Generated: {generated_text}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curves.png')
    plt.show()
    
    # Calculate perplexity
    final_perplexity = math.exp(val_losses[-1])
    print(f"Final Validation Perplexity: {final_perplexity:.2f}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Test the model components
    tokenizer = WordLevelTokenizer()
    ds = load_dataset("roneneldan/TinyStories")
    print("Dataset columns:", ds.column_names)
    
    tokenizer.build_vocab(ds['train']['text'][:1000], max_vocab_size=1000)
    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)
    
    d_model = 300
    batch_size = 4
    seq_length = 64
    
    # Test input
    input_tokens = torch.zeros((batch_size, seq_length), dtype=torch.long)
    for i in range(batch_size):
        token_ids = get_tokens(ds['validation']['text'][i], tokenizer, max_length=seq_length).squeeze(0)
        input_tokens[i] = token_ids
    
    print(f"Input tokens shape: {input_tokens.shape}")
    
    # Test with FastText if available, else random
    try:
        ft_model = fasttext.load_model("cc.en.300.bin")
        fasttext_embeddings = build_fasttext_embedding_matrix(tokenizer, ft_model, d_model)
        print("Using FastText embeddings")
    except:
        fasttext_embeddings = None
        print("Using random embeddings")
    
    # Test complete model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_length=seq_length,
        fasttext_embeddings=fasttext_embeddings
    )
    
    logits, attention_weights = model(input_tokens)
    print(f"Model output logits shape: {logits.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    print(f"Attention weights shape: {attention_weights[0].shape}")
    
    # Test generation
    prompt = "5+2="
    generated = model.generate(prompt, tokenizer, max_length=20, temperature=0.8)
    generated_text = tokenizer.decode(generated[0])
    print(f"Generated text: {generated_text}")
    
    # Uncomment to start training
    # print("\nStarting training...")
    # trained_model, trained_tokenizer = train_model()