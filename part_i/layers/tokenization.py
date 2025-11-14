import torch

class WordLevelTokenizer:
    """
    Simple word-level tokenizer implemented from scratch
    """
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1, 
            '<eos>': 2,
            '<unk>': 3
        }
        
    def build_vocab(self, texts, max_vocab_size=10000):
        """
        Build vocabulary from list of texts
        """
        word_freq = {}
        
        # Count word frequencies
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize with special tokens
        self.word_to_idx = self.special_tokens.copy()
        self.idx_to_word = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
        
        # Add most frequent words
        for word, freq in sorted_words[:max_vocab_size - len(self.special_tokens)]:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def _tokenize_text(self, text):
        """
        Simple word tokenization - split by space and basic punctuation
        """
        # Convert to lowercase and split
        words = text.lower().split()
        # Remove empty strings and basic cleaning
        words = [word.strip('.,!?;:"()[]') for word in words if word.strip()]
        return words
    
    def encode(self, text, max_length=None):
        """
        Convert text to token indices
        """
        words = self._tokenize_text(text)
        
        # Add SOS token
        tokens = [self.word_to_idx['<sos>']]
        
        # Convert words to indices
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx['<unk>'])
        
        # Add EOS token
        tokens.append(self.word_to_idx['<eos>'])
        
        # Pad if needed
        if max_length and len(tokens) < max_length:
            tokens.extend([self.word_to_idx['<pad>']] * (max_length - len(tokens)))
        elif max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            tokens[-1] = self.word_to_idx['<eos>']  # Ensure EOS at end
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens):
        """
        Convert token indices back to text
        """
        words = []
        for token in tokens:
            if token.item() in self.idx_to_word:
                word = self.idx_to_word[token.item()]
                if word not in ['<sos>', '<eos>', '<pad>']:
                    words.append(word)
        return ' '.join(words)

def get_tokens(text: str, tokenizer: WordLevelTokenizer, max_length=64):
    """
    Takes an input string and returns its token IDs using our custom tokenizer
    """
    return tokenizer.encode(text, max_length).unsqueeze(0)  # Add batch dimension

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