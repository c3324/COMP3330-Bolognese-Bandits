import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Encoder import EncoderBlock

HIDDEN_SIZE = 32
EMBEDDING_SIZE = 4
ENCODING_COUNT = 2
MAX_LEN = 600
HEADS = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))

class Control(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=EMBEDDING_SIZE, padding_idx=vocab['<pad>'])
        self.position_embedding = nn.Embedding(
            num_embeddings=MAX_LEN, embedding_dim=EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embed=EMBEDDING_SIZE, n_head=HEADS) for _ in range(ENCODING_COUNT)])
        self.ln = nn.LayerNorm(EMBEDDING_SIZE)
        self.out = nn.Linear(EMBEDDING_SIZE, 6)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        x = torch.amax(x, dim=1) # Reduce sequence dim (B, C)
        x = self.out(x)
        return x
    
class ExtraEmbedding(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=EMBEDDING_SIZE*2, padding_idx=vocab['<pad>'])
        self.position_embedding = nn.Embedding(
            num_embeddings=MAX_LEN, embedding_dim=EMBEDDING_SIZE*2)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embed=EMBEDDING_SIZE*2, n_head=HEADS) for _ in range(ENCODING_COUNT)])
        self.ln = nn.LayerNorm(EMBEDDING_SIZE*2)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(EMBEDDING_SIZE*2, 6)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        x = self.drop(x)
        x = torch.amax(x, dim=1) # Reduce sequence dim (B, C)
        x = self.out(x)
        return x

class Dropout(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=EMBEDDING_SIZE, padding_idx=vocab['<pad>'])
        self.position_embedding = nn.Embedding(
            num_embeddings=MAX_LEN, embedding_dim=EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embed=EMBEDDING_SIZE, n_head=HEADS) for _ in range(ENCODING_COUNT)])
        self.ln = nn.LayerNorm(EMBEDDING_SIZE)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(EMBEDDING_SIZE, 6)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        x = self.drop(x)
        x = torch.amax(x, dim=1) # Reduce sequence dim (B, C)
        x = self.out(x)
        return x
    

class ExtraHeads(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=EMBEDDING_SIZE, padding_idx=vocab['<pad>'])
        self.position_embedding = nn.Embedding(
            num_embeddings=MAX_LEN, embedding_dim=EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embed=EMBEDDING_SIZE, n_head=HEADS*2) for _ in range(ENCODING_COUNT)])
        self.ln = nn.LayerNorm(EMBEDDING_SIZE)
        self.out = nn.Linear(EMBEDDING_SIZE, 6)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        x = torch.amax(x, dim=1) # Reduce sequence dim (B, C)
        x = self.out(x)
        return x
    
class ExtraEncodings(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=EMBEDDING_SIZE, padding_idx=vocab['<pad>'])
        self.position_embedding = nn.Embedding(
            num_embeddings=MAX_LEN, embedding_dim=EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embed=EMBEDDING_SIZE, n_head=HEADS) for _ in range(ENCODING_COUNT*2)])
        self.ln = nn.LayerNorm(EMBEDDING_SIZE)
        self.out = nn.Linear(EMBEDDING_SIZE, 6)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)
        x = torch.amax(x, dim=1) # Reduce sequence dim (B, C)
        x = self.out(x)
        return x