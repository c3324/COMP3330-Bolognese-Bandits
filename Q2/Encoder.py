import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(
            embed_dim=n_embed, num_heads=n_head)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 2*n_embed),
            nn.ReLU(),
            nn.Linear(2*n_embed, n_embed),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)        
    def forward(self, x):
        x = x + self.sa(x, x, x)[0]
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x