import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, time_step, embed_dim, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(time_step, time_step)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, time_step, embed_dim, n_head, head_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(time_step, embed_dim, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, time_step, embed_dim, n_head):
        # embed_dim: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embed_dim // n_head
        self.sa = MultiHeadAttention(time_step, embed_dim, n_head, head_size)
        self.ffwd = FeedFoward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, gptconf):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.gptconf = gptconf
        self.token_embedding_table = nn.Embedding(gptconf.vocab_size, gptconf.embed_dim)
        self.position_embedding_table = nn.Embedding(gptconf.time_step, gptconf.embed_dim)
        self.blocks = nn.Sequential(*[Block(gptconf.time_step, gptconf.embed_dim, gptconf.n_head) for _ in range(gptconf.n_layer)])
        self.ln_f = nn.LayerNorm(gptconf.embed_dim) # final layer norm
        self.lm_head = nn.Linear(gptconf.embed_dim, gptconf.vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape # shape: [64, 256]

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C), shape: [64, 256, 384]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C), shape: [256, 384]
        x = tok_emb + pos_emb # (B,T,C), shape: [64, 256, 384]
        x = self.blocks(x) # (B,T,C), shape: [64, 256, 384]
        x = self.ln_f(x) # (B,T,C), shape: [64, 256, 384]
        logits = self.lm_head(x) # (B,T,vocab_size), shape: [64, 256, 65]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # shape: [64, 256, 65]
            logits = logits.view(B*T, C) # shape: [64*256, 65]
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last time_step tokens
            idx_cond = idx[:, -self.gptconf.time_step:]
            # get the predictions
            logits, loss = self(idx_cond) # self.forward(idx_cond), logits: [B, T, C]
            # focus only on the last time step, get the prediction of T+1 token
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__ == "__main__":
    # hyper parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_args = dict(n_layer=6, n_head=6, embed_dim=384, time_step=256,
                  bias=True, vocab_size=65)
    
    gptconf = GPTConfig(**model_args)
    B, T, C = gptconf.batch_size, gptconf.time_step, gptconf.embed_dim

    # input
    input = torch.randint(low=0, high=gptconf.vocab_size, size=(B, T), device=device)

    # model
    model = GPT(gptconf)
    model.to(device)
    print(model)

    # forward
    output, loss = model(input)
    print(output.shape)

    # generate
    context = torch.zeros((1, 2), dtype=torch.long, device=device) # (B, T) 
    print(model.generate(context, max_new_tokens=10)[0].tolist()) # (B, T + max_new_tokens)
