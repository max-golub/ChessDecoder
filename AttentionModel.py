import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        #linear layers represent the key, query, value weight matrices
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores (affinities), or how much every key matches with every query. The query is searching for certain attributes, 
        # and the key tells something about each token's attributes. When dot producted (through matrix multiplication), the affinity for each pair of points is calculated
        #Essentially, this means how much should the current token "pay attention" to each token before (or after) it. 
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #aggregate probability distribution against values (the information about each token that is given out)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    #Multiple unrelated heads of self-attention in parallel
    def __init__(self, n_embed, head_size, block_size, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Concatanates all attention heads together, in theory each head should learn something different about the relation in tokens 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, n_embed, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(n_embed, head_size, block_size, num_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, num_heads, n_layers, dropout):
        super().__init__()
        #This embedding maps each index to a score distribution of what the next index could be (4 -> 4, 5, 8, ...)
        # this means if the current index is 4, the score for the next token to be 1 is 5. This can be softmaxed to turn into a probability distribution  
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #4 headed self attention
        self.blocks = nn.Sequential(*[AttentionBlock(n_embed, num_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        #gets embedding for every single index given through idx (batch_size * window_size) -> (batch_size * window_size * vocab size) (linear layer is middle man)
        tok_emb = self.token_embedding_table(idx) #B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T)) #T, C
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            z= logits.shape
            w=targets.shape
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        B, T = idx.shape
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -T:]
            #first generate predictions
            logits, _ = self(idx_cond) #logits are (B, T, C) 
            #look at only the last one (bc bigram)
            logits = logits[:, -1, :] #(B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
