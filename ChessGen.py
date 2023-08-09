import configparser
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
torch.manual_seed(1337)
config = configparser.ConfigParser()
config.read('config.ini')

block_size = int(config['GENERATOR']['block_size'])
n_embed = int(config['GENERATOR']['n_embed'])
head_size = int(config['GENERATOR']['head_size'])
n_moves = int(config["GENERATOR"]['n_moves'])
n_layers = int(config['GENERATOR']['n_layers'])
#dropout = float(config['GENERATOR']['dropout'])
assert(n_embed % head_size == 0)
n_heads = n_embed // head_size

def get_angles(pos, i):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float16(n_embed))
    return pos * angle_rates

def positional_encoding():
    angle_rads = get_angles(np.arange(n_moves)[:, np.newaxis],
                          np.arange(n_embed)[np.newaxis, :])
  
  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    ret = torch.from_numpy(pos_encoding) 
    ret = ret.to(torch.float32)
    return ret


class Head(nn.Module):
    def __init__(self):  
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        #nn.init.xavier_uniform_(self.query.weight)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        #nn.init.xavier_uniform_(self.key.weight)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        #nn.init.xavier_uniform_(self.value.weight)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape #batch, time, context (dimension of t is the same as blocksize, representing the timeline of what we will be attending)
                            #context is the vector representation of each moment in time
        k = self.key(x) * (n_layers / 3 ** -0.25) #(B, T, hs)
        q = self.query(x) * (n_layers / 3  ** -0.25)#(B, T, hs)
        v = self.value(x) * (n_layers / 3  ** -0.25) #(B, T, hs)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # calculate affinities between all time indices using keys and values 
                                                    #B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        #wei = self.dropout(wei)
        out = wei @ v #(B, T, hs)
        return out 
        
class MultiHeadedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(Head() for _ in range(n_heads))
        self.proj = nn.Linear(n_embed, n_embed)
        #nn.init.xavier_uniform_(self.proj.weight)
        #self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(x)
        #out = self.dropout(self.proj(x))
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(n_embed, 4*n_embed)
        #nn.init.xavier_uniform_(self.out.weight)
        self.inp = nn.Linear(4*n_embed, n_embed)
        #nn.init.xavier_uniform_(self.inp.weight)
        self.forward = nn.Sequential(
            self.out, #*4
            nn.ReLU(),
            self.inp,
            #nn.Dropout(dropout)
        )
    def forward(self, x):
        out = self.forward(x)
        return out * (n_layers / 3 ** -0.25)

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadedAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.sa(self.ln2(x))
        return x
    
class ChessGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(n_moves, n_embed)
        self.token_embedding_table.weight.data.normal_(0, n_embed**-0.5)
        self.position_embedding_table = torch.squeeze(positional_encoding(), 0) #nn.Embedding(n_moves, n_embed) #change this to sine position embedding like in GPT 
        #self.position_embedding_table = nn.Embedding(n_moves, n_embed)
        self.blocks = nn.Sequential(*[AttentionBlock() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, n_moves)
        
    def forward(self, x, labels=None):
        tok_emb = self.token_embedding_table(x) #B, T, C
        pos_emb = self.position_embedding_table[x] #(x)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if labels is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            labels = labels.view(B*T)
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        B, T = x.shape
        for _ in range(max_new_tokens):
            idx_cond = x[:, -T:]
            #first generate predictions
            logits, _ = self(idx_cond) #logits are (B, T, C) 
            #look at only the last one (bc bigram)
            logits = logits[:, -1, :] #(B, C)
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1) #(B, 1)
        return x_next, probs
        