import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        #This embedding maps each index to a score distribution of what the next index could be (4 -> 4, 5, 8, ...)
        # this means if the current index is 4, the score for the next token to be 1 is 5. This can be softmaxed to turn into a probability distribution  
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    
    def forward(self, idx, targets=None):
        #gets embedding for every single index given through idx (batch_size * window_size) -> (batch_size * window_size * vocab size) (linear layer is middle man)
        tok_emb = self.token_embedding_table(idx) 
        logits = self.lm_head(tok_emb)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #first generate predictions
            logits, _ = self(idx) #logits are (B, T, C) 
            #look at only the last one (bc bigram)
            logits = logits[:, -1, :] #(B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
