import torch
import time
from AttentionModel import BigramLanguageModel

#hyperparams
batch_size = 4      #64 #how many independent sequences will we process in parallel
block_size = 8       #256 #what is the maximum context length for predictions
max_iters = 5000
learning_rate = 1e-3      #3e-4
eval_interval = 500
eval_iters = 200
n_embed = 32       #384
num_heads = 4 #6
n_layer = 3 #how many decoder blocks in the model 
dropout = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        print("length of dataset (chars)", len(text))
        print(type(text))
        all_chars = sorted(list(set(text)))

    ctoi = {ch:i for i, ch in enumerate(all_chars)}
    itoc = {i:ch for i, ch in enumerate(all_chars)}

    vocab_size = len(all_chars)
    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda i: ''.join([itoc[c] for c in i])
    
    print(encode("I be busy in the street I be riding on his meat"))
    print(decode(encode("I be busy in the street I be riding on his meat")))
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    torch.manual_seed(1337)

    print(device)

    xb, yb = get_batch('train')

    model = BigramLanguageModel(vocab_size, n_embed, block_size, num_heads, n_layer, dropout)
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    startTime = time.time()
    for step in range(max_iters):
        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    print('training time:', time.time() - startTime)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))