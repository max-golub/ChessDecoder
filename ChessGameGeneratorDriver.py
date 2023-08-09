import torch
import configparser
import time
from ChessGen import ChessGen

torch.manual_seed(420)
config = configparser.ConfigParser()
config.read('config.ini')

block_size = int(config["GENERATOR"]["block_size"])
batch_size = int(config["GENERATOR"]["batch_size"])
learning_rate = float(config["GENERATOR"]["learning_rate"])
max_iters = int(config["GENERATOR"]["max_iters"])
eval_iters = int(config["GENERATOR"]["eval_iters"])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

master_data = []

piece_list = ["K", "Q", "N", "B", "R", ""]
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
squares = [a+str(b) for a in files for b in range(1, 9)]
move_list = sorted([a+b for a in piece_list for b in squares])
move_list.append('O-O')
move_list.append('O-O-O')
move_list.append('s') #SOG character
move_list.append('e') #EOG character

move_ct = len(move_list)
mtoi = {mv:i for i, mv in enumerate(move_list)}
itom = {i:mv for i, mv in enumerate(move_list)}

encode = lambda mvs: [mtoi[m] for m in mvs]
decode = lambda ten: ' '.join([itom[t.item()] for t in ten.flatten()])

@torch.no_grad()
def estimate_loss(min_loss):
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
            if split == 'val' and loss < min_loss:
                min_loss = loss
                torch.save(m.state_dict(), "./models/nolambda")
        out[split] = losses.mean()
    m.train()
    return out, min_loss

def process_game(game_string):
    moves = ['s']
    moves.extend(game_string.split(' ')[:block_size-1]) #len: blocksize + 2
    moves.append('e')
    moves_enc = encode(moves)
    if(len(moves_enc) != block_size+1):
        return
    master_data.append(moves_enc)
    
def get_batch(split):
    data = train_data if split == 'train' else val_data
    game_inds = torch.randint(len(data), (batch_size, ))
    x = torch.stack([torch.tensor(data[i][:-1]) for i in game_inds])
    y = torch.stack([torch.tensor(data[i][1:]) for i in game_inds])
    x = x.to(device)
    y = y.to(device)
    return x, y

if __name__ == '__main__': 
    n_moves = len(move_list)

    #print(move_list, len(move_list))

    inp = open('fivegmgames.txt', 'r')
    num_games = 0
    for line in inp:
        line = line.replace('\n', '')
        if not line:
            continue
        process_game(line)
        num_games += 1


    n = int(0.9 * num_games)
    train_data = master_data[:n]
    val_data = master_data[n:]

    xb, yb = get_batch('train')
    print(decode(xb[0]))
    print(decode(yb[0]))

    model = ChessGen()
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), learning_rate)

    epoch_ct = 0
    startTime = time.time()
    min_loss = 10
    for step in range(max_iters):
        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
            
        
        if step % eval_iters == 0:
            losses, new_min = estimate_loss(min_loss)
            min_loss = new_min
            epoch_ct += 1
            print(f"{epoch_ct}: step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    print('training time:', time.time() - startTime)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))