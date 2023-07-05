import torch
torch.manual_seed(420)

piece_list = ["K", "Q", "N", "B", "R", ""]
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
squares = [a+str(b) for a in files for b in range(1, 9)]
move_list = sorted([a+b for a in piece_list for b in squares])
#print(move_list, len(move_list))

move_ct = len(move_list)
mtoi = {mv:i for i, mv in enumerate(move_list)}
itom = {i:mv for i, mv in enumerate(move_list)}

encode = lambda mvs: [mtoi[m] for m in mvs]
decode = lambda ten: ' '.join([itom[t] for t in ten])

inp = open('magnusgames.txt', 'r')
for line in inp:
    cur_moves = line.split(' ')
    enc = encode(cur_moves)
    print(enc)
    print(cur_moves)
    break