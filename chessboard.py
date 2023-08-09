import chess
import torch
from ChessGen import ChessGen
from ChessGameGeneratorDriver import encode, decode
from enum import Enum
from ChesGamesFromFile import convert_alg_to_red
import torch.nn.functional as F


class Type(Enum):
    PAWN = 1
    ROOK = 2
    KNIGHT = 3 


def pick_move(probs, legal_moves, board):
    legal_moves = list(legal_moves)
    red_list = encode([convert_alg_to_red(board.san(move)) for move in legal_moves])
    probs = F.softmax(probs, dim=-1)
    legal_inds = torch.ones(388, dtype=torch.bool)
    legal_inds[red_list] = 0
    probs = torch.masked_fill(probs, legal_inds, -1)
    pick = torch.argmax(probs)
    return convert_to_normal(legal_moves, decode(pick), board)
    

def convert_to_normal(move_list, unparsed, board):
    type = Type.PAWN
    cand = []
    uci_cand = []
    for m in move_list:
        alg = board.san(m)
        uci = board.uci(m)
        if len(uci) == 5:
            uci = uci[:-1]
        if (alg == 'O-O' or alg == 'O-O-O') and (alg == unparsed):
            return alg
        if uci[-2:] == unparsed[-2:]:
            if len(unparsed) == 2 and alg[0] in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}:
                cand.append(alg)
                uci_cand.append(uci)
                continue
            if unparsed[0] == alg[0]:
                type = Type.ROOK if unparsed[0] == 'R' else Type.KNIGHT
                cand.append(alg) 
                uci_cand.append(uci)
    if len(cand) == 1:
        return cand[0]
    if len(cand) != 2:
        print("You messed up in this function foo")
        exit(1)
        
    if type == Type.PAWN:
        return cand[0] if cand[0][0] in {'a', 'b', 'c', 'f', 'g', 'h'} else cand[1]
    elif type == Type.ROOK:
        return cand[0] if chess.square_distance(chess.parse_square(uci_cand[0][:2]), chess.parse_square(uci_cand[0][2:])) \
            < chess.square_distance(chess.parse_square(uci_cand[1][:2]), chess.parse_square(uci_cand[1][2:])) else cand[1]
    else:
        start1 = uci_cand[0][:2]
        start2 = uci_cand[1][:2]
        centers = ['e4', 'e5', 'd4', 'd5']
        r_avg0 = 0.0
        r_avg1 = 0.0
        for c in centers:
            r_avg0 += chess.square_distance(chess.parse_square(c), chess.parse_square(start1))
            r_avg1 += chess.square_distance(chess.parse_square(c), chess.parse_square(start2))
        return cand[0] if r_avg0 > r_avg1 else r_avg1


def check_move(valid_moves, move, board):
    move = move.replace('+', '')
    move = move.replace('#', '')
    valid_moves = list(valid_moves)
    valid_mv = False
    for m in valid_moves:
        alg = board.san(m)
        alg = alg.replace('+', '')
        alg = alg.replace('#', '')
        if alg == move:
            valid_mv = True
            break
    return valid_mv
        

def begingame(is_white):
    board = chess.Board()
    m = ChessGen()
    m.load_state_dict(torch.load("./models/nolambda"))
    context = torch.zeros((1,1), dtype=torch.long)
    context[0] = 386
    #code for s is 386
    
    if is_white:
        first_mv = input("Please make a move (algebraic notation)")
        while not check_move(board.legal_moves, first_mv, board):
            first_mv = input("Valid move, please!")
        board.push_san(first_mv)
        enc = torch.tensor([encode([first_mv])])
        context = torch.cat((context, enc), dim=1)
    
    while(True):
        print('Board state:')
        print(board.unicode())
        if board.outcome():
            print("the game is over, somehow")
            return
        
        cand, probs = m.generate(context, 1)
        move_list = decode(cand).split(' ')
        unparsed_next = move_list[-1]
        if unparsed_next == 'e' or len(context) >= 16:
            print("Opening is now over. Good luck!")
            return
        book_move = pick_move(probs, board.legal_moves, board) 
        board.push_san(book_move)
        enc = torch.tensor([encode([convert_alg_to_red(book_move)])])
        context = torch.cat((context, enc), dim=1)
        print(f"The oracle says... play {book_move}!")
        print("The position is now")
        print(board.unicode())
        if board.outcome():
            print("the game is over, somehow")
            return
        
        next_mv = input("Please make a move (algebraic notation)")
        while not check_move(board.legal_moves, next_mv, board):
            next_mv = input("Valid move, please!")
        board.push_san(next_mv)
        enc = torch.tensor([encode([convert_alg_to_red(next_mv)])])
        context = torch.cat((context, enc), dim=1)
        

if __name__ == '__main__':
    color = input("Do you want to play white or black? (W/B)")
    if color.lower() == 'w':
        begingame(True)
    elif color.lower() == 'b':
        begingame(False)
    else:
        print("You need to input a correct color")


