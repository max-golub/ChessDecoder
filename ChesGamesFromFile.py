inp = open('fivegmcheck.txt')

out = open('fivegmgames.txt', 'w')

def convert_alg_to_red(move):
    move = move.replace('x', '')
    move = move.replace('+', '')
    move = move.replace('#', '')
    final = move
    if '=' in move:
        equal_ind = move.index('=')
        final = move[:equal_ind]
    if len(final) == 4:
        final = final[0] + final[2:]
    if len(final) == 3 and final[0] in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}:
        final = final[1:]
    return final

if __name__ == '__main__': 
    for g, moves in enumerate(inp):
        moves = moves.replace('x', '')
        moves = moves.replace('+', '')
        moves = moves.replace('#', '')
        moves = moves.replace('\n', '')
        move_list = moves.split(' ')
        flag = False
        if not moves:
            continue
        for i, move in enumerate(move_list):
            if move[0] == 'P':
                flag = True
                break
            final = move
            if '=' in move:
                equal_ind = move.index('=')
                final = move[:equal_ind]
            if len(final) == 4:
                final = final[0] + final[2:]
            if len(final) == 3 and final[0] in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}:
                final = final[1:]
            move_list[i] = final
        if(flag):
            continue
        
        moves = " ".join(move_list)
        out.write(moves)
        out.write("\n")
        if g % 100 == 0:
            print(f"game {g}")
    


