import requests
import ndjson
api_token = 'lip_bG5X0P7jbEOAe5nrxNzb'

users = ['DrNykterstein', 'FairChess_on_YouTube', 'A-Liang', 'alireza2003', 'aspiringstar']


headers = {
    "Authorization": 'Bearer ' + api_token,
    "Accept": 'application/x-ndjson'
}

params = {
    'evals': False
}

out = open('fivegmgames.txt', 'w')
original = open('fivegmcheck.txt', 'w')

for i, user in enumerate(users):
    url = 'https://lichess.org/api/games/user/' + user
    
    print("about to response")
    response = requests.get(url, headers=headers, params=params)
    print("GOT IT")

    if response.status_code == 200:
        data = ndjson.loads(response.text)
        for g, d in enumerate(data):
            moves = d['moves']
            original.write(moves)
            original.write('\n')
            moves = moves.replace('x', '')
            moves = moves.replace('+', '')
            moves = moves.replace('#', '')
            move_list = moves.split(' ')
            flag = False
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
                    final[i] = final[1:]
                move_list[i] = final
            if(flag):
                continue
            
            moves = " ".join(move_list)
            out.write(moves)
            out.write("\n")
            if g % 100 == 0:
                print(f"Person{i}, game {g}")
    else:
        print('Request failed with code:', response.status_code)


