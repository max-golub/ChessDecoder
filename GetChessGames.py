import requests
import ndjson
api_token = 'lip_bG5X0P7jbEOAe5nrxNzb'

user = 'DrNykterstein'
url = 'https://lichess.org/api/games/user/' + user

headers = {
    "Authorization": 'Bearer ' + api_token,
    "Accept": 'application/x-ndjson'
}

params = {
    'evals': False
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = ndjson.loads(response.text)
    out = open('magnusgames.txt', 'w')
    for d in data:
        moves = d['moves']
        moves = moves.replace('x', '')
        moves = moves.replace('+', '')
        moves = moves.replace('#', '')
        move_list = moves.split(' ')
        for i, move in enumerate(move_list):
            if '=' in move:
                move_list[i] = move[:2]
            elif len(move) == 4:
                move_list[i] = move[0] + move[2:]
            elif len(move) == 3 and move[0] in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}:
                move_list[i] = move[1:]
        moves = " ".join(move_list)
        out.write(moves)
        out.write("\n")
else:
    print('Request failed with code:', response.status_code)


