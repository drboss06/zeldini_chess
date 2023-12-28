import requests
from pprint import pprint
import lichess
import berserk
import json

API_KEY = "lip_v5SVRcfvxQYPXs3oOtkm"

session = berserk.TokenSession("lip_v5SVRcfvxQYPXs3oOtkm")
client = berserk.Client(session=session)

def ans_to_list(ans):
    move = ""
    ans_ = []
    for i in ans:
        if i == " ":
            ans_.append(move)
            move = ""
        else:
            move += i
    ans_.append(move)
    return ans_

gameId = input()
#move = input()
# url = f"https://lichess.org/api/bot/game/{gameId}/move/{move}"

#client.board.make_move(gameId, "e2e4")
a = client.board.stream_game_state(gameId)


#print(i["state"]["moves"])

i = next(a)
b = i["state"]["moves"]
print(ans_to_list(b))
