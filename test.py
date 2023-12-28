import chess.pgn

def read_games(pgn_file):
    games = []
    with open(pgn_file) as pgn:
        while True:
            try:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                games.append(game)
            except Exception as e:
                break
    return games


# pgn = open("./lichess_db_standard_rated_2014-01.pgn")

# first_game = chess.pgn.read_game(pgn)
# second_game = chess.pgn.read_game(pgn)

# board = first_game.board()

# for move in first_game.mainline_moves():
#     board.move_stack.append(move)
#     print(move)
