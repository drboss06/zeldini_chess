import chess.pgn
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from io import BytesIO
import chess
import requests
from pprint import pprint
import berserk


API_KEY = "lip_v5SVRcfvxQYPXs3oOtkm"

session = berserk.TokenSession("lip_v5SVRcfvxQYPXs3oOtkm")
client = berserk.Client(session=session)

# Функция для чтения игр из файла PGN
def read_games(pgn_file, num_games=0):
    games = []
    if num_games > 0:
        counter = 0
        with open(pgn_file) as pgn:
            while counter <= num_games:
                try:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    games.append(game)
                    counter += 1
                except Exception as e:
                    break
        pgn.close()
        return games
    else:
        with open(pgn_file) as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    games.append(game)
                except Exception as e:
                    break
        pgn.close()
        return games

# Преобразование состояния доски в числовой тензор
def board_to_tensor(board):
    # Определяем словарь для преобразования фигур в индексы
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    
    # Создаем тензор нулей размером 8x8x14
    tensor = np.zeros((8, 8, 14), dtype=np.float32)
    
    # Проходим по всем клеткам доски
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, i))
            if piece:
                # Устанавливаем 1 на соответствующем месте тензора
                index = piece_to_index.get(piece.symbol())
                tensor[i, j, index] = 1
    
    # Добавляем дополнительные каналы для позиционных признаков
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 12] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 13] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 12] = -1
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 13] = -1

    # Дополнительные признаки, такие как текущий ход, можно добавить здесь

    return tensor

# Создание модели нейронной сети (заглушка)
def create_chess_model():
    # model = tf.keras.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 14)),
    #     #layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     #layers.Dense(64, activation='softmax'),
    #     layers.Dense(1, activation='linear')
    # ])

    # model = tf.keras.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 14), padding='same'),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     #layers.Conv2D(64, (3, 3), activation='relu'),
    #     #layers.BatchNormalization(),
    #     layers.Flatten(),

    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     layers.Dense(64, activation='relu'),
    #     #layers.Dense(64, activation='softmax'),
    #     layers.Dense(1, activation='linear')
    # ])
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 14), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        #layers.Conv2D(64, (3, 3), activation='relu'),
        #layers.BatchNormalization(),
        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        #layers.Dense(64, activation='softmax'),
        layers.Dense(1, activation='linear')
    ])
    return model

# Основная функция
def evaluation_function(board):
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -1  # Черные выиграли
        else:
            return 1  # Белые выиграли
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0  # Ничья
    
    # Материальные значения фигур
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }
    
    # Считаем материальное преимущество
    material = 0
    for piece_type in piece_values:
        material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        material -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    # Нормализуем оценку, чтобы она была в диапазоне от -1 до 1
    max_material = 8 * piece_values[chess.PAWN] + 2 * piece_values[chess.KNIGHT] + 2 * piece_values[chess.BISHOP] + 2 * piece_values[chess.ROOK] + piece_values[chess.QUEEN]
    return material / max_material

def choose_move(model, board):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_move_value = -np.inf
    
    # Итерируем по всем возможным ходам и оцениваем их
    for move in legal_moves:
        board.push(move)
        input_tensor = board_to_tensor(board)  # Преобразуем текущую доску в тензор
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Добавляем размерность для batch
        move_value = model.predict(input_tensor)[0][0]  # Получаем оценку хода от модели
        board.pop()  # Возвращаем доску в исходное состояние
        
        # Сохраняем лучший ход
        if move_value > best_move_value:
            best_move_value = move_value
            best_move = move

    return best_move

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

def main(epochs=10, num_games=100):
    games = read_games('lichess_db_standard_rated_2014-01.pgn', num_games)
    
    X_train = []
    y_train = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            X_train.append(board_to_tensor(board))
            y_train.append(evaluation_function(board))
    
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    
    # Убедитесь, что X_train и y_train содержат одинаковое количество образцов
    assert len(X_train) == len(y_train), "X_train and y_train must contain the same number of samples"
    
    model = create_chess_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    model.save('saved_model.h5')

#main(10, 6000)

model = create_chess_model()
model.load_weights('saved_model.h5')

# Создаем новую игру
board = chess.Board()

gameId = input("Enter id of game: ")

# Пример игрового цикла
# while not board.is_game_over():
#     move = choose_move(model, board)
#     board.push(move)
#     #display_board(board)
#     print(board)
#     # Здесь можно добавить код для хода противника
#     opponent_move = choose_move(model, board)
#     board.push(opponent_move)
#     #display_board(board)
#     print(board)

moves_counter = 0

while not board.is_game_over():
    move = choose_move(model, board)
    board.push(move)
    client.board.make_move(gameId, move)
    moves_counter += 1

    while True:
        i = client.board.stream_game_state(gameId)
        i = next(i)
        b = i["state"]["moves"]
        list_b = ans_to_list(b)
        if moves_counter < len(list_b):
            moves_counter += 1
            #board.push(list_b[len(list_b) - 1])
            board.push(chess.Move.from_uci(list_b[len(list_b) - 1]))
            break

    