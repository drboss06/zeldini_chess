import tensorflow as tf
from keras import layers
import numpy as np
from pgn_parser import parser, pgn



model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse')

# Функция для создания датасета
def generate_data(num_samples):
    X = np.random.rand(num_samples, 2)
    y = np.sum(X, axis=1, keepdims=True)
    return X, y

# Создание дата сета
X_train, y_train = generate_data(10000)
X_test, y_test = generate_data(1000)

# Тренировка
model.fit(X_train, y_train, epochs=10, batch_size=32)

# оценка
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Тестирование
X_new = np.array([[0.1, 0.2], [0.2, 0.3]])
y_pred = model.predict(X_new)
print(f"Predicted sums: {y_pred}")