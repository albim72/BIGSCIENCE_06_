import tensorflow as tf
import numpy as np
import random

# Zamiana liczby całkowitej na binarną listę bitów
def int2bin(n, bits=8):
    return [int(b) for b in format(n, f'0{bits}b')]

# Generowanie danych
X, Y = [], []
for _ in range(1000):
    a, b = random.randint(0, 127), random.randint(0, 127)
    x = int2bin(a) + int2bin(b)             # 16-bitowe wejście (2x8 bitów)
    y = int2bin(a + b, bits=9)              # 9-bitowy wynik (max 254)
    X.append(x)
    Y.append(y)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# Definicja modelu
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# Trening
model.fit(X, Y, epochs=50, batch_size=32, verbose=0)

# Predykcja i interpretacja
def predict_sum(a, b):
    x = np.array([int2bin(a) + int2bin(b)], dtype=np.float32)
    y_pred = model.predict(x)[0]
    bits = [str(int(round(v))) for v in y_pred]
    return int("".join(bits), 2)

# Test
print("AI testuje: 25 + 37 =", predict_sum(25, 37))
print("AI testuje: 13 + 120 =", predict_sum(13, 120))
print("AI testuje: 50 + 99 =", predict_sum(50, 99))
