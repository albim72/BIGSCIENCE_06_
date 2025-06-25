import numpy as np
import tensorflow as tf

# Dane: ciągi rosnące (arytmetyczne, geometryczne)
def generate_sequence():
    if np.random.rand() < 0.5:
        start = np.random.randint(0, 10)
        step = np.random.randint(1, 5)
        seq = [start + i * step for i in range(4)]  # arytmetyczny
    else:
        base = np.random.randint(1, 5)
        factor = np.random.randint(2, 4)
        seq = [base * (factor**i) for i in range(4)]  # geometryczny
    return seq[:-1], seq[-1]  # 3 wejściowe + 1 etykieta

# Generujemy dane
X, y = [], []
for _ in range(2000):
    seq, target = generate_sequence()
    X.append(seq)
    y.append(target)

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Normalizacja
X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)
y = (y - X_min) / (X_max - X_min)

# Model LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3, 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X[..., np.newaxis], y, epochs=50, verbose=0)

# Predykcja
def predict_next(seq):
    seq_norm = (np.array(seq) - X_min) / (X_max - X_min)
    input_tensor = seq_norm.reshape(1, 3, 1)
    pred_norm = model.predict(input_tensor)[0][0]
    pred = pred_norm * (X_max - X_min) + X_min
    return round(pred)

# Test
print("Przewidywania:")
for test_seq in [[2, 4, 6], [3, 6, 9], [1, 2, 4], [5, 10, 20], [7, 14, 21]]:
    print(f"{test_seq} ➜ {predict_next(test_seq)}")
