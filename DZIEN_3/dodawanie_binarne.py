import tensorflow as tf
import numpy as np
import random

# Zamiana liczby całkowitej na binarną listę bitów
def int2bin(n, bits=8):
    # return [int(b) for b in format(n, f'0{bits}b')]
    return list(map(int, format(n,f'0{bits}b')))

# Generowanie danych
# X, Y = [], []
# for _ in range(1000):

#     a, b = random.randint(0, 127), random.randint(0, 127)
#     x = int2bin(a) + int2bin(b)             # 16-bitowe wejście (2x8 bitów)
#     y = int2bin(a + b, bits=9)              # 9-bitowy wynik (max 254)
#     X.append(x)
#     Y.append(y)

# X = np.array(X, dtype=np.float32)
# Y = np.array(Y, dtype=np.float32)

def generate_data(max_num=128,bits=0):
    X, Y = [], []
    for a in range(max_num):
        for b in range(max_num):
            a_bin = int2bin(a, bits)
            b_bin = int2bin(b, bits)
            x = a_bin + b_bin
            y = int2bin(a + b, bits + 1)
            X.append(x)
            Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

BITS = 8
X, Y = generate_data(max_num=128,bits=BITS)

# Definicja modelu
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(BITS*2,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(BITS+1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# Trening
model.fit(X, Y, epochs=300, batch_size=256, verbose=0)

# Predykcja i interpretacja
def predict_sum(a, b,bits=BITS):
    x_input = np.array([int2bin(a,bits) + int2bin(b,bits)], dtype=np.float32)
    y_pred = model.predict(x_input)[0]
    bin_out = ''.join([str(int(v>0.5)) for v in y_pred])
    return int(bin_out,2)

# Test
print("AI testuje: 25 + 37 =", predict_sum(25, 37))
print("AI testuje: 13 + 120 =", predict_sum(13, 120))
print("AI testuje: 50 + 99 =", predict_sum(50, 99))
print("AI testuje: 127 + 127 =", predict_sum(127, 127))
