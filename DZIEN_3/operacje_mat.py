import tensorflow as tf
import numpy as np
import random

# Możliwe operacje i słowa
digits = list(map(str, range(10)))
operations = {
    "plus": "+",
    "minus": "-",
    "razy": "*",
    "podzielić przez": "/"
}

def generate_example():
    a, b = random.randint(1, 9), random.randint(1, 9)
    op_word, symbol = random.choice(list(operations.items()))
    sentence = f"ile to {a} {op_word} {b}"
    try:
        result = str(eval(f"{a}{symbol}{b}"))
    except ZeroDivisionError:
        result = "0"
    return sentence, result

# Tokenizacja
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
examples = [generate_example() for _ in range(1000)]
input_texts, target_texts = zip(*examples)

tokenizer.fit_on_texts(input_texts + target_texts)
X = tokenizer.texts_to_sequences(input_texts)
y = tokenizer.texts_to_sequences(target_texts)

X = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post')

X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y)

vocab_size = len(tokenizer.word_index) + 1

# Model: wejście tekst, wyjście liczba (w tokenach)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 32),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=30, verbose=0)

#Predykcja
def predict_equation(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=X.shape[1], padding='post')
    pred = model.predict(padded)
    indices = tf.argmax(pred, axis=-1).numpy()
    chars = [tokenizer.index_word.get(i, '') for i in indices]
    return ''.join(chars)

#Test
for test in ["ile to 7 razy 3", "ile to 9 minus 2", "ile to 5 plus 4", "ile to 8 podzielić przez 2"]:
    print(f"{test} = {predict_equation(test)}")
