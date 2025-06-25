import tensorflow as tf
import numpy as np
import random

# Dane
digits = list(map(str, range(10)))
operations = {
    "plus": "+",
    "minus": "-",
    "razy": "*",
    "podzielić przez": "//"
}

def generate_example():
    a, b = random.randint(1, 9), random.randint(1, 9)
    op_word, symbol = random.choice(list(operations.items()))
    question = f"ile to {a} {op_word} {b}"
    answer = str(eval(f"{a}{symbol}{b}"))
    return question, answer

# Dane treningowe
samples = [generate_example() for _ in range(2000)]
input_texts, target_texts = zip(*samples)

# Tokenizacja znak po znaku
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters='')
tokenizer.fit_on_texts(input_texts + target_texts)
vocab_size = len(tokenizer.word_index) + 1

max_input_len = max(len(t) for t in input_texts)
max_output_len = max(len(t) for t in target_texts)

X = tokenizer.texts_to_sequences(input_texts)
Y = tokenizer.texts_to_sequences(target_texts)

X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_input_len, padding='post')
Y = tf.keras.preprocessing.sequence.pad_sequences(Y, maxlen=max_output_len, padding='post')

X = np.array(X)
Y = tf.keras.utils.to_categorical(Y, num_classes=vocab_size)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_input_len),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=25, batch_size=64, verbose=1)

# Funkcja predykcji
def predict(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_input_len, padding='post')
    pred = model.predict(padded)[0]
    indices = np.argmax(pred, axis=-1)
    chars = [tokenizer.index_word.get(i, '') for i in indices]
    return ''.join(chars).strip()

# 🔍 Test
examples = ["ile to 3 plus 4", "ile to 8 minus 5", "ile to 6 razy 7", "ile to 9 podzielić przez 3"]
for ex in examples:
    print(f"{ex} → {predict(ex)}")
