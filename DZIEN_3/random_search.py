import tensorflow as tf
import numpy as np
import random

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

# 1. Dane
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Funkcja do budowy modelu
def build_model(num_layers, units, dropout_rate, activation, learning_rate):
    model = Sequential([Flatten(input_shape=(28, 28))])
    for _ in range(num_layers):
        model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Random Search
search_space = {
    'num_layers': [1, 2, 3],
    'units': [64, 128, 256],
    'dropout_rate': [0.0, 0.2, 0.5],
    'activation': ['relu', 'tanh'],
    'learning_rate': [0.001, 0.01, 0.1]
}

best_accuracy = 0
best_params = None
best_model = None

for i in range(10):  # 10 losowych prób
    params = {
        key: random.choice(values)
        for key, values in search_space.items()
    }

    print(f"\nTrial {i+1} with parameters: {params}")
    model = build_model(**params)
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        best_model = model

print("\n✅ Best parameters found:")
print(best_params)
print(f"Test accuracy: {best_accuracy:.4f}")
