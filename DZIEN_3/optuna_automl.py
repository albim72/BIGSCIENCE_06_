import optuna
import optuna.visualization
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Załaduj dane
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Funkcja oceny
def objective(trial):
    num_layers = trial.suggest_int("num_layers", 1, 3)
    units = trial.suggest_categorical("units", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    for _ in range(num_layers):
        model.add(Dense(units, activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=64,
        verbose=0,
        callbacks=[optuna.integration.TFKerasPruningCallback(trial, "val_accuracy")]
    )

    return max(history.history["val_accuracy"])

# Uruchom optymalizację
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Najlepszy model – trening + test
best = study.best_params
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
for _ in range(best["num_layers"]):
    model.add(Dense(best["units"], activation=best["activation"]))
    if best["dropout"] > 0:
        model.add(Dropout(best["dropout"]))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best["lr"]),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test accuracy:", test_accuracy)
print("Best hyperparameters:", best)

# Wykres
optuna.visualization.plot_optimization_history(study).show()
