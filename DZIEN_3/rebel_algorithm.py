import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Dane: binary classification (breast cancer)
data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model-building function
def build_model(params, input_dim):
    model = Sequential()
    for units in params['layers']:
        model.add(Dense(units, activation=params['activation'], input_dim=input_dim))
        input_dim = None
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Evaluate individual (return accuracy)
def evaluate(individual):
    model = build_model(individual, input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    y_pred = (y_pred > 0.5).astype(int)
    return accuracy_score(y_test, y_pred)

# 4. Mutate individual
def mutate(ind):
    new_ind = ind.copy()
    if random.random() < 0.5:
        new_ind['layers'] = random.choices([16, 32, 64, 128], k=random.randint(1, 3))
    if random.random() < 0.3:
        new_ind['activation'] = random.choice(['relu', 'tanh'])
    if random.random() < 0.3:
        new_ind['lr'] = random.choice([1e-4, 1e-3, 1e-2])
    return new_ind

# 5. Init population
def init_population(size):
    population = []
    for _ in range(size):
        layers = random.choices([16, 32, 64, 128], k=random.randint(1, 3))
        activation = random.choice(['relu', 'tanh'])
        lr = random.choice([1e-4, 1e-3, 1e-2])
        population.append({'layers': layers, 'activation': activation, 'lr': lr})
    return population

# 6. Main REBEL-like evolution
population_size = 10
generations = 5
population = init_population(population_size)

best_model = None
best_acc = 0

for gen in range(generations):
    print(f"\nGENERACJA {gen + 1}")
    scored = [(ind, evaluate(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_current = scored[0]

    print(f"Najlepszy osobnik: {best_current[0]} → ACC = {best_current[1]:.4f}")

    if best_current[1] > best_acc:
        best_acc = best_current[1]
        best_model = best_current[0]

    # Eksploatuj top 3, eksploruj dzieci
    parents = [x[0] for x in scored[:3]]
    children = [mutate(random.choice(parents)) for _ in range(population_size - 3)]
    population = parents + children

# 7. Predykcja i ocena końcowa
print("\nNAJLEPSZY ZNALEZIONY MODEL:")
print(best_model)
print(f"Dokładność na teście: {best_acc:.4f}")

# Trenowanie najlepszego modelu i predykcja
final_model = build_model(best_model, input_dim=X_train.shape[1])
final_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
y_pred = final_model.predict(X_test).flatten()
y_pred_bin = (y_pred > 0.5).astype(int)

# Ocena jakości predykcji
print("\nRAPORT KLASYFIKACJI:")
print(classification_report(y_test, y_pred_bin, digits=4))

print("\nMACIERZ POMYŁEK:")
print(confusion_matrix(y_test, y_pred_bin))


#_________________________________________________________________________________________

def init_population(size):
    population = []
    for _ in range(size):
        layers = random.choices([16, 32, 64, 128], k=random.randint(1, 3))
        activation = random.choice(['relu', 'tanh'])
        lr = random.choice([1e-4, 1e-3, 1e-2])
        epochs = random.randint(5, 20)  # NOWE
        population.append({'layers': layers, 'activation': activation, 'lr': lr, 'epochs': epochs})
    return population


def mutate(ind):
    new_ind = ind.copy()
    if random.random() < 0.5:
        new_ind['layers'] = random.choices([16, 32, 64, 128], k=random.randint(1, 3))
    if random.random() < 0.3:
        new_ind['activation'] = random.choice(['relu', 'tanh'])
    if random.random() < 0.3:
        new_ind['lr'] = random.choice([1e-4, 1e-3, 1e-2])
    if random.random() < 0.4:  # NOWE
        new_ind['epochs'] = random.randint(5, 20)
    return new_ind


def evaluate(individual):
    model = build_model(individual, input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=individual['epochs'], batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    y_pred = (y_pred > 0.5).astype(int)
    return accuracy_score(y_test, y_pred)



final_model.fit(X_train, y_train, epochs=best_model['epochs'], batch_size=32, verbose=0)

