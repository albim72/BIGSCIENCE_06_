import numpy as np
import matplotlib.pyplot as plt

# Funkcja Rastrigina
def rastrigin(x):
    return 10*len(x) + np.sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# Funkcja mutacji (dodanie szumu Gaussa)
def mutate(x, sigma=0.3):
    return x + np.random.normal(0, sigma, size=x.shape)

# Parametry algorytmu
population_size = 20
dimensions = 2
generations = 200

# Inicjalizacja populacji (losowe punkty w zakresie [-5.12, 5.12])
population = [np.random.uniform(-5.12, 5.12, size=dimensions) for _ in range(population_size)]

# Lista do zapisywania najlepszego osobnika z każdej generacji
best_individuals = []

# Główna pętla optymalizacji
for gen in range(generations):
    fitness = [rastrigin(ind) for ind in population]
    idx = np.argsort(fitness)
    elites = [population[i] for i in idx[:5]]  # najlepszych 5 osobników
    best_individuals.append(population[idx[0]])  # zapisz najlepszego

    # Tworzenie nowej populacji
    new_population = elites.copy()
    while len(new_population) < population_size:
        parent = elites[np.random.randint(0, len(elites))]
        child = mutate(parent)
        new_population.append(child)

    population = new_population

# Zamień listę najlepszych osobników na tablicę NumPy
best_individuals = np.array(best_individuals)

# Tworzenie wykresu trajektorii
plt.figure(figsize=(10, 6))
plt.plot(best_individuals[:, 0], best_individuals[:, 1],
         marker='o', markersize=3, linewidth=1, label="Trajektoria najlepszego osobnika")
plt.scatter([0], [0], color='red', label='Minimum globalne (0,0)')
plt.title("Trajektoria najlepszego osobnika w funkcji Rastrigina (2D)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend()
plt.show()
