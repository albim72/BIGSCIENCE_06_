import numpy as np

def rastrigin(x):
    return 10*len(x) + np.sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

def mutate(x,sigma=0.3):
    return x + np.random.normal(0,sigma,size=x.shape)

#parametry
population_size = 20
dimensions = 2
generations = 1000

#inicjalizacja populacji
population = [np.random.uniform(-5.12,5.12,size=dimensions) for _ in range(population_size)]

#główna pętla
for gen in range(generations):
    fitness = [rastrigin(ind) for ind in population]
    idx = np.argsort(fitness)
    elites = [population[i] for i in idx[:5]] #wybieramy 5 najlepszych
    new_population = elites.copy()

    while len(new_population) < population_size:
        parent = elites[np.random.randint(0,len(elites))]
        child = mutate(parent)
        new_population.append(child)
    population = new_population

    print(
        f"Generation {gen}: best individual = {rastrigin(population[0]):.4f}"
    )
