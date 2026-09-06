from kodane_library.core.optimizer import Optimizer


optimizer = Optimizer(
    problem=None,
    population_size=10,
    max_iterations=100
)


print("Population size:", optimizer.population_size)
print("Max iterations:", optimizer.max_iterations)
print("Best solution:", optimizer.best_solution)
print("Best fitness:", optimizer.best_fitness)