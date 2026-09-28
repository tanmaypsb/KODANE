from kodane_library.algorithms.cpo import CPO
from kodane_library.core.problem import Problem


def sphere(x, problem):
    return sum(value ** 2 for value in x)


problem = Problem(
    objective=sphere,
    dimension=5,
    bounds=[(-10, 10)] * 5
)

optimizer = CPO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

best = optimizer.optimize()

print("Best solution:", best.variables)
print("Best fitness:", best.fitness)
print("History length:", len(optimizer.history))
print("NFE:", optimizer.nfe)
