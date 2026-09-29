from kodane_library.algorithms.rvea import RVEA
from kodane_library.core.problem import Problem


def multi_objective(x, problem):
    f1 = sum(value ** 2 for value in x)

    f2 = sum(
        (value - 2.0) ** 2
        for value in x
    )

    return [f1, f2]


problem = Problem(
    objective=multi_objective,
    dimension=5,
    bounds=[(-5, 5)] * 5
)


optimizer = RVEA(
    problem,
    population_size=30,
    max_iterations=20,
    seed=42
)

pareto_front = optimizer.optimize()


print("========================================")
print("          KODANE - RVEA SMOKE TEST")
print("========================================")

print()
print("Pareto solutions:", len(pareto_front))
print("NFE:", optimizer.nfe)
print("History length:", len(optimizer.history))

print()
print("Sample Pareto solutions:")

for solution in pareto_front[:5]:

    print(
        "Variables:",
        solution.variables,
        "Objectives:",
        solution.objectives
    )

print()
print("========================================")