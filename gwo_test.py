from kodane_library.algorithms.gwo import GWO
from kodane_library.core.problem import Problem
from kodane_library.benchmarks.sphere import sphere


def create_problem():
    dimension = 10
    bounds = [(-5.12, 5.12) for _ in range(dimension)]

    return Problem(
        objective=sphere,
        dimension=dimension,
        bounds=bounds
    )


# --------------------------------------------------
# Test 1: Basic optimization
# --------------------------------------------------

problem = create_problem()

optimizer = GWO(
    problem=problem,
    population_size=30,
    max_iterations=100,
    seed=42
)

solution = optimizer.optimize()

assert solution is not None
assert solution.fitness is not None

print("PASS: Basic optimization")


# --------------------------------------------------
# Test 2: Sphere convergence
# --------------------------------------------------

assert solution.fitness < 1e-6

print("PASS: Sphere convergence")


# --------------------------------------------------
# Test 3: Solution dimension
# --------------------------------------------------

assert len(solution.variables) == 10

print("PASS: Solution dimension")


# --------------------------------------------------
# Test 4: Bounds
# --------------------------------------------------

for x in solution.variables:
    assert -5.12 <= x <= 5.12

print("PASS: Solution within bounds")


# --------------------------------------------------
# Test 5: Reproducibility
# --------------------------------------------------

problem1 = create_problem()
problem2 = create_problem()

optimizer1 = GWO(
    problem=problem1,
    population_size=30,
    max_iterations=100,
    seed=42
)

optimizer2 = GWO(
    problem=problem2,
    population_size=30,
    max_iterations=100,
    seed=42
)

solution1 = optimizer1.optimize()
solution2 = optimizer2.optimize()

assert solution1.variables == solution2.variables
assert solution1.fitness == solution2.fitness

print("PASS: Reproducibility")


# --------------------------------------------------
# Test 6: Different seeds
# --------------------------------------------------

problem3 = create_problem()

optimizer3 = GWO(
    problem=problem3,
    population_size=30,
    max_iterations=100,
    seed=123
)

solution3 = optimizer3.optimize()

assert solution3.variables != solution1.variables

print("PASS: Different seeds produce different results")


# --------------------------------------------------
# Test 7: History length
# --------------------------------------------------

assert len(optimizer1.history) == 101

print("PASS: History length")


# --------------------------------------------------
# Test 8: Number of Function Evaluations
# --------------------------------------------------

expected_nfe = 30 * (100 + 1)

assert optimizer1.nfe == expected_nfe

print("PASS: NFE count")


# --------------------------------------------------
# Test 9: Invalid population size
# --------------------------------------------------

try:
    GWO(
        problem=create_problem(),
        population_size=2,
        max_iterations=100
    )
    assert False
except ValueError:
    pass

print("PASS: Invalid population size rejected")


# --------------------------------------------------
# Test 10: Invalid iterations
# --------------------------------------------------

try:
    GWO(
        problem=create_problem(),
        population_size=30,
        max_iterations=-1
    )
    assert False
except ValueError:
    pass

print("PASS: Invalid iterations rejected")


# --------------------------------------------------
# Test 11: Zero iterations
# --------------------------------------------------

problem4 = create_problem()

optimizer4 = GWO(
    problem=problem4,
    population_size=30,
    max_iterations=0,
    seed=42
)

solution4 = optimizer4.optimize()

assert solution4 is not None
assert len(optimizer4.history) == 1
assert optimizer4.nfe == 30

print("PASS: Zero-iteration case")


print()
print("All GWO tests passed.")