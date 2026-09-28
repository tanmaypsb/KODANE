from kodane_library.algorithms.gwo import GWO
from kodane_library.core.problem import Problem

from kodane_library.benchmarks.sphere import sphere
from kodane_library.benchmarks.ellipsoid import ellipsoid
from kodane_library.benchmarks.bent_cigar import bent_cigar
from kodane_library.benchmarks.discus import discus
from kodane_library.benchmarks.rosenbrock import rosenbrock
from kodane_library.benchmarks.rastrigin import rastrigin
from kodane_library.benchmarks.ackley import ackley
from kodane_library.benchmarks.griewank import griewank
from kodane_library.benchmarks.schwefel import schwefel
from kodane_library.benchmarks.levy import levy
from kodane_library.benchmarks.zakharov import zakharov
from kodane_library.benchmarks.step import step


POPULATION_SIZE = 30
MAX_ITERATIONS = 100
DIMENSION = 10
SEED = 42

def make_bounds(lower, upper):
    return [(lower, upper) for _ in range(DIMENSION)]



benchmarks = [
    ("Sphere", sphere, make_bounds(-5.12, 5.12)),
    ("Ellipsoid", ellipsoid, make_bounds(-5.12, 5.12)),
    ("Bent Cigar", bent_cigar, make_bounds(-5.12, 5.12)),
    ("Discus", discus, make_bounds(-5.12, 5.12)),
    ("Rosenbrock", rosenbrock, make_bounds(-5.0, 10.0)),
    ("Rastrigin", rastrigin, make_bounds(-5.12, 5.12)),
    ("Ackley", ackley, make_bounds(-32.768, 32.768)),
    ("Griewank", griewank, make_bounds(-600.0, 600.0)),
    ("Schwefel", schwefel, make_bounds(-500.0, 500.0)),
    ("Levy", levy, make_bounds(-10.0, 10.0)),
    ("Zakharov", zakharov, make_bounds(-5.0, 10.0)),
    ("Step", step, make_bounds(-5.12, 5.12)),
]


print()
print("KODANE - GWO Benchmark Suite")
print("=" * 75)
print(f"Population Size : {POPULATION_SIZE}")
print(f"Iterations      : {MAX_ITERATIONS}")
print(f"Dimension       : {DIMENSION}")
print(f"Seed            : {SEED}")
print()

print(
    f"{'Function':<15}"
    f"{'Initial Fitness':>20}"
    f"{'Final Fitness':>20}"
    f"{'NFE':>10}"
)

print("-" * 75)


expected_nfe = POPULATION_SIZE * (MAX_ITERATIONS + 1)
expected_history = MAX_ITERATIONS + 1


for name, objective, bounds in benchmarks:

    problem = Problem(
        objective=objective,
        dimension=DIMENSION,
        bounds=bounds
    )

    optimizer = GWO(
        problem=problem,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        seed=SEED
    )

    solution = optimizer.optimize()

    initial_fitness = optimizer.history[0]
    final_fitness = solution.fitness

    print(
        f"{name:<15}"
        f"{initial_fitness:>20.6e}"
        f"{final_fitness:>20.6e}"
        f"{optimizer.nfe:>10}"
    )


print()
print(f"Expected NFE per benchmark     : {expected_nfe}")
print(f"Expected history length        : {expected_history}")
print()