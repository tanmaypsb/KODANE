from kodane_library.algorithms.cpo import CPO
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


DIMENSION = 10
POPULATION_SIZE = 30
MAX_ITERATIONS = 100
SEED = 42


BENCHMARKS = [
    ("Sphere", sphere, (-5.12, 5.12)),
    ("Ellipsoid", ellipsoid, (-5.12, 5.12)),
    ("Bent Cigar", bent_cigar, (-100, 100)),
    ("Discus", discus, (-100, 100)),
    ("Rosenbrock", rosenbrock, (-30, 30)),
    ("Rastrigin", rastrigin, (-5.12, 5.12)),
    ("Ackley", ackley, (-32.768, 32.768)),
    ("Griewank", griewank, (-600, 600)),
    ("Schwefel", schwefel, (-500, 500)),
    ("Levy", levy, (-10, 10)),
    ("Zakharov", zakharov, (-5, 10)),
    ("Step", step, (-100, 100)),
]


def make_bounds(lower, upper):
    return [
        (lower, upper)
        for _ in range(DIMENSION)
    ]


print(
    f"{'Function':<18}"
    f"{'Initial Fitness':>22}"
    f"{'Final Fitness':>22}"
    f"{'NFE':>10}"
)

print("-" * 74)


for name, objective, bounds in BENCHMARKS:

    problem = Problem(
        objective=objective,
        dimension=DIMENSION,
        bounds=make_bounds(
            bounds[0],
            bounds[1]
        )
    )

    optimizer = CPO(
        problem,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        seed=SEED
    )

    result = optimizer.optimize()

    initial_fitness = optimizer.history[0]
    final_fitness = result.fitness

    print(
        f"{name:<18}"
        f"{initial_fitness:>22.6e}"
        f"{final_fitness:>22.6e}"
        f"{optimizer.nfe:>10}"
    )
