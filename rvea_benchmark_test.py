import math

from kodane_library.algorithms.rvea import RVEA
from kodane_library.core.problem import Problem


# ============================================================
# CONFIGURATION
# ============================================================

DIMENSION = 10
POPULATION_SIZE = 30
MAX_ITERATIONS = 100
SEED = 42


# ============================================================
# MULTI-OBJECTIVE BENCHMARKS
# ============================================================

def zdt1(x, problem):
    f1 = x[0]

    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)

    f2 = g * (
        1.0 - math.sqrt(f1 / g)
    )

    return [f1, f2]


def zdt2(x, problem):
    f1 = x[0]

    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)

    f2 = g * (
        1.0 - (f1 / g) ** 2
    )

    return [f1, f2]


def zdt3(x, problem):
    f1 = x[0]

    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)

    f2 = g * (
        1.0
        - math.sqrt(f1 / g)
        - (f1 / g)
        * math.sin(
            10.0 * math.pi * f1
        )
    )

    return [f1, f2]


def dtlz2(x, problem):
    m = 3

    g = sum(
        (value - 0.5) ** 2
        for value in x[m - 1:]
    )

    f1 = (
        1.0 + g
    ) * math.cos(
        x[0] * math.pi / 2.0
    ) * math.cos(
        x[1] * math.pi / 2.0
    )

    f2 = (
        1.0 + g
    ) * math.cos(
        x[0] * math.pi / 2.0
    ) * math.sin(
        x[1] * math.pi / 2.0
    )

    f3 = (
        1.0 + g
    ) * math.sin(
        x[0] * math.pi / 2.0
    )

    return [f1, f2, f3]


# ============================================================
# BENCHMARK DEFINITIONS
# ============================================================

BENCHMARKS = [
    (
        "ZDT1",
        zdt1,
        [(0.0, 1.0)] * DIMENSION
    ),
    (
        "ZDT2",
        zdt2,
        [(0.0, 1.0)] * DIMENSION
    ),
    (
        "ZDT3",
        zdt3,
        [(0.0, 1.0)] * DIMENSION
    ),
    (
        "DTLZ2",
        dtlz2,
        [(0.0, 1.0)] * DIMENSION
    ),
]


# ============================================================
# RUN BENCHMARKS
# ============================================================

print()
print("==========================================================")
print("              KODANE - RVEA BENCHMARK")
print("==========================================================")

print()
print(
    f"Population: {POPULATION_SIZE}"
)

print(
    f"Iterations: {MAX_ITERATIONS}"
)

print(
    f"Dimension:  {DIMENSION}"
)

print(
    f"Seed:       {SEED}"
)

print()
print(
    f"{'Function':<12}"
    f"{'Pareto Size':<15}"
    f"{'NFE':<10}"
)

print("-" * 50)


for name, objective, bounds in BENCHMARKS:

    problem = Problem(
        objective=objective,
        dimension=DIMENSION,
        bounds=bounds
    )

    optimizer = RVEA(
        problem,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        seed=SEED
    )

    pareto_front = optimizer.optimize()

    print(
        f"{name:<12}"
        f"{len(pareto_front):<15}"
        f"{optimizer.nfe:<10}"
    )


print()
print("==========================================================")
print("RVEA benchmark completed.")
print("==========================================================")