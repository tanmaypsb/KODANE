import itertools

from kodane_library.core.problem import Problem
from kodane_library.core.robot import Robot
from kodane_library.core.task import Task
from kodane_library.core.solution import Solution
from kodane_library.algorithms.discrete_gwo import DiscreteGWO

from kodane_library.objectives.distance import (
    total_distance,
    battery_constraint,
    total_battery_constraint
)


# ============================================================
# REALISTIC KODANE ASSIGNMENT PROBLEM
# ============================================================

robots = [
    Robot(1, (0, 0), 100, 10),
    Robot(2, (10, 0), 100, 10),
    Robot(3, (0, 10), 100, 10),
    Robot(4, (10, 10), 100, 10),
    Robot(5, (5, 5), 100, 10)
]

tasks = [
    Task(1, (1, 1), 10, 1),
    Task(2, (8, 1), 10, 1),
    Task(3, (2, 9), 10, 1),
    Task(4, (9, 9), 10, 1)
]


# ============================================================
# CONSTRAINED PROBLEM
# ============================================================

problem = Problem(
    robots=robots,
    tasks=tasks,
    objective=total_distance,
    constraints=[
        battery_constraint,
        total_battery_constraint
    ]
)


# ============================================================
# BRUTE-FORCE SEARCH
# ============================================================

best_assignment = None
best_fitness = float("inf")

number_of_robots = len(robots)
number_of_tasks = len(tasks)

for assignment in itertools.product(
    range(1, number_of_robots + 1),
    repeat=number_of_tasks
):

    solution = Solution(list(assignment))

    fitness = problem.evaluate(solution)

    if fitness < best_fitness:
        best_fitness = fitness
        best_assignment = list(assignment)


print("\n--- CONSTRAINED BRUTE FORCE ---")

print(
    "Best assignment :",
    best_assignment
)

print(
    "Best fitness    :",
    best_fitness
)


# ============================================================
# SINGLE CONSTRAINED GWO
# ============================================================

print("\n--- CONSTRAINED DISCRETE GWO ---")

optimizer = DiscreteGWO(
    problem,
    population_size=10,
    max_iterations=50,
    seed=42
)

result = optimizer.optimize()

print(
    "Best solution :",
    result.variables
)

print(
    "Best fitness  :",
    result.fitness
)

print(
    "Initial fitness:",
    optimizer.history[0]
)

print(
    "Final fitness  :",
    optimizer.history[-1]
)

print(
    "History length:",
    len(optimizer.history)
)

print(
    "NFE:",
    optimizer.nfe
)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert len(result.variables) == number_of_tasks

for robot_id in result.variables:
    assert 1 <= robot_id <= number_of_robots


# Verify reported fitness

verified_fitness = problem.evaluate(
    Solution(result.variables)
)

assert abs(
    result.fitness - verified_fitness
) < 1e-9


# Verify history

assert len(optimizer.history) == 51


# Verify NFE

assert optimizer.nfe == 10 * 51


# ============================================================
# 20-SEED CONSTRAINED VALIDATION
# ============================================================

print("\n--- 20-SEED CONSTRAINED VALIDATION ---")

results = []

for seed in range(20):

    optimizer = DiscreteGWO(
        problem,
        population_size=10,
        max_iterations=50,
        seed=seed
    )

    result = optimizer.optimize()

    # Valid assignment

    assert len(result.variables) == number_of_tasks

    for robot_id in result.variables:

        assert (
            1 <= robot_id <= number_of_robots
        )

    # Fitness consistency

    verified_fitness = problem.evaluate(
        Solution(result.variables)
    )

    assert abs(
        result.fitness - verified_fitness
    ) < 1e-9

    # History

    assert len(optimizer.history) == 51

    # NFE

    assert optimizer.nfe == 10 * 51

    results.append(
        (
            seed,
            result.variables,
            result.fitness
        )
    )

    print(
        f"Seed {seed:2d}: "
        f"{result.variables} "
        f"-> {result.fitness:.6f}"
    )


# ============================================================
# SUMMARY
# ============================================================

best_run = min(
    results,
    key=lambda x: x[2]
)

average_fitness = sum(
    fitness
    for _, _, fitness in results
) / len(results)


exact_matches = sum(
    abs(fitness - best_fitness) < 1e-9
    for _, _, fitness in results
)


print("\n--- CONSTRAINED SUMMARY ---")

print(
    "Global optimum      :",
    best_fitness
)

print(
    "Global assignment   :",
    best_assignment
)

print(
    "Best GWO fitness    :",
    best_run[2]
)

print(
    "Best GWO assignment :",
    best_run[1]
)

print(
    "Average GWO fitness :",
    average_fitness
)

print(
    "Exact optimum runs  :",
    exact_matches,
    "/ 20"
)


print(
    "\nAll constrained "
    "Discrete GWO tests passed."
)