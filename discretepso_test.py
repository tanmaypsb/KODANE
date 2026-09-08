from kodane_library.core.problem import Problem
from kodane_library.core.robot import Robot
from kodane_library.core.task import Task
from kodane_library.algorithms.discrete_pso import DiscretePSO
from kodane_library.core.solution import Solution
from kodane_library.objectives.distance import (
    total_distance,
    battery_constraint,
    total_battery_constraint
)
import itertools


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


problem = Problem(
    robots=robots,
    tasks=tasks,
    objective=total_distance
)


optimizer = DiscretePSO(
    problem,
    population_size=5,
    seed=42
)


best = optimizer.optimize()

print("Best solution:", best.variables)
print("Best fitness:", best.fitness)

print("\nParticles:")

for particle in optimizer.particles:
    print(
        "Position:",
        particle["position"],
        "| pbest:",
        particle["pbest"],
        "| pbest fitness:",
        particle["pbest_fitness"]
    )

print("\n--- DISCRETE PSO TEST ---")

optimizer = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

best = optimizer.optimize()

print("Best solution:", best.variables)
print("Best fitness:", best.fitness)
print("History length:", len(optimizer.history))
print("Initial fitness:", optimizer.history[0])
print("Final fitness:", optimizer.history[-1])


print("\n--- DISCRETE PSO VALIDATION ---")

history = optimizer.history

print("History:", history)

assert len(history) == 21

for i in range(len(history) - 1):
    assert history[i + 1] <= history[i]

print("Fitness history is non-increasing.")

for particle in optimizer.particles:
    position = particle["position"]

    assert len(position) == len(tasks)

    for robot_id in position:
        assert 1 <= robot_id <= len(robots)

print("All particle assignments are valid.")

print("\n--- DISCRETE PSO REPRODUCIBILITY TEST ---")

optimizer_1 = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

optimizer_2 = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

best_1 = optimizer_1.optimize()
best_2 = optimizer_2.optimize()

print("Run 1 solution:", best_1.variables)
print("Run 2 solution:", best_2.variables)

print("Run 1 fitness:", best_1.fitness)
print("Run 2 fitness:", best_2.fitness)

print("Same solution:", best_1.variables == best_2.variables)
print("Same fitness:", best_1.fitness == best_2.fitness)
print("Same history:", optimizer_1.history == optimizer_2.history)

assert best_1.variables == best_2.variables
assert best_1.fitness == best_2.fitness
assert optimizer_1.history == optimizer_2.history

print("Reproducibility test passed.")

print("\n--- DISCRETE PSO DIFFERENT SEED TEST ---")

optimizer_1 = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

optimizer_2 = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=123
)

best_1 = optimizer_1.optimize()
best_2 = optimizer_2.optimize()

print("Seed 42 solution:", best_1.variables)
print("Seed 123 solution:", best_2.variables)

print("Seed 42 fitness:", best_1.fitness)
print("Seed 123 fitness:", best_2.fitness)

print(
    "Different results:",
    best_1.variables != best_2.variables
    or best_1.fitness != best_2.fitness
)

print("\n--- KNOWN SOLUTION TEST ---")

known_solution = Solution([1, 2, 3, 4])

known_fitness = problem.evaluate(known_solution)

print("Known solution:", known_solution.variables)
print("Known fitness:", known_fitness)

print("\n--- BRUTE FORCE OPTIMUM TEST ---")

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

print("Brute force best assignment:", best_assignment)
print("Brute force best fitness:", best_fitness)

print("\n--- PSO VS BRUTE FORCE ---")

# A single stochastic run is not guaranteed to find the global optimum.
# Instead we:
#   1. Verify each PSO run returns a legal solution.
#   2. Verify the returned fitness is consistent with problem.evaluate().
#   3. Verify that at least one run across several seeds discovers the
#      brute-force global optimum (this is tractable: 5^4 = 625 candidates).

_pso_found_optimum = False

for _seed in range(20):
    _opt = DiscretePSO(
        problem,
        population_size=10,
        max_iterations=50,
        seed=_seed
    )

    _result = _opt.optimize()

    # 1. Legal solution: correct length, all valid robot IDs.
    assert len(_result.variables) == len(tasks), (
        f"Seed {_seed}: solution length mismatch"
    )
    for _rid in _result.variables:
        assert 1 <= _rid <= len(robots), (
            f"Seed {_seed}: invalid robot ID {_rid}"
        )

    # 2. Fitness in Solution must equal re-evaluating the same variables.
    _re_eval = problem.evaluate(Solution(_result.variables))
    assert abs(_result.fitness - _re_eval) < 1e-12, (
        f"Seed {_seed}: fitness inconsistency "
        f"({_result.fitness} vs {_re_eval})"
    )

    if _result.variables == best_assignment:
        _pso_found_optimum = True

    print(
        f"Seed {_seed:2d}: {_result.variables}  "
        f"fitness={_result.fitness:.6f}  "
        f"optimum={'YES' if _result.variables == best_assignment else 'no'}"
    )

print("Brute force optimum:", best_assignment, "fitness:", best_fitness)
assert _pso_found_optimum, (
    "PSO did not discover the brute-force global optimum in any of the "
    "20 runs. This may indicate a bug in the update logic."
)

print("Discrete PSO found the global optimum in at least one run.")

print("\n--- BATTERY CONSTRAINT TEST ---")

battery_problem = Problem(
    robots=[
        Robot(1, (0, 0), 20, 10),
        Robot(2, (10, 0), 100, 10),
        Robot(3, (0, 10), 80, 10),
    ],
    tasks=[
        Task(1, (1, 1), 30, 1),
        Task(2, (8, 1), 50, 1),
    ],
    objective=total_distance,
    constraints=[battery_constraint]
)

solution_1 = Solution([1, 2])
solution_2 = Solution([2, 2])

print(
    "Solution [1, 2] violation:",
    battery_constraint(solution_1.variables, battery_problem)
)

print(
    "Solution [2, 2] violation:",
    battery_constraint(solution_2.variables, battery_problem)
)

print("\n--- CONSTRAINT EVALUATION TEST ---")

solution_1 = Solution([1, 2])
solution_2 = Solution([2, 2])

fitness_1 = battery_problem.evaluate(solution_1)
fitness_2 = battery_problem.evaluate(solution_2)

print("Solution [1, 2] fitness:", fitness_1)
print("Solution [2, 2] fitness:", fitness_2)

print("Penalty applied:", fitness_1 > fitness_2)

assert fitness_1 > fitness_2

print("Constraint evaluation test passed.")

print("\n--- DISCRETE PSO WITH BATTERY CONSTRAINT ---")

constrained_optimizer = DiscretePSO(
    battery_problem,
    population_size=20,
    max_iterations=30,
    seed=123
)

constrained_best = constrained_optimizer.optimize()

print("Best assignment:", constrained_best.variables)
print("Best fitness:", constrained_best.fitness)

violation = battery_constraint(
    constrained_best.variables,
    battery_problem
)

print("Battery violation:", violation)

assert violation == 0

print("Discrete PSO found a battery-feasible solution.")

print("\n--- TOTAL BATTERY CONSTRAINT TEST ---")

total_battery_problem = Problem(
    robots=[
        Robot(1, (0, 0), 50, 10),
        Robot(2, (10, 0), 100, 10),
    ],
    tasks=[
        Task(1, (1, 1), 30, 1),
        Task(2, (8, 1), 40, 1),
        Task(3, (5, 5), 50, 1),
    ],
    objective=total_distance,
    constraints=[total_battery_constraint]
)

solution = Solution([2, 2, 2])

violation = total_battery_constraint(
    solution.variables,
    total_battery_problem
)

print("Assignment:", solution.variables)
print("Battery violation:", violation)

assert violation == 20.0

print("Total battery constraint test passed.")

feasible_solution = Solution([2, 2, 1])

feasible_violation = total_battery_constraint(
    feasible_solution.variables,
    total_battery_problem
)

print("Feasible assignment:", feasible_solution.variables)
print("Feasible violation:", feasible_violation)

assert feasible_violation == 0.0

print("Feasible assignment test passed.")

print("\n--- DISCRETE PSO WITH TOTAL BATTERY CONSTRAINT ---")

constrained_optimizer = DiscretePSO(
    total_battery_problem,
    population_size=20,
    max_iterations=30,
    seed=123
)

constrained_best = constrained_optimizer.optimize()

print("Best assignment:", constrained_best.variables)
print("Best fitness:", constrained_best.fitness)

violation = total_battery_constraint(
    constrained_best.variables,
    total_battery_problem
)

print("Battery violation:", violation)

assert violation == 0.0

print("Discrete PSO found a feasible solution.")

print("\n--- CONSTRAINED PSO VS BRUTE FORCE ---")

brute_best_assignment = None
brute_best_fitness = float("inf")

number_of_robots = len(total_battery_problem.robots)
number_of_tasks = len(total_battery_problem.tasks)

for assignment in itertools.product(
    range(1, number_of_robots + 1),
    repeat=number_of_tasks
):
    solution = Solution(list(assignment))

    violation = total_battery_constraint(
        solution.variables,
        total_battery_problem
    )

    if violation > 0:
        continue

    fitness = total_battery_problem.evaluate(solution)

    if fitness < brute_best_fitness:
        brute_best_fitness = fitness
        brute_best_assignment = list(assignment)

print("PSO assignment:", constrained_best.variables)
print("PSO fitness:", constrained_best.fitness)

print("Brute force assignment:", brute_best_assignment)
print("Brute force fitness:", brute_best_fitness)

# Regression test: for this specific tiny problem (2 robots, 3 tasks,
# seed=123), the constrained feasible space is small enough that PSO
# reliably finds the optimum. This is not a general guarantee.
assert constrained_best.variables == brute_best_assignment
assert constrained_best.fitness == brute_best_fitness

print("Constrained PSO matched brute-force optimum (seed=123, tiny problem).")

print("\n--- DISCRETE PSO PARAMETER VALIDATION ---")

try:
    DiscretePSO(
        problem,
        population_size=0
    )
    assert False, "Expected ValueError for population_size=0"
except ValueError:
    pass

try:
    DiscretePSO(
        problem,
        max_iterations=-1
    )
    assert False, "Expected ValueError for max_iterations=-1"
except ValueError:
    pass

try:
    DiscretePSO(
        problem,
        inertia=-1
    )
    assert False, "Expected ValueError for negative inertia"
except ValueError:
    pass

try:
    DiscretePSO(
        problem,
        cognitive=-1
    )
    assert False, "Expected ValueError for negative cognitive"
except ValueError:
    pass

try:
    DiscretePSO(
        problem,
        social=-1
    )
    assert False, "Expected ValueError for negative social"
except ValueError:
    pass

try:
    DiscretePSO(
        problem,
        population_size=2.5
    )
    assert False, "Expected TypeError for non-integer population_size"
except TypeError:
    pass

try:
    DiscretePSO(
        problem,
        max_iterations="100"
    )
    assert False, "Expected TypeError for non-integer max_iterations"
except TypeError:
    pass

try:
    DiscretePSO(
        problem,
        inertia="0.7"
    )
    assert False, "Expected TypeError for non-numeric inertia"
except TypeError:
    pass

print("Parameter validation tests passed.")

print("\n--- DISCRETE PSO EDGE CASE TESTS ---")

# Population size = 1
single_particle = DiscretePSO(
    problem,
    population_size=1,
    max_iterations=5,
    seed=42
)

single_result = single_particle.optimize()

assert len(single_result.variables) == len(problem.tasks)
assert single_result.fitness is not None

print("Population size = 1: passed.")


# Zero iterations
zero_iteration = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=0,
    seed=42
)

zero_result = zero_iteration.optimize()

assert zero_result.fitness is not None
assert len(zero_iteration.history) == 1

print("Zero iterations: passed.")


# Repeated optimization on the same object
optimizer = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

result_1 = optimizer.optimize()
history_1 = optimizer.history.copy()

result_2 = optimizer.optimize()
history_2 = optimizer.history.copy()

assert result_1.variables == result_2.variables
assert result_1.fitness == result_2.fitness
assert history_1 == history_2

print("Repeated optimization: passed.")

print("All edge case tests passed.")


print("\n--- ZERO-WEIGHT EDGE CASE TESTS ---")

# A) inertia=0, cognitive=0, social=1.5 — purely social: must run successfully.
social_only = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    inertia=0,
    cognitive=0,
    social=1.5,
    seed=42
)

social_result = social_only.optimize()

assert len(social_result.variables) == len(tasks)
assert social_result.fitness is not None

for _rid in social_result.variables:
    assert 1 <= _rid <= len(robots)

print("Zero inertia/cognitive, social=1.5: passed.")


# B) inertia=0.7, cognitive=0, social=0 — purely inertia: must run successfully.
inertia_only = DiscretePSO(
    problem,
    population_size=10,
    max_iterations=20,
    inertia=0.7,
    cognitive=0,
    social=0,
    seed=42
)

inertia_result = inertia_only.optimize()

assert len(inertia_result.variables) == len(tasks)
assert inertia_result.fitness is not None

for _rid in inertia_result.variables:
    assert 1 <= _rid <= len(robots)

print("Inertia=0.7, zero cognitive/social: passed.")


# C) inertia=0, cognitive=0, social=0 — all-zero weights: must raise ValueError.
try:
    DiscretePSO(
        problem,
        inertia=0,
        cognitive=0,
        social=0
    )
    assert False, "Expected ValueError for all-zero weights"
except ValueError as _e:
    assert "Sum of weights must be greater than zero" in str(_e), (
        f"Unexpected ValueError message: {_e}"
    )

print("All-zero weights raises ValueError: passed.")

print("Zero-weight edge case tests passed.")