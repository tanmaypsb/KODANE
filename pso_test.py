from kodane_library.core.problem import Problem
from kodane_library.algorithms.pso import PSO
from kodane_library.benchmarks.sphere import sphere
from kodane_library.core.problem import Problem
from kodane_library.core.robot import Robot
from kodane_library.core.task import Task


def shifted_sphere(solution, problem):
    return sum((x - 3) ** 2 for x in solution)


problem = Problem(
    objective=sphere,
    dimension=5,
     bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

pso1 = PSO(
    problem=problem,
    population_size=30,
    max_iterations=100,
    seed=42
)

pso2 = PSO(
    problem=problem,
    population_size=30,
    max_iterations=100,
    seed=42
)

best_solution_1 = pso1.optimize()

best_solution_2 = pso2.optimize()

print("PSO 1 best fitness:", best_solution_1.fitness)
print("PSO 2 best fitness:", best_solution_2.fitness)

print("PSO 1 best solution:", best_solution_1.variables)
print("PSO 2 best solution:", best_solution_2.variables)

print("PSO 1 history length:", len(pso1.history))
print("PSO 2 history length:", len(pso2.history))

print("\n--- VELOCITY LIMIT TEST ---")

for particle in pso1.particles:
    for i, velocity in enumerate(particle["velocity"]):
        vmax = pso1.velocity_limits[i]

        assert -vmax <= velocity <= vmax, (
            f"Velocity limit exceeded: "
            f"{velocity} not in [-{vmax}, {vmax}]"
        )

print("All particle velocities are within their limits.")

print("\n--- DIFFERENT BOUNDS TEST ---")

different_bounds_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (0, 100),
        (-1, 1)
    ]
)

different_bounds_pso = PSO(
    problem=different_bounds_problem,
    population_size=20,
    max_iterations=50,
    seed=42
)

result = different_bounds_pso.optimize()

print("Best solution:", result.variables)
print("Best fitness:", result.fitness)
print("Velocity limits:", different_bounds_pso.velocity_limits)

print("\n--- SMALL BOUNDS TEST ---")

small_bounds_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-0.001, 0.001),
        (-0.0001, 0.0001),
        (-1e-6, 1e-6)
    ]
)

small_bounds_pso = PSO(
    problem=small_bounds_problem,
    population_size=20,
    max_iterations=50,
    seed=42
)

small_result = small_bounds_pso.optimize()

print("Best solution:", small_result.variables)
print("Best fitness:", small_result.fitness)
print("Velocity limits:", small_bounds_pso.velocity_limits)

print("\n--- ONE DIMENSION TEST ---")

one_d_problem = Problem(
    objective=sphere,
    dimension=1,
    bounds=[(-10, 10)]
)

one_d_pso = PSO(
    problem=one_d_problem,
    population_size=10,
    max_iterations=50,
    seed=42
)

one_d_result = one_d_pso.optimize()

print("Best solution:", one_d_result.variables)
print("Best fitness:", one_d_result.fitness)
print("Velocity limits:", one_d_pso.velocity_limits)
print("History length:", len(one_d_pso.history))

print("\n--- SINGLE PARTICLE TEST ---")

single_particle_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

single_particle_pso = PSO(
    problem=single_particle_problem,
    population_size=1,
    max_iterations=20,
    seed=42
)

single_result = single_particle_pso.optimize()

print("Best solution:", single_result.variables)
print("Best fitness:", single_result.fitness)
print("History length:", len(single_particle_pso.history))

print("\n--- ONE ITERATION TEST ---")

one_iteration_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

one_iteration_pso = PSO(
    problem=one_iteration_problem,
    population_size=10,
    max_iterations=1,
    seed=42
)

one_iteration_result = one_iteration_pso.optimize()

print("Best solution:", one_iteration_result.variables)
print("Best fitness:", one_iteration_result.fitness)
print("History length:", len(one_iteration_pso.history))

print("\n--- REPEATED OPTIMIZATION TEST ---")

repeat_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

repeat_pso = PSO(
    problem=repeat_problem,
    population_size=10,
    max_iterations=20,
    seed=42
)

first_result = repeat_pso.optimize()

print("First run fitness:", first_result.fitness)
print("First run history:", len(repeat_pso.history))

second_result = repeat_pso.optimize()

print("Second run fitness:", second_result.fitness)
print("Second run history:", len(repeat_pso.history))

print("\n--- INVALID COEFFICIENT TEST ---")

test_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

try:
    PSO(test_problem, inertia=-0.1)
except ValueError as e:
    print("Negative inertia:", e)

try:
    PSO(test_problem, cognitive=-0.1)
except ValueError as e:
    print("Negative cognitive:", e)

try:
    PSO(test_problem, social=-0.1)
except ValueError as e:
    print("Negative social:", e)


print("\n--- ZERO COEFFICIENT TEST ---")

zero_coefficient_pso = PSO(
    test_problem,
    population_size=10,
    max_iterations=10,
    inertia=0,
    cognitive=0,
    social=0,
    seed=42
)

zero_result = zero_coefficient_pso.optimize()

print("Best fitness:", zero_result.fitness)
print("History length:", len(zero_coefficient_pso.history))    

print("\n--- INVALID BOUNDS TEST ---")

try:
    Problem(
        objective=sphere,
        dimension=3,
        bounds=[
            (-10, 10),
            (-10, 10)
        ]
    )
except ValueError as e:
    print("Wrong number of bounds:", e)

print("\n--- INVALID BOUND SHAPE TEST ---")

try:
    Problem(
        objective=sphere,
        dimension=3,
        bounds=[
            (-10, 10),
            (-5, 5, 10),
            (-1, 1)
        ]
    )
except ValueError as e:
    print("Invalid bound shape:", e)


print("\n--- NON-NUMERIC BOUNDS TEST ---")

try:
    Problem(
        objective=sphere,
        dimension=3,
        bounds=[
            (-10, 10),
            ("low", 10),
            (-1, 1)
        ]
    )
except (ValueError, TypeError) as e:
    print("Non-numeric bound:", e)

print("\n--- NON-FINITE BOUNDS TEST ---")

for bad_bound in [
    (float("nan"), 10),
    (-10, float("inf")),
    (float("-inf"), 10)
]:
    try:
        Problem(
            objective=sphere,
            dimension=1,
            bounds=[bad_bound]
        )
    except ValueError as e:
        print("Invalid bound:", e)


print("\n--- EQUAL BOUNDS TEST ---")

equal_bounds_problem = Problem(
    objective=sphere,
    dimension=1,
    bounds=[(5, 5)]
)

equal_bounds_pso = PSO(
    problem=equal_bounds_problem,
    population_size=10,
    max_iterations=10,
    seed=42
)

equal_result = equal_bounds_pso.optimize()

print("Best solution:", equal_result.variables)
print("Best fitness:", equal_result.fitness)
print("Velocity limits:", equal_bounds_pso.velocity_limits)

print("\n--- POSITION BOUNDARY TEST ---")

boundary_problem = Problem(
    objective=sphere,
    dimension=1,
    bounds=[(-1, 1)]
)

boundary_pso = PSO(
    problem=boundary_problem,
    population_size=20,
    max_iterations=50,
    seed=42
)

boundary_result = boundary_pso.optimize()

for particle in boundary_pso.particles:
    position = particle["position"][0]

    assert -1 <= position <= 1, (
        f"Position escaped bounds: {position}"
    )

print("All particle positions are within bounds.")
print("Best solution:", boundary_result.variables)
print("Best fitness:", boundary_result.fitness)


print("\n--- CONVERGENCE HISTORY TEST ---")

history_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

history_pso = PSO(
    problem=history_problem,
    population_size=20,
    max_iterations=50,
    seed=42
)

history_result = history_pso.optimize()

for i in range(1, len(history_pso.history)):
    assert history_pso.history[i] <= history_pso.history[i - 1], (
        f"History increased at index {i}: "
        f"{history_pso.history[i - 1]} -> {history_pso.history[i]}"
    )

print("Convergence history is non-increasing.")
print("Initial fitness:", history_pso.history[0])
print("Final fitness:", history_pso.history[-1])
print("History length:", len(history_pso.history))

print("\n--- DIFFERENT OBJECTIVE TEST ---")

shifted_problem = Problem(
    objective=shifted_sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

shifted_pso = PSO(
    problem=shifted_problem,
    population_size=30,
    max_iterations=100,
    seed=42
)

shifted_result = shifted_pso.optimize()

print("Best solution:", shifted_result.variables)
print("Best fitness:", shifted_result.fitness)


print("\n--- DIFFERENT SEED TEST ---")

seed_problem = Problem(
    objective=sphere,
    dimension=3,
    bounds=[
        (-10, 10),
        (-10, 10),
        (-10, 10)
    ]
)

pso_seed_42 = PSO(
    problem=seed_problem,
    population_size=20,
    max_iterations=30,
    seed=42
)

pso_seed_123 = PSO(
    problem=seed_problem,
    population_size=20,
    max_iterations=30,
    seed=123
)

result_42 = pso_seed_42.optimize()
result_123 = pso_seed_123.optimize()

print("Seed 42 fitness:", result_42.fitness)
print("Seed 123 fitness:", result_123.fitness)

assert (
    result_42.variables != result_123.variables
    or result_42.fitness != result_123.fitness
), "Different seeds produced identical results."

print("Different seeds produce independent results.")


print("\n--- FITNESS CONSISTENCY TEST ---")

calculated_fitness = shifted_sphere(
    shifted_result.variables,
    shifted_problem
)

print("Reported fitness:", shifted_result.fitness)
print("Calculated fitness:", calculated_fitness)

assert abs(
    shifted_result.fitness - calculated_fitness
) < 1e-12, (
    "Returned fitness does not match the objective function."
)

print("Fitness is consistent with the objective function.")


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
    objective=lambda solution, problem: 0,
)

optimizer = DiscretePSO(
    problem,
    population_size=5,
    seed=42
)

optimizer.initialize()

for particle in optimizer.particles:
    print(particle["position"])