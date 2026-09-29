from kodane_library.algorithms.rvea import RVEA
from kodane_library.core.problem import Problem


# ============================================================
# TEST OBJECTIVE
# ============================================================

def multi_objective(x, problem):
    f1 = sum(value ** 2 for value in x)

    f2 = sum(
        (value - 2.0) ** 2
        for value in x
    )

    return [f1, f2]


def create_problem():
    return Problem(
        objective=multi_objective,
        dimension=5,
        bounds=[(-5, 5)] * 5
    )


# ============================================================
# TEST 1 — BASIC OPTIMIZATION
# ============================================================

def test_basic_optimization():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    result = optimizer.optimize()

    assert result is not None
    assert len(result) > 0

    print("PASS: Basic optimization")


# ============================================================
# TEST 2 — PARETO FRONT
# ============================================================

def test_pareto_front():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=30,
        max_iterations=20,
        seed=42
    )

    pareto = optimizer.optimize()

    assert len(pareto) > 0

    for solution in pareto:
        assert len(solution.objectives) == 2

    print("PASS: Pareto front generation")


# ============================================================
# TEST 3 — OBJECTIVE COUNT
# ============================================================

def test_objective_count():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    optimizer.optimize()

    assert optimizer.objective_count == 2

    print("PASS: Objective count")


# ============================================================
# TEST 4 — BOUNDS
# ============================================================

def test_bounds():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    pareto = optimizer.optimize()

    for solution in pareto:

        for value in solution.variables:

            assert -5.0 <= value <= 5.0

    print("PASS: Solutions within bounds")


# ============================================================
# TEST 5 — REPRODUCIBILITY
# ============================================================

def test_reproducibility():

    problem1 = create_problem()

    optimizer1 = RVEA(
        problem1,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    result1 = optimizer1.optimize()

    problem2 = create_problem()

    optimizer2 = RVEA(
        problem2,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    result2 = optimizer2.optimize()

    objectives1 = [
        solution.objectives
        for solution in result1
    ]

    objectives2 = [
        solution.objectives
        for solution in result2
    ]

    assert objectives1 == objectives2

    print("PASS: Reproducibility")


# ============================================================
# TEST 6 — DIFFERENT SEEDS
# ============================================================

def test_different_seeds():

    problem1 = create_problem()

    optimizer1 = RVEA(
        problem1,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    result1 = optimizer1.optimize()

    problem2 = create_problem()

    optimizer2 = RVEA(
        problem2,
        population_size=20,
        max_iterations=10,
        seed=123
    )

    result2 = optimizer2.optimize()

    objectives1 = [
        solution.objectives
        for solution in result1
    ]

    objectives2 = [
        solution.objectives
        for solution in result2
    ]

    assert objectives1 != objectives2

    print("PASS: Different seeds produce different results")


# ============================================================
# TEST 7 — HISTORY
# ============================================================

def test_history():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=20,
        max_iterations=10,
        seed=42
    )

    optimizer.optimize()

    assert len(optimizer.history) == 11

    print("PASS: History length")


# ============================================================
# TEST 8 — NFE
# ============================================================

def test_nfe():

    population_size = 20
    iterations = 10

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=population_size,
        max_iterations=iterations,
        seed=42
    )

    optimizer.optimize()

    expected_nfe = (
        population_size
        +
        population_size * iterations * 1
    )

    assert optimizer.nfe == expected_nfe

    print("PASS: NFE count")


# ============================================================
# TEST 9 — INVALID POPULATION
# ============================================================

def test_invalid_population():

    problem = create_problem()

    try:

        RVEA(
            problem,
            population_size=1,
            max_iterations=10
        )

        assert False

    except ValueError:
        pass

    print("PASS: Invalid population size rejected")


# ============================================================
# TEST 10 — INVALID ITERATIONS
# ============================================================

def test_invalid_iterations():

    problem = create_problem()

    try:

        RVEA(
            problem,
            population_size=20,
            max_iterations=-1
        )

        assert False

    except ValueError:
        pass

    print("PASS: Invalid iterations rejected")


# ============================================================
# TEST 11 — ZERO ITERATIONS
# ============================================================

def test_zero_iterations():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=20,
        max_iterations=0,
        seed=42
    )

    result = optimizer.optimize()

    assert len(result) > 0
    assert optimizer.nfe == 20
    assert len(optimizer.history) == 1

    print("PASS: Zero-iteration case")


# ============================================================
# TEST 12 — PARETO NON-DOMINATION
# ============================================================

def test_non_dominated():

    problem = create_problem()

    optimizer = RVEA(
        problem,
        population_size=30,
        max_iterations=20,
        seed=42
    )

    pareto = optimizer.optimize()

    for i in range(len(pareto)):

        for j in range(len(pareto)):

            if i == j:
                continue

            assert not optimizer._dominates(
                pareto[j],
                pareto[i]
            )

    print("PASS: Pareto solutions are non-dominated")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    test_basic_optimization()
    test_pareto_front()
    test_objective_count()
    test_bounds()
    test_reproducibility()
    test_different_seeds()
    test_history()
    test_nfe()
    test_invalid_population()
    test_invalid_iterations()
    test_zero_iterations()
    test_non_dominated()

    print()
    print("All RVEA tests passed.")