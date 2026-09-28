from kodane_library.algorithms.cpo import CPO
from kodane_library.core.problem import Problem


def sphere(x, problem):
    return sum(v ** 2 for v in x)


def make_problem():
    return Problem(
        objective=sphere,
        dimension=5,
        bounds=[(-10, 10)] * 5
    )


def test_basic():
    optimizer = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    result = optimizer.optimize()

    assert result.fitness >= 0
    assert len(result.variables) == 5
    print("PASS: Basic optimization")


def test_convergence():
    optimizer = CPO(
        make_problem(),
        population_size=20,
        max_iterations=50,
        seed=42
    )

    result = optimizer.optimize()

    assert result.fitness < 1.0
    print("PASS: Sphere convergence")


def test_bounds():
    optimizer = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    result = optimizer.optimize()

    for value in result.variables:
        assert -10 <= value <= 10

    print("PASS: Solution within bounds")


def test_reproducibility():
    optimizer1 = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    optimizer2 = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    result1 = optimizer1.optimize()
    result2 = optimizer2.optimize()

    assert result1.fitness == result2.fitness
    assert result1.variables == result2.variables

    print("PASS: Reproducibility")


def test_different_seeds():
    optimizer1 = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    optimizer2 = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=99
    )

    result1 = optimizer1.optimize()
    result2 = optimizer2.optimize()

    assert (
        result1.fitness != result2.fitness
        or result1.variables != result2.variables
    )

    print("PASS: Different seeds")


def test_history():
    optimizer = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    optimizer.optimize()

    assert len(optimizer.history) == 21

    for i in range(1, len(optimizer.history)):
        assert (
            optimizer.history[i]
            <=
            optimizer.history[i - 1]
        )

    print("PASS: History")


def test_nfe():
    optimizer = CPO(
        make_problem(),
        population_size=10,
        max_iterations=20,
        seed=42
    )

    optimizer.optimize()

    assert optimizer.nfe >= 10

    print("PASS: NFE")


def test_zero_iterations():
    optimizer = CPO(
        make_problem(),
        population_size=10,
        max_iterations=0,
        seed=42
    )

    result = optimizer.optimize()

    assert len(optimizer.history) == 1
    assert optimizer.nfe == 10
    assert result.fitness >= 0

    print("PASS: Zero-iteration case")


def test_invalid_population():
    try:
        CPO(
            make_problem(),
            population_size=0
        )
        assert False
    except ValueError:
        pass

    print("PASS: Invalid population size rejected")


def test_invalid_iterations():
    try:
        CPO(
            make_problem(),
            max_iterations=-1
        )
        assert False
    except ValueError:
        pass

    print("PASS: Invalid iterations rejected")


def run_tests():

    test_basic()
    test_convergence()
    test_bounds()
    test_reproducibility()
    test_different_seeds()
    test_history()
    test_nfe()
    test_zero_iterations()
    test_invalid_population()
    test_invalid_iterations()

    print()
    print("All CPO tests passed.")


if __name__ == "__main__":
    run_tests()
