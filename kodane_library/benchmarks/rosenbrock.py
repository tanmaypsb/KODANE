def rosenbrock(solution, problem):
    return sum(
        100 * (solution[i + 1] - solution[i] ** 2) ** 2
        + (solution[i] - 1) ** 2
        for i in range(len(solution) - 1)
    )