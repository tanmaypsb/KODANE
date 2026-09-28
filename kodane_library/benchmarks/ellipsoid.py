def ellipsoid(solution, problem):
    n = len(solution)

    return sum(
        (10 ** (6 * i / (n - 1))) * x ** 2
        for i, x in enumerate(solution)
    )
