def zakharov(solution, problem):
    sum1 = sum(x ** 2 for x in solution)

    sum2 = sum(
        0.5 * (i + 1) * x
        for i, x in enumerate(solution)
    )

    return sum1 + sum2 ** 2 + sum2 ** 4
