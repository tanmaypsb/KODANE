import math


def griewank(solution, problem):
    total = sum(
        x ** 2
        for x in solution
    ) / 4000

    product = 1.0

    for i, x in enumerate(solution):
        product *= math.cos(
            x / math.sqrt(i + 1)
        )

    return total - product + 1