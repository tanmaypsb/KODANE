import math


def ackley(solution, problem):
    n = len(solution)

    sum1 = sum(x ** 2 for x in solution)

    sum2 = sum(
        math.cos(2 * math.pi * x)
        for x in solution
    )

    return (
        -20 * math.exp(-0.2 * math.sqrt(sum1 / n))
        - math.exp(sum2 / n)
        + 20
        + math.e
    )