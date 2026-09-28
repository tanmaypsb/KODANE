import math


def rastrigin(solution, problem):
    return sum(
        x ** 2 - 10 * math.cos(2 * math.pi * x) + 10
        for x in solution
    )