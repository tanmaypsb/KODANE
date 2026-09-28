import math


def step(solution, problem):
    return sum(
        math.floor(x + 0.5) ** 2
        for x in solution
    )
