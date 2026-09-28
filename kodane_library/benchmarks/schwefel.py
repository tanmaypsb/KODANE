import math


def schwefel(solution, problem):
    n = len(solution)

    return (
        418.9829 * n
        - sum(
            x * math.sin(math.sqrt(abs(x)))
            for x in solution
        )
    )
