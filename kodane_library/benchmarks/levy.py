import math


def levy(solution, problem):
    w = [
        1 + (x - 1) / 4
        for x in solution
    ]

    term1 = math.sin(math.pi * w[0]) ** 2

    term2 = sum(
        (w[i] - 1) ** 2
        * (1 + 10 * math.sin(math.pi * w[i] + 1) ** 2)
        for i in range(len(w) - 1)
    )

    term3 = (
        (w[-1] - 1) ** 2
        * (1 + math.sin(2 * math.pi * w[-1]) ** 2)
    )

    return term1 + term2 + term3
