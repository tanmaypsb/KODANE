def discus(solution, problem):
    return (
        10 ** 6 * solution[0] ** 2
        + sum(x ** 2 for x in solution[1:])
    )
