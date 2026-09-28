def bent_cigar(solution, problem):
    return (
        solution[0] ** 2
        + 10 ** 6 * sum(x ** 2 for x in solution[1:])
    )
