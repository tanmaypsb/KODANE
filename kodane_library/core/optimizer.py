class Optimizer:

    def __init__(self, problem, population_size=10, max_iterations=100):
        self.problem = problem
        self.population_size = population_size
        self.max_iterations = max_iterations

        self.best_solution = None
        self.best_fitness = None

    def optimize(self):
        raise NotImplementedError(
            "The optimize() method must be implemented by the algorithm."
        )