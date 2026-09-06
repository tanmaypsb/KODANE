class Solution:

    def __init__(self, variables):
        self.variables = variables
        self.fitness = None

    def set_fitness(self, fitness):
        self.fitness = fitness

    def __repr__(self):
        return (
            f"Solution("
            f"variables={self.variables}, "
            f"fitness={self.fitness})"
        )