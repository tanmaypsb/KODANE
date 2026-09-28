import random

from kodane_library.core.optimizer import Optimizer
from kodane_library.core.solution import Solution


class DiscreteGWO(Optimizer):

    """
    Discrete Grey Wolf Optimization for assignment problems.

    Each wolf represents an assignment:
        [robot_id, robot_id, robot_id, ...]
    """

    def __init__(
        self,
        problem,
        population_size=30,
        max_iterations=100,
        seed=None
    ):
        if not isinstance(population_size, int):
            raise TypeError("population_size must be an integer.")

        if population_size < 3:
            raise ValueError(
                "population_size must be at least 3."
            )

        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer.")

        if max_iterations < 0:
            raise ValueError(
                "max_iterations cannot be negative."
            )

        super().__init__(
            problem,
            population_size,
            max_iterations
        )

        self.seed = seed
        self.rng = random.Random(seed)

        self.wolves = []

        self.alpha = None
        self.beta = None
        self.delta = None

        self.alpha_fitness = None
        self.beta_fitness = None
        self.delta_fitness = None

        self.history = []
        self.nfe = 0

    def initialize(self):

        self.rng = random.Random(self.seed)

        self.wolves = []
        self.history = []

        self.best_solution = None
        self.best_fitness = None

        self.alpha = None
        self.beta = None
        self.delta = None

        self.alpha_fitness = None
        self.beta_fitness = None
        self.delta_fitness = None

        number_of_tasks = len(self.problem.tasks)
        number_of_robots = len(self.problem.robots)

        for _ in range(self.population_size):

            position = [
                self.rng.randint(1, number_of_robots)
                for _ in range(number_of_tasks)
            ]

            self.wolves.append(position)

    def _evaluate(self, position):

        solution = Solution(position)

        fitness = self.problem.evaluate(solution)

        self.nfe += 1

        return fitness

    def _update_leaders(self, position, fitness):

        if (
            self.alpha_fitness is None
            or fitness < self.alpha_fitness
        ):
            self.delta = (
                self.beta.copy()
                if self.beta is not None
                else None
            )
            self.delta_fitness = self.beta_fitness

            self.beta = (
                self.alpha.copy()
                if self.alpha is not None
                else None
            )
            self.beta_fitness = self.alpha_fitness

            self.alpha = position.copy()
            self.alpha_fitness = fitness

        elif (
            self.beta_fitness is None
            or fitness < self.beta_fitness
        ):
            self.delta = (
                self.beta.copy()
                if self.beta is not None
                else None
            )
            self.delta_fitness = self.beta_fitness

            self.beta = position.copy()
            self.beta_fitness = fitness

        elif (
            self.delta_fitness is None
            or fitness < self.delta_fitness
        ):
            self.delta = position.copy()
            self.delta_fitness = fitness

        if (
            self.best_fitness is None
            or fitness < self.best_fitness
        ):
            self.best_fitness = fitness
            self.best_solution = position.copy()

    def _update_position(self, position, a):

        new_position = []

        leader_probability = 1.0 - (a / 2.0)

        for i in range(len(position)):

            current_value = position[i]

            alpha_value = self.alpha[i]
            beta_value = self.beta[i]
            delta_value = self.delta[i]

            if self.rng.random() < leader_probability:

                new_value = self.rng.choice([
                    alpha_value,
                    beta_value,
                    delta_value
                ])

            else:

                new_value = current_value

            new_position.append(new_value)

        return new_position

    def optimize(self):

        self.initialize()

        # Evaluate initial population

        for position in self.wolves:

            fitness = self._evaluate(position)

            self._update_leaders(
                position,
                fitness
            )

        self.history = [self.best_fitness]

        # Main optimization loop

        for iteration in range(self.max_iterations):

            if self.max_iterations == 0:
                a = 0.0
            else:
                a = (
                    2.0
                    - 2.0
                    * iteration
                    / self.max_iterations
                )

            new_wolves = []

            for position in self.wolves:

                new_position = self._update_position(
                    position,
                    a
                )

                new_wolves.append(new_position)

            self.wolves = new_wolves

            for position in self.wolves:

                fitness = self._evaluate(position)

                self._update_leaders(
                    position,
                    fitness
                )

            self.history.append(
                self.best_fitness
            )

        best_solution = Solution(
            self.best_solution
        )

        best_solution.set_fitness(
            self.best_fitness
        )

        return best_solution