import random

from kodane_library.core.optimizer import Optimizer
from kodane_library.core.solution import Solution


class GWO(Optimizer):
    """
    Grey Wolf Optimization for continuous optimization problems.

    Each wolf represents a continuous candidate solution.

    The three best wolves:
        Alpha -> best solution
        Beta  -> second-best solution
        Delta -> third-best solution

    guide the rest of the population.
    """

    def __init__(
        self,
        problem,
        population_size=30,
        max_iterations=100,
        seed=None
    ):
        """
        Initialize the Grey Wolf Optimizer.

        Parameters:
            problem: Continuous optimization problem.
            population_size: Number of wolves.
            max_iterations: Number of optimization iterations.
            seed: Random seed for reproducibility.
        """

        if not isinstance(population_size, int):
            raise TypeError("population_size must be an integer.")

        if population_size < 3:
            raise ValueError(
                "population_size must be at least 3 for GWO."
            )

        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer.")

        if max_iterations < 0:
            raise ValueError(
                "max_iterations cannot be negative."
            )

        if problem.dimension is None:
            raise ValueError(
                "GWO requires a problem dimension."
            )

        if problem.bounds is None:
            raise ValueError(
                "GWO requires bounds for every dimension."
            )

        if len(problem.bounds) != problem.dimension:
            raise ValueError(
                "Number of bounds must match problem dimension."
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

        # Number of objective evaluations
        self.nfe = 0

    def initialize(self):
        """
        Randomly initialize all wolves inside the problem bounds.
        """

        self.rng = random.Random(self.seed)

        self.wolves = []

        self.alpha = None
        self.beta = None
        self.delta = None

        self.alpha_fitness = None
        self.beta_fitness = None
        self.delta_fitness = None

        self.best_solution = None
        self.best_fitness = None

        self.history = []
        self.nfe = 0

        for _ in range(self.population_size):

            position = []

            for lower, upper in self.problem.bounds:

                value = self.rng.uniform(
                    lower,
                    upper
                )

                position.append(value)

            self.wolves.append(position)

    def _evaluate_wolf(self, position):
        """
        Evaluate a wolf using the KODANE Problem interface.
        """

        solution = Solution(position)

        fitness = self.problem.evaluate(solution)

        self.nfe += 1

        return solution, fitness

    def _update_leaders(self, evaluated_wolves):
        """
        Identify Alpha, Beta and Delta wolves.

        evaluated_wolves:
            List of (position, solution, fitness)
        """

        # Sort by fitness.
        # KODANE currently assumes minimization.
        evaluated_wolves.sort(
            key=lambda item: item[2]
        )

        # Best three wolves
        alpha_position = evaluated_wolves[0][0]
        alpha_solution = evaluated_wolves[0][1]
        alpha_fitness = evaluated_wolves[0][2]

        beta_position = evaluated_wolves[1][0]
        beta_fitness = evaluated_wolves[1][2]

        delta_position = evaluated_wolves[2][0]
        delta_fitness = evaluated_wolves[2][2]

        self.alpha = alpha_position.copy()
        self.beta = beta_position.copy()
        self.delta = delta_position.copy()

        self.alpha_fitness = alpha_fitness
        self.beta_fitness = beta_fitness
        self.delta_fitness = delta_fitness

        # Update global best
        if (
            self.best_fitness is None
            or alpha_fitness < self.best_fitness
        ):
            self.best_fitness = alpha_fitness

            self.best_solution = Solution(
                alpha_position.copy()
            )

            self.best_solution.set_fitness(
                alpha_fitness
            )

    def _update_position(self, position, a):
        """
        Calculate the next position of one wolf.

        Standard GWO equations:

            A = 2*a*r1 - a
            C = 2*r2

            D = |C*leader - X|

            X_new = leader - A*D

        The new position is the average of the
        positions calculated using Alpha, Beta
        and Delta.
        """

        new_position = []

        for dimension_index, current_value in enumerate(position):

            # -------------------------
            # Alpha
            # -------------------------

            r1 = self.rng.random()
            r2 = self.rng.random()

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = abs(
                C1 * self.alpha[dimension_index]
                - current_value
            )

            X1 = (
                self.alpha[dimension_index]
                - A1 * D_alpha
            )

            # -------------------------
            # Beta
            # -------------------------

            r1 = self.rng.random()
            r2 = self.rng.random()

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = abs(
                C2 * self.beta[dimension_index]
                - current_value
            )

            X2 = (
                self.beta[dimension_index]
                - A2 * D_beta
            )

            # -------------------------
            # Delta
            # -------------------------

            r1 = self.rng.random()
            r2 = self.rng.random()

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = abs(
                C3 * self.delta[dimension_index]
                - current_value
            )

            X3 = (
                self.delta[dimension_index]
                - A3 * D_delta
            )

            # Average of Alpha, Beta and Delta
            new_value = (
                X1 + X2 + X3
            ) / 3.0

            # Keep the wolf inside its bounds
            lower, upper = self.problem.bounds[
                dimension_index
            ]

            new_value = max(
                lower,
                min(upper, new_value)
            )

            new_position.append(new_value)

        return new_position

    def optimize(self):
        """
        Run the Grey Wolf Optimization algorithm.

        Returns:
            Best Solution found by GWO.
        """

        self.initialize()

        # --------------------------------------------------
        # Initial population evaluation
        # --------------------------------------------------

        evaluated_wolves = []

        for position in self.wolves:

            solution, fitness = self._evaluate_wolf(
                position
            )

            evaluated_wolves.append(
                (
                    position,
                    solution,
                    fitness
                )
            )

        self._update_leaders(
            evaluated_wolves
        )

        self.history.append(
            self.best_fitness
        )

        # --------------------------------------------------
        # Main GWO loop
        # --------------------------------------------------

        for iteration in range(
            self.max_iterations
        ):

            # Linearly decrease a from 2 to 0
            a = 2 - (
                2 * iteration
                / self.max_iterations
            ) if self.max_iterations > 0 else 0

            new_wolves = []

            for position in self.wolves:

                new_position = self._update_position(
                    position,
                    a
                )

                new_wolves.append(
                    new_position
                )

            self.wolves = new_wolves

            # Evaluate new population
            evaluated_wolves = []

            for position in self.wolves:

                solution, fitness = self._evaluate_wolf(
                    position
                )

                evaluated_wolves.append(
                    (
                        position,
                        solution,
                        fitness
                    )
                )

            # Update Alpha, Beta and Delta
            self._update_leaders(
                evaluated_wolves
            )

            # Store best fitness
            self.history.append(
                self.best_fitness
            )

        return self.best_solution