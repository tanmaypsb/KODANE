import random

from kodane_library.core.optimizer import Optimizer
from kodane_library.core.solution import Solution


class DiscretePSO(Optimizer):

    """
    Discrete Particle Swarm Optimization for assignment problems.

    Each particle represents a discrete assignment, where each position
    contains a valid discrete value such as a robot ID.

    """

    def __init__(
        self,
        problem,
        population_size=30,
        max_iterations=100,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=None
    ):
        """
        Initialize the Discrete PSO optimizer.

        Parameters:
            problem: Optimization problem to solve.
            population_size: Number of particles in the swarm.
            max_iterations: Number of optimization iterations.
            inertia: Weight for maintaining the current assignment.
            cognitive: Weight for following the particle's personal best.
            social: Weight for following the swarm's global best.
            seed: Optional random seed for reproducibility.
        """

        if not isinstance(population_size, int):
            raise TypeError("population_size must be an integer.")

        if population_size <= 0:
            raise ValueError("population_size must be greater than zero.")

        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer.")

        if max_iterations < 0:
            raise ValueError("max_iterations cannot be negative.")

        if not isinstance(inertia, (int, float)):
            raise TypeError("inertia must be numeric.")

        if not isinstance(cognitive, (int, float)):
            raise TypeError("cognitive must be numeric.")

        if not isinstance(social, (int, float)):
            raise TypeError("social must be numeric.")

        if inertia < 0:
            raise ValueError("inertia cannot be negative.")

        if cognitive < 0:
            raise ValueError("cognitive cannot be negative.")

        if social < 0:
            raise ValueError("social cannot be negative.")

        if inertia + cognitive + social == 0:
            raise ValueError(
                "Sum of weights must be greater than zero."
                )

        super().__init__(
            problem,
            population_size,
            max_iterations
        )

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        self.seed = seed
        self.rng = random.Random(seed)

        self.particles = []
        self.history = []


 
    def initialize(self):

        
        # Initialize the particle population and reset optimizer state.
        # Each particle receives a random valid assignment for every task.


        self.rng = random.Random(self.seed)
        self.particles = []
        self.history = []
        self.best_solution = None
        self.best_fitness = None
        number_of_tasks = len(self.problem.tasks)
        number_of_robots = len(self.problem.robots)

        for _ in range(self.population_size):
            position = [
                self.rng.randint(1, number_of_robots)
                for _ in range(number_of_tasks)
                ]

            particle = {
                "position": position,
                "pbest": position.copy(),
                "pbest_fitness": None

                }

            self.particles.append(particle)

    def _update_position(self, particle):

        """
        Update a particle's position using discrete PSO influences.

        Each task assignment is selected using weighted influence from
        the current position, personal best, and global best.
        """ 

        current = particle["position"]
        pbest = particle["pbest"]
        gbest = self.best_solution

        new_position = []

        for i in range(len(current)):

            current_value = current[i]
            pbest_value = pbest[i]
            gbest_value = gbest[i]

            if (
                current_value == pbest_value
                and current_value == gbest_value
            ):
                new_position.append(current_value)
                continue

            choices = [
                current_value,
                pbest_value,
                gbest_value
            ]

            weights = [
                self.inertia,
                self.cognitive,
                self.social
            ]

            new_value = self.rng.choices(
                choices,
                weights=weights,
                k=1
            )[0]

            new_position.append(new_value)

        particle["position"] = new_position

    def optimize(self):

        """
        Run the Discrete PSO optimization process.

        Initializes the swarm, evaluates particles, updates personal
        and global best solutions, and records convergence history.

        Returns:
            Solution: The best solution found during optimization.
        """

        self.initialize()

        for particle in self.particles:

            solution = Solution(
                particle["position"]
            )

            fitness = self.problem.evaluate(solution)

            particle["pbest_fitness"] = fitness

            if (
                self.best_fitness is None
                or fitness < self.best_fitness
            ):
                self.best_fitness = fitness
                self.best_solution = particle["position"].copy()

        self.history = [self.best_fitness]

        for _ in range(self.max_iterations):

            for particle in self.particles:

                self._update_position(particle)

                solution = Solution(
                    particle["position"]
                )

                fitness = self.problem.evaluate(solution)

                if fitness < particle["pbest_fitness"]:
                    particle["pbest"] = particle["position"].copy()
                    particle["pbest_fitness"] = fitness

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = particle["position"].copy()

            self.history.append(self.best_fitness)

        best_solution = Solution(
            self.best_solution
        )

        best_solution.set_fitness(
            self.best_fitness
        )

        return best_solution