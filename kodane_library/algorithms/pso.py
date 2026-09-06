import random

from kodane_library.core.optimizer import Optimizer
from kodane_library.core.solution import Solution


class PSO(Optimizer):

    """

    Particle Swarm Optimization for continuous optimization problems.

    PSO searches for the minimum of an objective function using a swarm
    of particles. Each particle has a position, velocity, and personal best,
    while the swarm maintains a global best solution.

    """

    def __init__(
        self,
        problem,
        population_size=30,
        max_iterations=100,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed = None
    ):

        """
        Initialize the PSO optimizer.

        Args:
            problem: Problem instance defining the objective and bounds.
            population_size: Number of particles in the swarm.
            max_iterations: Number of optimization iterations.
            inertia: Weight controlling the particle's previous velocity.
            cognitive: Weight controlling attraction toward the particle's
                personal best.
            social: Weight controlling attraction toward the swarm's global best.
            seed: Optional random seed for reproducible optimization.

        """

        super().__init__(
            problem,
            population_size,
            max_iterations
        )

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        if not isinstance(population_size, int):
            raise TypeError("population_size must be an integer")
        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if not isinstance(inertia, (int, float)):
            raise TypeError("inertia must be a number")
        if not isinstance(cognitive, (int, float)):
            raise TypeError("cognitive must be a number")
        if not isinstance(social, (int, float)):
            raise TypeError("social must be a number")

        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if inertia < 0:
            raise ValueError("inertia must be non-negative")
        if cognitive < 0:
            raise ValueError("cognitive must be non-negative")
        if social < 0:
            raise ValueError("social must be non-negative")
        if problem.bounds is None: 
            raise ValueError(
                "PSO requires problem bounds."
                )    

        self.seed = seed        
        self.rng = random.Random(seed)    

        # Limit velocity to 10% of each dimension's search range.  
        self.velocity_limits = [
            0.1 * (upper - lower)
            for lower, upper in problem.bounds

        ]

        self.particles = []
        self.history = []       # history is also part of the PSO object's state. If we put it inside the iteration loop, we'd recreate the list every iteration and lose everything recorded before it

    def initialize(self):

        """

        Reset the optimizer state and create the initial particle swarm.

        """

        self.rng = random.Random(self.seed)

        self.particles = []
        self.history = []
        self.best_solution = None
        self.best_fitness = None 

        for _ in range(self.population_size):
            position = [
                self.rng.uniform(lower, upper)
                for lower, upper in self.problem.bounds
                ]
            velocity = [
                self.rng.uniform(-limit, limit)
                for limit in self.velocity_limits
            ]

            particle = {
                "position": position,
                "velocity": velocity,
                "pbest": position.copy(),
                "pbest_fitness": None
            }

            self.particles.append(particle)

    def optimize(self):

        """
        Run PSO and return the best solution found.

        """

        self.initialize()

        for particle in self.particles:
            fitness = self.problem.evaluate(
                Solution(particle["position"])
            )

            particle["pbest_fitness"] = fitness

            if (
                self.best_fitness is None
                or fitness < self.best_fitness
            ):
                self.best_fitness = fitness
                self.best_solution = particle["position"].copy()
        self.history = [self.best_fitness]  # including the initials too as it helps

        for _ in range(self.max_iterations):

            for particle in self.particles:

                for i in range(self.problem.dimension):

                    r1 = self.rng.random()
                    r2 = self.rng.random()

                    particle["velocity"][i] = (
                        self.inertia * particle["velocity"][i]
                        + self.cognitive * r1
                        * (
                            particle["pbest"][i]
                            - particle["position"][i]
                        )
                        + self.social * r2
                        * (
                            self.best_solution[i]
                            - particle["position"][i]
                        )
                    )

                    vmax = self.velocity_limits[i]
                    if particle["velocity"][i] < -vmax:
                        particle["velocity"][i] = -vmax
                    if particle["velocity"][i] > vmax:
                        particle["velocity"][i] = vmax

                    particle["position"][i] += (
                        particle["velocity"][i]
                    )

                    lower, upper = self.problem.bounds[i]
                    if particle["position"][i] < lower:
                        particle["position"][i] = lower
                    if particle["position"][i] > upper:
                        particle["position"][i] = upper

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

        best_solution = Solution(self.best_solution)
        best_solution.set_fitness(self.best_fitness)

        return best_solution