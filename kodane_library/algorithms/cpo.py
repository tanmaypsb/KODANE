import math
import random

from kodane_library.core.optimizer import Optimizer
from kodane_library.core.solution import Solution


class CPO(Optimizer):

    """
    Crested Porcupine Optimizer (CPO).

    Continuous, bounded, single-objective minimization.

    Defense mechanisms:
        1. Sight
        2. Sound
        3. Odor
        4. Physical attack

    Sight and Sound -> exploration
    Odor and Physical Attack -> exploitation

    Also implements Cyclic Population Reduction (CPR).
    """

    def __init__(
        self,
        problem,
        population_size=30,
        max_iterations=100,
        alpha=0.1,
        tf=0.5,
        cycles=2,
        min_population_size=None,
        seed=None
    ):
        if not isinstance(population_size, int):
            raise TypeError(
                "population_size must be an integer."
            )

        if population_size <= 0:
            raise ValueError(
                "population_size must be greater than zero."
            )

        if not isinstance(max_iterations, int):
            raise TypeError(
                "max_iterations must be an integer."
            )

        if max_iterations < 0:
            raise ValueError(
                "max_iterations cannot be negative."
            )

        if not isinstance(alpha, (int, float)):
            raise TypeError(
                "alpha must be numeric."
            )

        if alpha < 0:
            raise ValueError(
                "alpha cannot be negative."
            )

        if not isinstance(tf, (int, float)):
            raise TypeError(
                "tf must be numeric."
            )

        if not 0 < tf < 1:
            raise ValueError(
                "tf must be between 0 and 1."
            )

        if not isinstance(cycles, int):
            raise TypeError(
                "cycles must be an integer."
            )

        if cycles <= 0:
            raise ValueError(
                "cycles must be greater than zero."
            )

        if problem.dimension is None:
            raise ValueError(
                "CPO requires problem.dimension."
            )

        if problem.bounds is None:
            raise ValueError(
                "CPO requires problem.bounds."
            )

        if len(problem.bounds) != problem.dimension:
            raise ValueError(
                "Number of bounds must match problem dimension."
            )

        if min_population_size is None:
            min_population_size = max(
                3,
                population_size // 2
            )

        if not isinstance(min_population_size, int):
            raise TypeError(
                "min_population_size must be an integer."
            )

        if min_population_size < 2:
            raise ValueError(
                "min_population_size must be at least 2."
            )

        if min_population_size > population_size:
            raise ValueError(
                "min_population_size cannot exceed population_size."
            )

        super().__init__(
            problem,
            population_size,
            max_iterations
        )

        self.alpha = alpha
        self.tf = tf
        self.cycles = cycles
        self.min_population_size = min_population_size

        self.seed = seed
        self.rng = random.Random(seed)

        self.population = []
        self.fitness = []

        self.best_solution = None
        self.best_fitness = None

        self.history = []
        self.nfe = 0

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def initialize(self):
        self.rng = random.Random(self.seed)

        self.population = []
        self.fitness = []

        self.best_solution = None
        self.best_fitness = None

        self.history = []
        self.nfe = 0

        for _ in range(self.population_size):

            position = []

            for lower, upper in self.problem.bounds:

                position.append(
                    self.rng.uniform(
                        lower,
                        upper
                    )
                )

            self.population.append(position)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def _evaluate(self, position):

        solution = Solution(position)

        fitness = self.problem.evaluate(solution)

        self.nfe += 1

        return fitness

    # ---------------------------------------------------------
    # Boundary handling
    # ---------------------------------------------------------

    def _clip(self, position):

        result = []

        for value, (lower, upper) in zip(
            position,
            self.problem.bounds
        ):
            result.append(
                max(
                    lower,
                    min(
                        value,
                        upper
                    )
                )
            )

        return result

    # ---------------------------------------------------------
    # Best solution
    # ---------------------------------------------------------

    def _update_best(
        self,
        position,
        fitness
    ):

        if (
            self.best_fitness is None
            or fitness < self.best_fitness
        ):

            self.best_fitness = fitness

            self.best_solution = (
                position.copy()
            )

    # ---------------------------------------------------------
    # Random distinct indices
    # ---------------------------------------------------------

    def _random_indices(
        self,
        current_index,
        count
    ):

        available = [
            i
            for i in range(
                len(self.population)
            )
            if i != current_index
        ]

        if len(available) < count:

            raise RuntimeError(
                "CPO requires at least "
                f"{count + 1} active individuals."
            )

        return self.rng.sample(
            available,
            count
        )

    # ---------------------------------------------------------
    # Defense factor
    # ---------------------------------------------------------

    def _gamma(self, iteration):

        if self.max_iterations == 0:
            return 0.0

        ratio = (
            iteration
            /
            self.max_iterations
        )

        return (
            2.0
            * self.rng.random()
            *
            (
                1.0
                -
                ratio
            )
            ** ratio
        )

    # ---------------------------------------------------------
    # Sight
    # ---------------------------------------------------------

    def _sight(
        self,
        position,
        current_index
    ):
        """
        First defense mechanism.

        x_i(t+1)
        =
        x_i(t)
        +
        tau_1 *
        |2 * tau_2 * x_CP(t) - y_i(t)|

        tau_1 is sampled from a normal distribution.
        tau_2 is sampled uniformly from [0, 1].

        y_i is generated between the current solution
        and a randomly selected solution.
        """

        index = self._random_indices(
            current_index,
            1
        )[0]

        random_solution = (
            self.population[index]
        )

        y = [
            (
                position[j]
                +
                random_solution[j]
            )
            / 2.0
            for j in range(
                self.problem.dimension
            )
        ]

        tau_1 = self.rng.gauss(
            0.0,
            1.0
        )

        tau_2 = self.rng.random()

        new_position = []

        for j in range(
            self.problem.dimension
        ):

            value = (
                position[j]
                +
                tau_1
                *
                abs(
                    2.0
                    * tau_2
                    * self.best_solution[j]
                    -
                    y[j]
                )
            )

            new_position.append(value)

        return new_position

    # ---------------------------------------------------------
    # Sound
    # ---------------------------------------------------------

    def _sound(
        self,
        position,
        current_index
    ):
        """
        Second defense mechanism.

        x_i(t+1)
        =
        (1-U1)x_i(t)
        +
        U1[
            y_i(t)
            +
            tau_3(
                x_r1(t)-x_r2(t)
            )
        ]

        U1 is a random vector in [0,1].
        """

        indices = self._random_indices(
            current_index,
            2
        )

        random_1 = self.population[
            indices[0]
        ]

        random_2 = self.population[
            indices[1]
        ]

        y = [
            (
                position[j]
                +
                self.best_solution[j]
            )
            / 2.0
            for j in range(
                self.problem.dimension
            )
        ]

        new_position = []

        for j in range(
            self.problem.dimension
        ):

            u1 = self.rng.random()

            tau_3 = self.rng.random()

            value = (
                (1.0 - u1)
                * position[j]
                +
                u1
                * (
                    y[j]
                    +
                    tau_3
                    * (
                        random_1[j]
                        -
                        random_2[j]
                    )
                )
            )

            new_position.append(value)

        return new_position

    # ---------------------------------------------------------
    # Odor
    # ---------------------------------------------------------

    def _odor(
        self,
        position,
        current_index,
        iteration
    ):
        """
        Third defense mechanism.

        x_i(t+1)
        =
        (1-U1)x_i(t)
        +
        U1[
            x_r1
            +
            S_i(
                x_r2-x_r3
            )
            -
            tau_3 * delta * gamma_t * S_i
        ]
        """

        indices = self._random_indices(
            current_index,
            3
        )

        random_1 = self.population[
            indices[0]
        ]

        random_2 = self.population[
            indices[1]
        ]

        random_3 = self.population[
            indices[2]
        ]

        epsilon = 1e-12

        total_fitness = (
            sum(self.fitness)
        )

        denominator = (
            total_fitness
            +
            epsilon
        )

        exponent = (
            self.fitness[current_index]
            /
            denominator
        )

        exponent = max(
            -50.0,
            min(
                exponent,
                50.0
            )
        )

        s = math.exp(
            exponent
        )

        gamma = self._gamma(
            iteration
        )

        new_position = []

        for j in range(
            self.problem.dimension
        ):

            u1 = self.rng.random()

            tau_3 = self.rng.random()

            delta = (
                1.0
                if self.rng.random() <= 0.5
                else -1.0
            )

            value = (
                (1.0 - u1)
                * position[j]
                +
                u1
                * (
                    random_1[j]
                    +
                    s
                    * (
                        random_2[j]
                        -
                        random_3[j]
                    )
                    -
                    tau_3
                    * delta
                    * gamma
                    * s
                )
            )

            new_position.append(value)

        return new_position

    # ---------------------------------------------------------
    # Physical attack
    # ---------------------------------------------------------

    def _physical_attack(
        self,
        position,
        current_index,
        iteration
    ):
        """
        Fourth defense mechanism.

        x_i(t+1)
        =
        x_CP
        +
        [alpha(1-tau_4)+tau_4]
        [delta*x_CP-x_i]
        -
        tau_5*delta*gamma_t*F_i

        F_i:
        tau_6 * m_i *
        (v_i(t+1)-v_i(t)) / Delta_t
        """

        random_index = self._random_indices(
            current_index,
            1
        )[0]

        random_solution = (
            self.population[
                random_index
            ]
        )

        epsilon = 1e-12

        total_fitness = (
            sum(self.fitness)
        )

        # Original CPO mass formulation.
        exponent = (
            total_fitness
            +
            epsilon
        )

        if exponent > 700:
            mass = 0.0
        else:
            mass = (
                self.fitness[current_index]
                /
                (
                    math.exp(exponent)
                    +
                    epsilon
                )
            )

        gamma = self._gamma(
            iteration
        )

        delta = (
            1.0
            if self.rng.random() <= 0.5
            else -1.0
        )

        tau_4 = self.rng.random()
        tau_5 = self.rng.random()

        delta_t = max(
            1.0,
            float(iteration)
        )

        new_position = []

        for j in range(
            self.problem.dimension
        ):

            tau_6 = self.rng.random()

            velocity_difference = (
                random_solution[j]
                -
                position[j]
            )

            force = (
                tau_6
                * mass
                * velocity_difference
                /
                delta_t
            )

            coefficient = (
                self.alpha
                * (
                    1.0
                    -
                    tau_4
                )
                +
                tau_4
            )

            value = (
                self.best_solution[j]
                +
                coefficient
                * (
                    delta
                    * self.best_solution[j]
                    -
                    position[j]
                )
                -
                tau_5
                * delta
                * gamma
                * force
            )

            new_position.append(
                value
            )

        return new_position

    # ---------------------------------------------------------
    # Strategy selection
    # ---------------------------------------------------------

    def _generate_candidate(
        self,
        position,
        current_index,
        iteration
    ):

        tau_8 = self.rng.random()
        tau_9 = self.rng.random()

        # Exploration phase
        if tau_8 < tau_9:

            tau_6 = self.rng.random()
            tau_7 = self.rng.random()

            if tau_6 < tau_7:

                candidate = self._sight(
                    position,
                    current_index
                )

            else:

                candidate = self._sound(
                    position,
                    current_index
                )

        # Exploitation phase
        else:

            tau_10 = self.rng.random()

            if tau_10 < self.tf:

                candidate = self._odor(
                    position,
                    current_index,
                    iteration
                )

            else:

                candidate = self._physical_attack(
                    position,
                    current_index,
                    iteration
                )

        return self._clip(
            candidate
        )

    # ---------------------------------------------------------
    # Cyclic Population Reduction
    # ---------------------------------------------------------

    def _target_population_size(
        self,
        iteration
    ):
        """
        CPO cyclic population reduction:

        N =
        N_min
        +
        (N_max - N_min)
        *
        (
            1 -
            (
                (t % cycle_length)
                / cycle_length
            )
        )

        The population therefore shrinks during a cycle
        and is restored when the next cycle begins.
        """

        if self.max_iterations == 0:
            return self.population_size

        cycle_length = (
            self.max_iterations
            /
            self.cycles
        )

        if cycle_length <= 1:
            return self.min_population_size

        remainder = (
            iteration
            %
            cycle_length
        )

        ratio = (
            remainder
            /
            cycle_length
        )

        size = (
            self.min_population_size
            +
            (
                self.population_size
                -
                self.min_population_size
            )
            *
            (
                1.0
                -
                ratio
            )
        )

        size = int(
            round(size)
        )

        return max(
            self.min_population_size,
            min(
                self.population_size,
                size
            )
        )

    def _reduce_population(
        self,
        target_size
    ):

        if len(self.population) <= target_size:
            return

        ranked = sorted(
            zip(
                self.population,
                self.fitness
            ),
            key=lambda item: item[1]
        )

        ranked = ranked[
            :target_size
        ]

        self.population = [
            position.copy()
            for position, _ in ranked
        ]

        self.fitness = [
            fitness
            for _, fitness in ranked
        ]

    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------

    def optimize(self):

        self.initialize()

        # Initial population
        for position in self.population:

            fitness = self._evaluate(
                position
            )

            self.fitness.append(
                fitness
            )

            self._update_best(
                position,
                fitness
            )

        self.history = [
            self.best_fitness
        ]

        # Main loop
        for iteration in range(
            self.max_iterations
        ):

            new_population = []
            new_fitness = []

            for index, position in enumerate(
                self.population
            ):

                candidate = (
                    self._generate_candidate(
                        position,
                        index,
                        iteration
                    )
                )

                candidate_fitness = (
                    self._evaluate(
                        candidate
                    )
                )

                current_fitness = (
                    self.fitness[index]
                )

                # Greedy selection
                if (
                    candidate_fitness
                    <=
                    current_fitness
                ):

                    selected_position = (
                        candidate
                    )

                    selected_fitness = (
                        candidate_fitness
                    )

                else:

                    selected_position = (
                        position.copy()
                    )

                    selected_fitness = (
                        current_fitness
                    )

                new_population.append(
                    selected_position
                )

                new_fitness.append(
                    selected_fitness
                )

                self._update_best(
                    selected_position,
                    selected_fitness
                )

            self.population = (
                new_population
            )

            self.fitness = (
                new_fitness
            )

            # Apply CPR
            target_size = (
                self._target_population_size(
                    iteration + 1
                )
            )

            self._reduce_population(
                target_size
            )

            self.history.append(
                self.best_fitness
            )

        result = Solution(
            self.best_solution.copy()
        )

        result.set_fitness(
            self.best_fitness
        )

        return result