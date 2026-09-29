import math
import random

from kodane_library.core.optimizer import Optimizer
from kodane_library.core.solution import Solution


class RVEA(Optimizer):
    """
    Reference Vector Guided Evolutionary Algorithm (RVEA).

    Multi-objective / many-objective evolutionary optimizer.

    The problem objective must return a sequence of objective values:

        def objective(x, problem):
            return [
                objective_1(x),
                objective_2(x),
                ...
            ]

    All objectives are minimized.
    """

    def __init__(
        self,
        problem,
        population_size=30,
        max_iterations=100,
        alpha=2.0,
        adaptation_frequency=0.1,
        crossover_probability=1.0,
        mutation_probability=None,
        eta_c=20.0,
        eta_m=20.0,
        seed=None
    ):
        super().__init__(
            problem,
            population_size=population_size,
            max_iterations=max_iterations
        )

        if population_size < 2:
            raise ValueError(
                "population_size must be at least 2."
            )

        if max_iterations < 0:
            raise ValueError(
                "max_iterations must be non-negative."
            )

        if not 0.0 < adaptation_frequency <= 1.0:
            raise ValueError(
                "adaptation_frequency must be in (0, 1]."
            )

        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError(
                "crossover_probability must be in [0, 1]."
            )

        if eta_c <= 0.0:
            raise ValueError(
                "eta_c must be positive."
            )

        if eta_m <= 0.0:
            raise ValueError(
                "eta_m must be positive."
            )

        if alpha <= 0.0:
            raise ValueError(
                "alpha must be positive."
            )

        self.alpha = alpha
        self.adaptation_frequency = adaptation_frequency
        self.crossover_probability = crossover_probability
        self.eta_c = eta_c
        self.eta_m = eta_m

        self.rng = random.Random(seed)
        self.seed = seed

        dimension = self._get_dimension()

        if mutation_probability is None:
            self.mutation_probability = (
                1.0 / dimension
            )
        else:
            if not 0.0 <= mutation_probability <= 1.0:
                raise ValueError(
                    "mutation_probability must be in [0, 1]."
                )

            self.mutation_probability = (
                mutation_probability
            )

        self.objective_count = None

        self.reference_vectors = None
        self.base_reference_vectors = None
        self.gamma = None

        self.population = []
        self.pareto_front = []

        self.history = []
        self.objective_history = []

        self.nfe = 0

        self.ideal_point = None
        self.nadir_point = None

    # ========================================================
    # BASIC PROBLEM INFORMATION
    # ========================================================

    def _get_dimension(self):
        dimension = getattr(
            self.problem,
            "dimension",
            None
        )

        if dimension is None:
            raise ValueError(
                "RVEA requires a continuous problem dimension."
            )

        if dimension <= 0:
            raise ValueError(
                "Problem dimension must be positive."
            )

        return dimension

    def _get_bounds(self):
        bounds = getattr(
            self.problem,
            "bounds",
            None
        )

        if bounds is None:
            raise ValueError(
                "RVEA requires bounds."
            )

        dimension = self._get_dimension()

        if len(bounds) != dimension:
            raise ValueError(
                "Number of bounds must match the problem dimension."
            )

        return bounds

    # ========================================================
    # OBJECTIVE EVALUATION
    # ========================================================

    def _evaluate_solution(self, variables):

        objective = getattr(
            self.problem,
            "objective",
            None
        )

        if objective is None:
            raise ValueError(
                "Problem must define an objective."
            )

        values = objective(
            list(variables),
            self.problem
        )

        if not hasattr(values, "__iter__"):
            raise ValueError(
                "RVEA objective must return multiple "
                "objective values."
            )

        objectives = list(values)

        if len(objectives) == 0:
            raise ValueError(
                "Objective function returned no objective values."
            )

        for value in objectives:

            if not isinstance(value, (int, float)):
                raise ValueError(
                    "All objective values must be numeric."
                )

            if not math.isfinite(value):
                raise ValueError(
                    "Objective values must be finite."
                )

        if self.objective_count is None:

            self.objective_count = len(
                objectives
            )

        elif len(objectives) != self.objective_count:

            raise ValueError(
                "All solutions must have the same number "
                "of objectives."
            )

        constraints = getattr(
            self.problem,
            "constraints",
            []
        )

        penalty = 0.0

        for constraint in constraints:

            violation = constraint(
                list(variables),
                self.problem
            )

            if violation > 0:
                penalty += violation

        if penalty > 0:

            objectives = [
                value + penalty
                for value in objectives
            ]

        self.nfe += 1

        return objectives

    def _create_solution(self, variables):

        solution = Solution(
            list(variables)
        )

        objectives = self._evaluate_solution(
            variables
        )

        solution.objectives = objectives
        solution.fitness = objectives

        return solution

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_population(self):

        bounds = self._get_bounds()

        population = []

        for _ in range(
            self.population_size
        ):

            variables = []

            for lower, upper in bounds:

                variables.append(
                    self.rng.uniform(
                        lower,
                        upper
                    )
                )

            population.append(
                self._create_solution(
                    variables
                )
            )

        return population

    # ========================================================
    # REFERENCE VECTOR GENERATION
    # ========================================================

    def _composition_count(
        self,
        objectives,
        divisions
    ):

        return math.comb(
            divisions + objectives - 1,
            objectives - 1
        )

    def _generate_simplex_vectors(
        self,
        objectives,
        divisions
    ):

        vectors = []

        def generate(
            remaining,
            slots,
            prefix
        ):

            if slots == 1:

                vectors.append(
                    prefix
                    + [remaining / divisions]
                )

                return

            for value in range(
                remaining + 1
            ):

                generate(
                    remaining - value,
                    slots - 1,
                    prefix
                    + [value / divisions]
                )

        generate(
            divisions,
            objectives,
            []
        )

        return vectors

    def _generate_reference_vectors(self):

        m = self.objective_count
        target = self.population_size

        divisions = 1

        while (
            self._composition_count(
                m,
                divisions
            ) < target
        ):
            divisions += 1

        vectors = self._generate_simplex_vectors(
            m,
            divisions
        )

        if len(vectors) > target:

            vectors = self._select_spread_vectors(
                vectors,
                target
            )

        return [
            self._normalize_vector(vector)
            for vector in vectors
        ]

    def _select_spread_vectors(
        self,
        vectors,
        target
    ):
        """
        Select a well-spread subset of reference vectors.

        Uses farthest-point sampling and does not assume
        that the number of vectors is larger than the
        number of objectives.
        """

        if target >= len(vectors):
            return vectors

        if target <= 0:
            return []

        selected = []

        # Start with the vector closest to the first
        # objective axis.
        first_index = max(
            range(len(vectors)),
            key=lambda i: vectors[i][0]
        )

        selected.append(first_index)

        while len(selected) < target:

            best_index = None
            best_distance = -1.0

            for i, vector in enumerate(vectors):

                if i in selected:
                    continue

                minimum_distance = float("inf")

                for selected_index in selected:

                    distance = self._euclidean_distance(
                        vector,
                        vectors[selected_index]
                    )

                    minimum_distance = min(
                        minimum_distance,
                        distance
                    )

                if minimum_distance > best_distance:

                    best_distance = minimum_distance
                    best_index = i

            if best_index is None:
                break

            selected.append(best_index)

        return [
            vectors[index]
            for index in selected
        ]

    def _normalize_vector(self, vector):

        norm = math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

        if norm <= 1e-15:

            return [
                1.0 / len(vector)
                for _ in vector
            ]

        return [
            value / norm
            for value in vector
        ]

    # ========================================================
    # VECTOR UTILITIES
    # ========================================================

    def _dot(self, a, b):

        return sum(
            x * y
            for x, y in zip(a, b)
        )

    def _norm(self, vector):

        return math.sqrt(
            sum(
                value * value
                for value in vector
            )
        )

    def _euclidean_distance(self, a, b):

        return math.sqrt(
            sum(
                (x - y) ** 2
                for x, y in zip(a, b)
            )
        )

    def _angle(self, a, b):

        norm_a = self._norm(a)
        norm_b = self._norm(b)

        if (
            norm_a <= 1e-15
            or norm_b <= 1e-15
        ):
            return math.pi / 2.0

        cosine = (
            self._dot(a, b)
            /
            (norm_a * norm_b)
        )

        cosine = max(
            -1.0,
            min(1.0, cosine)
        )

        return math.acos(cosine)

    # ========================================================
    # GAMMA VALUES
    # ========================================================

    def _calculate_gamma(self):

        vectors = self.reference_vectors

        gamma = []

        for i, vector in enumerate(vectors):

            smallest_angle = math.pi

            for j, other in enumerate(vectors):

                if i == j:
                    continue

                angle = self._angle(
                    vector,
                    other
                )

                if angle < smallest_angle:
                    smallest_angle = angle

            if smallest_angle <= 1e-15:
                smallest_angle = 1e-15

            gamma.append(
                smallest_angle
            )

        return gamma

    # ========================================================
    # SBX CROSSOVER
    # ========================================================

    def _sbx_crossover(
        self,
        parent1,
        parent2
    ):

        bounds = self._get_bounds()

        child1 = list(parent1)
        child2 = list(parent2)

        if (
            self.rng.random()
            >
            self.crossover_probability
        ):
            return child1, child2

        for i, (lower, upper) in enumerate(bounds):

            x1 = parent1[i]
            x2 = parent2[i]

            if abs(x1 - x2) <= 1e-14:
                continue

            if self.rng.random() > 0.5:
                continue

            if x1 > x2:
                x1, x2 = x2, x1

            rand = self.rng.random()

            beta = (
                1.0
                + 2.0
                * (x1 - lower)
                / (x2 - x1)
            )

            alpha = (
                2.0
                - beta ** (
                    -(self.eta_c + 1.0)
                )
            )

            if rand <= 1.0 / alpha:

                beta_q = (
                    rand * alpha
                ) ** (
                    1.0
                    /
                    (self.eta_c + 1.0)
                )

            else:

                beta_q = (
                    1.0
                    /
                    (
                        2.0
                        - rand * alpha
                    )
                ) ** (
                    1.0
                    /
                    (self.eta_c + 1.0)
                )

            child_a = (
                0.5
                * (
                    x1
                    + x2
                    - beta_q
                    * (x2 - x1)
                )
            )

            beta = (
                1.0
                + 2.0
                * (upper - x2)
                / (x2 - x1)
            )

            alpha = (
                2.0
                - beta ** (
                    -(self.eta_c + 1.0)
                )
            )

            if rand <= 1.0 / alpha:

                beta_q = (
                    rand * alpha
                ) ** (
                    1.0
                    /
                    (self.eta_c + 1.0)
                )

            else:

                beta_q = (
                    1.0
                    /
                    (
                        2.0
                        - rand * alpha
                    )
                ) ** (
                    1.0
                    /
                    (self.eta_c + 1.0)
                )

            child_b = (
                0.5
                * (
                    x1
                    + x2
                    + beta_q
                    * (x2 - x1)
                )
            )

            child_a = max(
                lower,
                min(upper, child_a)
            )

            child_b = max(
                lower,
                min(upper, child_b)
            )

            if self.rng.random() <= 0.5:

                child1[i] = child_b
                child2[i] = child_a

            else:

                child1[i] = child_a
                child2[i] = child_b

        return child1, child2

    # ========================================================
    # POLYNOMIAL MUTATION
    # ========================================================

    def _polynomial_mutation(
        self,
        variables
    ):

        bounds = self._get_bounds()

        mutated = list(variables)

        for i, (lower, upper) in enumerate(bounds):

            if (
                self.rng.random()
                >
                self.mutation_probability
            ):
                continue

            if upper - lower <= 1e-15:
                continue

            x = mutated[i]

            delta_1 = (
                x - lower
            ) / (
                upper - lower
            )

            delta_2 = (
                upper - x
            ) / (
                upper - lower
            )

            rand = self.rng.random()

            mut_pow = 1.0 / (
                self.eta_m + 1.0
            )

            if rand <= 0.5:

                xy = 1.0 - delta_1

                value = (
                    2.0 * rand
                    + (
                        1.0
                        - 2.0 * rand
                    )
                    * (
                        xy
                        ** (
                            self.eta_m + 1.0
                        )
                    )
                )

                delta_q = (
                    value ** mut_pow
                    - 1.0
                )

            else:

                xy = 1.0 - delta_2

                value = (
                    2.0 * (1.0 - rand)
                    + 2.0 * (
                        rand - 0.5
                    )
                    * (
                        xy
                        ** (
                            self.eta_m + 1.0
                        )
                    )
                )

                delta_q = (
                    1.0
                    - value ** mut_pow
                )

            x = x + (
                delta_q
                * (upper - lower)
            )

            mutated[i] = max(
                lower,
                min(upper, x)
            )

        return mutated

    # ========================================================
    # DOMINANCE
    # ========================================================

    def _dominates(
        self,
        first,
        second
    ):

        first_objectives = first.objectives
        second_objectives = second.objectives

        no_worse = all(
            a <= b
            for a, b in zip(
                first_objectives,
                second_objectives
            )
        )

        strictly_better = any(
            a < b
            for a, b in zip(
                first_objectives,
                second_objectives
            )
        )

        return (
            no_worse
            and strictly_better
        )

    # ========================================================
    # TOURNAMENT
    # ========================================================

    def _tournament(self):

        first = self.rng.choice(
            self.population
        )

        second = self.rng.choice(
            self.population
        )

        if self._dominates(
            first,
            second
        ):
            return first

        if self._dominates(
            second,
            first
        ):
            return second

        first_sum = sum(
            first.objectives
        )

        second_sum = sum(
            second.objectives
        )

        if first_sum <= second_sum:
            return first

        return second

    # ========================================================
    # OFFSPRING
    # ========================================================

    def _generate_offspring(self):

        offspring = []

        while len(offspring) < self.population_size:

            parent1 = self._tournament()
            parent2 = self._tournament()

            child1, child2 = self._sbx_crossover(
                parent1.variables,
                parent2.variables
            )

            child1 = self._polynomial_mutation(
                child1
            )

            child2 = self._polynomial_mutation(
                child2
            )

            offspring.append(
                self._create_solution(
                    child1
                )
            )

            if len(offspring) < self.population_size:

                offspring.append(
                    self._create_solution(
                        child2
                    )
                )

        return offspring

    # ========================================================
    # OBJECTIVE TRANSLATION
    # ========================================================

    def _translate_objectives(
        self,
        population
    ):

        ideal = [
            min(
                solution.objectives[j]
                for solution in population
            )
            for j in range(
                self.objective_count
            )
        ]

        translated = []

        for solution in population:

            translated.append([
                solution.objectives[j]
                - ideal[j]
                for j in range(
                    self.objective_count
                )
            ])

        return translated, ideal

    # ========================================================
    # REFERENCE VECTOR ASSOCIATION
    # ========================================================

    def _associate(
        self,
        translated
    ):

        assignments = []
        angles = []

        for objective_vector in translated:

            vector_angles = [
                self._angle(
                    objective_vector,
                    reference
                )
                for reference in self.reference_vectors
            ]

            best_index = min(
                range(
                    len(vector_angles)
                ),
                key=lambda i: vector_angles[i]
            )

            assignments.append(
                best_index
            )

            angles.append(
                vector_angles[best_index]
            )

        return assignments, angles

    # ========================================================
    # APD SELECTION
    # ========================================================

    def _apd_selection(
        self,
        population,
        generation
    ):

        translated, ideal = (
            self._translate_objectives(
                population
            )
        )

        assignments, angles = (
            self._associate(
                translated
            )
        )

        selected = []

        if self.max_iterations > 0:

            progress = (
                generation
                /
                self.max_iterations
            )

        else:

            progress = 0.0

        progress = max(
            0.0,
            min(1.0, progress)
        )

        penalty_factor = (
            self.objective_count
            * (
                progress
                ** self.alpha
            )
        )

        niches = [
            []
            for _ in self.reference_vectors
        ]

        for index, niche in enumerate(
            assignments
        ):

            niches[niche].append(
                index
            )

        for niche_index, candidates in enumerate(
            niches
        ):

            if not candidates:
                continue

            reference = self.reference_vectors[
                niche_index
            ]

            gamma = self.gamma[
                niche_index
            ]

            best_index = None
            best_apd = float("inf")

            for candidate_index in candidates:

                objective_vector = translated[
                    candidate_index
                ]

                distance = self._norm(
                    objective_vector
                )

                theta = self._angle(
                    objective_vector,
                    reference
                )

                penalty = (
                    penalty_factor
                    * theta
                    / gamma
                )

                apd = (
                    distance
                    * (1.0 + penalty)
                )

                if apd < best_apd:

                    best_apd = apd
                    best_index = candidate_index

            if best_index is not None:

                selected.append(
                    population[best_index]
                )

        # Fill any remaining slots.
        if len(selected) < self.population_size:

            selected_ids = {
                id(solution)
                for solution in selected
            }

            remaining = [
                solution
                for solution in population
                if id(solution)
                not in selected_ids
            ]

            remaining = (
                self._sort_by_scalar_quality(
                    remaining
                )
            )

            for solution in remaining:

                if len(selected) >= self.population_size:
                    break

                selected.append(
                    solution
                )

        selected = selected[
            :self.population_size
        ]

        return selected, ideal

    # ========================================================
    # FALLBACK QUALITY
    # ========================================================

    def _sort_by_scalar_quality(
        self,
        population
    ):

        return sorted(
            population,
            key=lambda solution: sum(
                solution.objectives
            )
        )

    # ========================================================
    # REFERENCE VECTOR ADAPTATION
    # ========================================================

    def _adapt_reference_vectors(
        self,
        population
    ):

        if not population:
            return

        minimum = [
            min(
                solution.objectives[j]
                for solution in population
            )
            for j in range(
                self.objective_count
            )
        ]

        maximum = [
            max(
                solution.objectives[j]
                for solution in population
            )
            for j in range(
                self.objective_count
            )
        ]

        ranges = [
            maximum[j] - minimum[j]
            for j in range(
                self.objective_count
            )
        ]

        ranges = [
            value
            if value > 1e-15
            else 1.0
            for value in ranges
        ]

        adapted = []

        for base_vector in (
            self.base_reference_vectors
        ):

            vector = [
                base_vector[j] * ranges[j]
                for j in range(
                    self.objective_count
                )
            ]

            adapted.append(
                self._normalize_vector(
                    vector
                )
            )

        self.reference_vectors = adapted

        self.gamma = (
            self._calculate_gamma()
        )

    # ========================================================
    # PARETO FRONT
    # ========================================================

    def _get_pareto_front(
        self,
        population
    ):

        pareto = []

        for candidate in population:

            dominated = False

            for other in population:

                if candidate is other:
                    continue

                if self._dominates(
                    other,
                    candidate
                ):

                    dominated = True
                    break

            if not dominated:

                pareto.append(
                    candidate
                )

        return pareto

    # ========================================================
    # HISTORY
    # ========================================================

    def _record_history(
        self,
        population
    ):

        pareto = self._get_pareto_front(
            population
        )

        self.pareto_front = pareto

        self.objective_history.append([
            list(solution.objectives)
            for solution in pareto
        ])

        self.history.append(
            len(pareto)
        )

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    def optimize(self):

        self.nfe = 0
        self.history = []
        self.objective_history = []

        # ----------------------------------------------------
        # Initial population
        # ----------------------------------------------------

        self.population = (
            self._initialize_population()
        )

        if self.objective_count is None:

            raise ValueError(
                "Could not determine number of objectives."
            )

        if self.objective_count < 2:

            raise ValueError(
                "RVEA requires at least two objectives."
            )

        # ----------------------------------------------------
        # Reference vectors
        # ----------------------------------------------------

        self.base_reference_vectors = (
            self._generate_reference_vectors()
        )

        self.reference_vectors = [
            list(vector)
            for vector in (
                self.base_reference_vectors
            )
        ]

        self.gamma = (
            self._calculate_gamma()
        )

        # ----------------------------------------------------
        # Initial history
        # ----------------------------------------------------

        self._record_history(
            self.population
        )

        # ----------------------------------------------------
        # Evolution
        # ----------------------------------------------------

        for generation in range(
            1,
            self.max_iterations + 1
        ):

            offspring = (
                self._generate_offspring()
            )

            combined = (
                self.population
                +
                offspring
            )

            selected, ideal = (
                self._apd_selection(
                    combined,
                    generation
                )
            )

            self.population = selected
            self.ideal_point = ideal

            adaptation_interval = max(
                1,
                int(
                    math.ceil(
                        self.max_iterations
                        * self.adaptation_frequency
                    )
                )
            )

            if (
                generation
                % adaptation_interval
                == 0
            ):

                self._adapt_reference_vectors(
                    self.population
                )

            self._record_history(
                self.population
            )

        # ----------------------------------------------------
        # Final Pareto front
        # ----------------------------------------------------

        self.pareto_front = (
            self._get_pareto_front(
                self.population
            )
        )

        if self.pareto_front:

            self.best_solution = (
                self.pareto_front[0]
            )

            self.best_fitness = (
                self.best_solution.objectives
            )

        else:

            self.best_solution = None
            self.best_fitness = None

        return self.pareto_front