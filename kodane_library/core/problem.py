import math

class Problem:

    def __init__(
        self,
        robots=None,
        tasks=None,
        objective=None,
        constraints=None,
        dimension=None,
        bounds = None
    ):
        self.robots = robots
        self.tasks = tasks
        self.objective = objective
        self.constraints = constraints or []
        self.dimension = dimension
        self.bounds = bounds

        if self.bounds is not None:

            if self.dimension is None:
                raise ValueError(
                    "dimension must be specified when bounds are provided."
                    )
            if len(self.bounds) != self.dimension:
                raise ValueError(
                    "Number of bounds must match the problem dimension."
                    )

            for bound in self.bounds:
                
                if len(bound) != 2:
                    raise ValueError(
                        "Each bound must contain exactly two values: "
                        "(lower, upper)."
                        )
                        
                lower, upper = bound

                if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)) or not math.isfinite(lower) or not math.isfinite(upper):
                    raise ValueError(
                        "Bounds must contain finite numeric values."
                    )
                        
                if lower > upper:
                    raise ValueError(
                        "Lower bound cannot be greater than upper bound."
                    )


    def evaluate(self, solution):
        self._validate_solution(solution.variables)

        fitness = self.objective(solution.variables, self)

        for constraint in self.constraints:
            violation = constraint(solution.variables, self)

            if violation > 0:
                fitness += violation

        solution.set_fitness(fitness)

        return fitness

    def _validate_solution(self, solution):
        if self.tasks is not None:
            if len(solution) != len(self.tasks):
                raise ValueError(
                    "Solution length must match the number of tasks."
                )

        elif self.dimension is not None:
            if len(solution) != self.dimension:
                raise ValueError(
                    "Solution length must match the problem dimension."
                )

        if self.robots is not None:
            number_of_robots = len(self.robots)

            for robot_id in solution:
                if robot_id < 1 or robot_id > number_of_robots:
                    raise ValueError(
                        f"Invalid robot ID: {robot_id}"
                    )





