from kodane_library.core.robot import Robot
from kodane_library.core.task import Task
from kodane_library.core.problem import Problem
from kodane_library.core.solution import Solution
from kodane_library.objectives.distance import total_distance


robots = [
    Robot(1, (0, 0), 100, 5),
    Robot(2, (10, 10), 80, 5),
    Robot(3, (20, 20), 70, 4),
    Robot(4, (5, 15), 90, 6),
    Robot(5, (15, 5), 60, 4)
]

tasks = [
    Task(1, (13, 14), 10, 3),
    Task(2, (2, 4), 15, 2),
    Task(3, (18, 8), 20, 5),
    Task(4, (7, 18), 10, 1),
    Task(5, (12, 12), 25, 4)
]


problem = Problem(
    robots=robots,
    tasks=tasks,
    objective=total_distance
)

solution = Solution([2, 1, 5, 3, 2])

fitness = problem.evaluate(solution)

print("Solution:", solution.variables)
print("Total distance:", fitness)