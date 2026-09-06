import math


def calculate_distance(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def total_distance(solution, problem):
    total = 0.0

    for task_index, robot_id in enumerate(solution):
        robot = problem.robots[robot_id - 1]
        task = problem.tasks[task_index]

        distance = calculate_distance(
            robot.position,
            task.position
        )

        total += distance

    return total