import math

from kodane_library.algorithms.cpo import CPO
from kodane_library.core.problem import Problem


# ============================================================
# PATH PLANNING CONFIGURATION
# ============================================================

START = (0.0, 0.0)
GOAL = (10.0, 10.0)

# Rectangular obstacles:
# (x_min, y_min, x_max, y_max)

OBSTACLES = [
    (3.0, 3.0, 5.5, 7.0),
    (6.0, 1.5, 8.0, 5.5),
    (6.0, 7.0, 9.0, 8.5),
]

NUM_WAYPOINTS = 5

# 5 waypoints × 2 coordinates = 10 decision variables
DIMENSION = NUM_WAYPOINTS * 2

WORKSPACE_BOUNDS = [
    (0.0, 10.0),
    (0.0, 10.0),
] * NUM_WAYPOINTS


# ============================================================
# GEOMETRY
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def point_inside_obstacle(point, obstacle):
    x, y = point
    x_min, y_min, x_max, y_max = obstacle

    return (
        x_min <= x <= x_max
        and
        y_min <= y <= y_max
    )


def orientation(a, b, c):
    value = (
        (b[1] - a[1]) * (c[0] - b[0])
        -
        (b[0] - a[0]) * (c[1] - b[1])
    )

    if abs(value) < 1e-10:
        return 0

    return 1 if value > 0 else 2


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and
        min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    if o2 == 0 and on_segment(p1, q2, p2):
        return True

    if o3 == 0 and on_segment(q1, p1, q2):
        return True

    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def segment_hits_obstacle(p1, p2, obstacle):

    x_min, y_min, x_max, y_max = obstacle

    if point_inside_obstacle(p1, obstacle):
        return True

    if point_inside_obstacle(p2, obstacle):
        return True

    corners = [
        ((x_min, y_min), (x_max, y_min)),
        ((x_max, y_min), (x_max, y_max)),
        ((x_max, y_max), (x_min, y_max)),
        ((x_min, y_max), (x_min, y_min)),
    ]

    for edge_start, edge_end in corners:

        if segments_intersect(
            p1,
            p2,
            edge_start,
            edge_end
        ):
            return True

    return False


# ============================================================
# PATH UTILITIES
# ============================================================

def decode_path(solution):

    waypoints = []

    for i in range(0, len(solution), 2):

        waypoints.append(
            (
                solution[i],
                solution[i + 1]
            )
        )

    return [START] + waypoints + [GOAL]


def count_collisions(path):

    collisions = 0

    for i in range(len(path) - 1):

        p1 = path[i]
        p2 = path[i + 1]

        for obstacle in OBSTACLES:

            if segment_hits_obstacle(
                p1,
                p2,
                obstacle
            ):
                collisions += 1

    return collisions


# ============================================================
# PATH OBJECTIVE
# ============================================================

def path_objective(solution, problem):

    path = decode_path(solution)

    total_distance = 0.0
    obstacle_penalty = 0.0

    for i in range(len(path) - 1):

        p1 = path[i]
        p2 = path[i + 1]

        # ----------------------------------------------------
        # Path length
        # ----------------------------------------------------

        total_distance += distance(p1, p2)

        # ----------------------------------------------------
        # Obstacle penalty
        # ----------------------------------------------------

        for obstacle in OBSTACLES:

            x_min, y_min, x_max, y_max = obstacle

            # Hard collision penalty
            if segment_hits_obstacle(
                p1,
                p2,
                obstacle
            ):
                obstacle_penalty += 500.0

            # Soft clearance penalty
            #
            # This gives CPO information about points that
            # are getting close to obstacles.

            for point in (p1, p2):

                x, y = point

                dx = max(
                    x_min - x,
                    0.0,
                    x - x_max
                )

                dy = max(
                    y_min - y,
                    0.0,
                    y - y_max
                )

                clearance = math.sqrt(
                    dx * dx +
                    dy * dy
                )

                if clearance < 1.0:

                    obstacle_penalty += (
                        (1.0 - clearance) * 50.0
                    )

    return total_distance + obstacle_penalty


# ============================================================
# CREATE KODANE PROBLEM
# ============================================================

problem = Problem(
    objective=path_objective,
    dimension=DIMENSION,
    bounds=WORKSPACE_BOUNDS
)


# ============================================================
# CREATE CPO
# ============================================================

optimizer = CPO(
    problem,
    population_size=30,
    max_iterations=100,
    seed=42
)


# ============================================================
# OPTIMIZE
# ============================================================

best = optimizer.optimize()


# ============================================================
# DECODE BEST PATH
# ============================================================

best_path = decode_path(best.variables)

best_waypoints = best_path[1:-1]

collisions = count_collisions(best_path)


# ============================================================
# CALCULATE ACTUAL PATH LENGTH
# ============================================================

actual_path_length = 0.0

for i in range(len(best_path) - 1):

    actual_path_length += distance(
        best_path[i],
        best_path[i + 1]
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("========================================")
print("       KODANE - CPO PATH PLANNING")
print("========================================")

print()
print("Start:")
print(f"  {START}")

print()
print("Waypoints:")

for i, waypoint in enumerate(
    best_waypoints,
    start=1
):

    print(
        f"  W{i}: "
        f"({waypoint[0]:.4f}, "
        f"{waypoint[1]:.4f})"
    )

print()
print("Goal:")
print(f"  {GOAL}")

print()
print("----------------------------------------")

print(
    f"Path Length:       {actual_path_length:.6f}"
)

print(
    f"Best Fitness:      {best.fitness:.6f}"
)

print(
    f"Obstacle Collisions: {collisions}"
)

print(
    f"Objective Evaluations: {optimizer.nfe}"
)

print(
    f"Iterations:        {optimizer.max_iterations}"
)

print("----------------------------------------")

if collisions == 0:
    print("STATUS: VALID COLLISION-FREE PATH")
else:
    print("STATUS: PATH STILL INTERSECTS OBSTACLES")

print("========================================")