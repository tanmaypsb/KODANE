import math

from kodane_library.algorithms.rvea import RVEA
from kodane_library.core.problem import Problem


# ============================================================
# PATH PLANNING CONFIGURATION
# ============================================================

START = (0.0, 0.0)
GOAL = (10.0, 10.0)

OBSTACLES = [
    (3.0, 3.0, 5.5, 7.0),
    (6.0, 1.5, 8.0, 5.5),
    (6.0, 7.0, 9.0, 8.5),
]

NUM_WAYPOINTS = 5
DIMENSION = NUM_WAYPOINTS * 2

BOUNDS = [
    (0.0, 10.0),
    (0.0, 10.0),
] * NUM_WAYPOINTS


# ============================================================
# GEOMETRY
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
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

    edges = [
        ((x_min, y_min), (x_max, y_min)),
        ((x_max, y_min), (x_max, y_max)),
        ((x_max, y_max), (x_min, y_max)),
        ((x_min, y_max), (x_min, y_min)),
    ]

    for edge_start, edge_end in edges:

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


def path_length(path):

    total = 0.0

    for i in range(len(path) - 1):

        total += distance(
            path[i],
            path[i + 1]
        )

    return total


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
# OBJECTIVE 1 — DISTANCE
# ============================================================

def distance_objective(path):

    return path_length(path)


# ============================================================
# OBJECTIVE 2 — ENERGY
# ============================================================

def energy_objective(path):

    energy = 0.0

    for i in range(len(path) - 1):

        segment = distance(
            path[i],
            path[i + 1]
        )

        # Slightly higher cost for longer movement.
        energy += segment ** 1.2

    return energy


# ============================================================
# OBJECTIVE 3 — SMOOTHNESS
# ============================================================

def smoothness_objective(path):

    smoothness = 0.0

    for i in range(1, len(path) - 1):

        previous = path[i - 1]
        current = path[i]
        following = path[i + 1]

        v1 = (
            current[0] - previous[0],
            current[1] - previous[1]
        )

        v2 = (
            following[0] - current[0],
            following[1] - current[1]
        )

        norm1 = math.sqrt(
            v1[0] ** 2 + v1[1] ** 2
        )

        norm2 = math.sqrt(
            v2[0] ** 2 + v2[1] ** 2
        )

        if norm1 < 1e-10 or norm2 < 1e-10:
            continue

        cosine = (
            (v1[0] * v2[0] + v1[1] * v2[1])
            / (norm1 * norm2)
        )

        cosine = max(
            -1.0,
            min(1.0, cosine)
        )

        angle = math.acos(cosine)

        smoothness += angle ** 2

    return smoothness


# ============================================================
# MULTI-OBJECTIVE FUNCTION
# ============================================================

def path_objective(solution, problem):

    path = decode_path(solution)

    collisions = count_collisions(path)

    distance_cost = distance_objective(path)
    energy_cost = energy_objective(path)
    smoothness_cost = smoothness_objective(path)

    # Large penalty keeps infeasible paths away from
    # the Pareto front.
    if collisions > 0:

        penalty = collisions * 1000.0

        distance_cost += penalty
        energy_cost += penalty
        smoothness_cost += penalty

    return [
        distance_cost,
        energy_cost,
        smoothness_cost
    ]


# ============================================================
# CREATE PROBLEM
# ============================================================

problem = Problem(
    objective=path_objective,
    dimension=DIMENSION,
    bounds=BOUNDS
)


# ============================================================
# CREATE RVEA
# ============================================================

optimizer = RVEA(
    problem,
    population_size=30,
    max_iterations=100,
    seed=42
)


# ============================================================
# RUN OPTIMIZATION
# ============================================================

pareto_front = optimizer.optimize()


# ============================================================
# RESULTS
# ============================================================

print()
print("==========================================================")
print("       KODANE - RVEA MULTI-OBJECTIVE PATH PLANNING")
print("==========================================================")

print()
print("Start:")
print(f"  {START}")

print()
print("Goal:")
print(f"  {GOAL}")

print()
print("Pareto Solutions:")
print(f"  {len(pareto_front)}")

print()
print("NFE:")
print(f"  {optimizer.nfe}")

print()
print("----------------------------------------------------------")
print("PARETO FRONT")
print("----------------------------------------------------------")

for index, solution in enumerate(
    pareto_front,
    start=1
):

    path = decode_path(
        solution.variables
    )

    collisions = count_collisions(path)

    print()
    print(f"Solution {index}")

    print(
        f"  Distance:   "
        f"{solution.objectives[0]:.6f}"
    )

    print(
        f"  Energy:     "
        f"{solution.objectives[1]:.6f}"
    )

    print(
        f"  Smoothness: "
        f"{solution.objectives[2]:.6f}"
    )

    print(
        f"  Collisions: "
        f"{collisions}"
    )

    print("  Waypoints:")

    for waypoint_index, waypoint in enumerate(
        path[1:-1],
        start=1
    ):

        print(
            f"    W{waypoint_index}: "
            f"({waypoint[0]:.4f}, "
            f"{waypoint[1]:.4f})"
        )

print()
print("==========================================================")
print("RVEA path-planning experiment completed.")
print("==========================================================")