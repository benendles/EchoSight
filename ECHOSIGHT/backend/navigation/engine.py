"""
ECHOSIGHT NAVIGATION ENGINE

Navigation decision engine.

Input:
    Structured scene produced by vision/scene.py

Output:
    Navigation instruction such as:

        STOP
        CAUTION
        MOVE LEFT
        MOVE RIGHT
        CONTINUE

This is a prototype reasoning layer for the
EchoSight hackathon demo. It is NOT a
safety-certified collision avoidance system.
"""


# CONFIGURATION

# Objects that should generally be treated as
# important obstacles when they are in the path.

OBSTACLE_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
}


# OBJECT PRIORITY

def get_object_priority(object_info):
    """
    Determine how important an object is
    for navigation.

    Higher value = more important.
    """

    object_class = object_info["class"]

    path_score = object_info["path_score"]

    distance = object_info["distance"]

    confidence = object_info["confidence"]


    # Base priority

    priority = path_score


    # Distance weighting

    if distance == "close":
        priority += 0.30
    elif distance == "medium":
        priority += 0.15
    elif distance == "far":
        priority += 0.00


    # Object type

    if object_class in OBSTACLE_CLASSES:
        priority += 0.10


    # Detection confidence

    if confidence >= 0.80:
        priority += 0.05

    return min(priority, 1.0)


# FIND MOST IMPORTANT OBSTACLE

def find_primary_obstacle(scene):
    """
    Find the object that currently represents
    the greatest navigation concern.

    Returns:
        object_info
        or None
    """

    obstacles = []

    for obj in scene:

        if not obj["in_walking_path"]:
            continue

        if obj["class"] not in OBSTACLE_CLASSES:
            continue

        priority = get_object_priority(obj)

        obstacles.append(
            (
                priority,
                obj
            )
        )


    if not obstacles:

        return None


    # Highest priority first

    obstacles.sort(
        key=lambda item: item[0],
        reverse=True
    )


    return obstacles[0][1]


# --------------------------------------------
# NAVIGATION DECISION
# --------------------------------------------

def make_navigation_decision(scene):
    """
    Convert scene understanding into a
    navigation command.

    Possible commands:

        STOP
        CAUTION
        MOVE LEFT
        MOVE RIGHT
        CONTINUE
    """

    # No objects

    if not scene:

        return {
            "action": "CONTINUE",
            "priority": 0.0,
            "message": "The path appears clear."
        }


    # Find primary obstacle

    obstacle = find_primary_obstacle(scene)


    # No obstacle

    if obstacle is None:

        return {
            "action": "CONTINUE",
            "priority": 0.0,
            "message": "The path appears clear."
        }


    # Extract information

    object_class = obstacle["class"]

    position = obstacle["position"]

    direction = obstacle["direction"]

    distance = obstacle["distance"]

    path_score = obstacle["path_score"]

    priority = get_object_priority(
        obstacle
    )


    # CLOSE OBSTACLE

    if distance == "close":

        return {
            "action": "STOP",
            "priority": priority,
            "object": object_class,
            "position": position,
            "distance": distance,
            "message": (
                f"Stop. There is a "
                f"{object_class} "
                f"{position}."
            )
        }


    # HIGH PATH SCORE

    if path_score >= 0.80:

        return {
            "action": "STOP",
            "priority": priority,
            "object": object_class,
            "position": position,
            "distance": distance,
            "message": (
                f"Stop. There is a "
                f"{object_class} "
                f"{position}."
            )
        }


    # MEDIUM DISTANCE

    if distance == "medium":

        # Object on the left

        if direction == "left":

            return {
                "action": "MOVE RIGHT",
                "priority": priority,
                "object": object_class,
                "position": position,
                "distance": distance,
                "message": (
                    f"Caution. There is a "
                    f"{object_class} "
                    f"to your left. "
                    f"Move right."
                )
            }


        # Object on the right

        if direction == "right":

            return {
                "action": "MOVE LEFT",
                "priority": priority,
                "object": object_class,
                "position": position,
                "distance": distance,
                "message": (
                    f"Caution. There is a "
                    f"{object_class} "
                    f"to your right. "
                    f"Move left."
                )
            }


        # Object directly ahead

        if direction == "center":

            return {
                "action": "CAUTION",
                "priority": priority,
                "object": object_class,
                "position": position,
                "distance": distance,
                "message": (
                    f"Caution. There is a "
                    f"{object_class} "
                    f"directly ahead."
                )
            }


    # FALLBACK

    return {
        "action": "CONTINUE",
        "priority": priority,
        "object": object_class,
        "position": position,
        "distance": distance,
        "message": "Continue carefully."
    }


# HUMAN-READABLE NAVIGATION

def describe_navigation(decision):
    """
    Return only the spoken navigation message.
    """

    return decision["message"]