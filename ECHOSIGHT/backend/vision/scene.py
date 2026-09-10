from .metric_depth import calibrate_depth

from navigation.spatial import (
    get_object_depth,
    classify_distance,
    build_spatial_description
)


def analyze_scene(detections, depth_map):
    """
    Combine YOLO detections, Depth Anything V2,
    and relative spatial reasoning.

    Pipeline:

        YOLO
          ↓
        bounding boxes
          ↓
        Depth Anything V2
          ↓
        Relative spatial reasoning
          ↓
        Structured scene
    """

    scene = []

    # --------------------------------
    # Scene depth distribution
    # --------------------------------

    valid_depth = depth_map[
        depth_map == depth_map
    ]

    for detection in detections:

        box = detection["box"]

        # --------------------------------
        # 1. Get relative depth
        # --------------------------------

        relative_depth = get_object_depth(
            depth_map,
            box
        )

        # --------------------------------
        # 2. Classify relative distance
        # --------------------------------

        distance = classify_distance(
            relative_depth,
            valid_depth
        )

        # --------------------------------
        # 3. Optional estimated meters
        # --------------------------------

        distance_meters = calibrate_depth(
            relative_depth
        )

        # --------------------------------
        # 4. Spatial reasoning
        # --------------------------------

        spatial = build_spatial_description(
            object_class=detection["class"],
            direction=detection["direction"],
            vertical_position=detection["vertical_position"],
            distance=distance,
            box=box,
            image_width=depth_map.shape[1],
            image_height=depth_map.shape[0]
        )

        # --------------------------------
        # 5. Build final object state
        # --------------------------------

        object_info = {
            "class": detection["class"],
            "confidence": detection["confidence"],
            "box": box,

            # Spatial information
            "direction": spatial["direction"],
            "vertical_position": spatial["vertical_position"],
            "position": spatial["position"],

            # Depth
            "relative_depth": relative_depth,

            # Approximate metric estimate
            "distance_meters": distance_meters,

            # Relative distance
            "distance": spatial["distance"],

            # Navigation information
            "path_score": spatial["path_score"],
            "box_area_ratio": spatial["box_area_ratio"],
            "in_walking_path": spatial["in_walking_path"]
        }

        scene.append(object_info)

    return scene


def describe_object(object_info):
    """
    Convert structured spatial information
    into a human-readable sentence.
    """

    object_class = object_info["class"]
    position = object_info["position"]
    distance = object_info["distance"]

    return (
        f"There is a {object_class} "
        f"{position}, {distance}."
    )


def describe_scene(scene):
    """
    Convert the entire scene into natural language.
    """

    if not scene:
        return (
            "I don't see any recognizable objects."
        )

    descriptions = []

    for object_info in scene:

        descriptions.append(
            describe_object(object_info)
        )

    return " ".join(descriptions)