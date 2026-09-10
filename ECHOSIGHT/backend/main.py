from navigation.engine import make_navigation_decision
from services.assemblyai import run_voice_agent
from vision.depth import estimate_depth
from vision.detector import detect
from vision.scene import analyze_scene, describe_scene


IMAGE_PATH = "bus.jpg"


def analyze_current_scene():
    """Run EchoSight's complete vision pipeline on the current image."""
    print("\n[ECHOSIGHT] Analyzing scene...")

    detections = detect(IMAGE_PATH)
    depth_map = estimate_depth(IMAGE_PATH)
    scene = analyze_scene(detections, depth_map)
    decision = make_navigation_decision(scene)

    print("[ECHOSIGHT] Scene:")
    for obj in scene:
        print(
            f"  {obj['class']} | "
            f"confidence: {obj['confidence']:.2f} | "
            f"direction: {obj['direction']} | "
            f"distance: {obj['distance']} | "
            f"meters: {obj['distance_meters']:.2f}"
        )

    print(f"[ECHOSIGHT] Navigation: {decision['message']}")
    return scene, decision


def handle_voice_command(command):
    """Convert a spoken request into an EchoSight response."""
    command = command.lower().strip()
    scene, decision = analyze_current_scene()

    if any(phrase in command for phrase in (
        "what do you see",
        "what is ahead",
        "what's ahead",
        "what can you see",
        "what is in front",
        "what's in front",
        "what is around me",
        "what's around me",
        "what is on my left",
        "what is on my right",
    )):
        return describe_scene(scene)

    if any(phrase in command for phrase in (
        "path",
        "navigate",
        "where should i go",
        "where do i go",
        "can i walk",
        "is it safe",
        "obstacle",
        "can i move",
        "which way should i go",
    )):
        return decision["message"]

    return describe_scene(scene)


if __name__ == "__main__":
    print("========================================")
    print("        ECHOSIGHT VOICE AGENT")
    print("========================================")
    print("Vision: YOLO + Depth + Spatial Reasoning")
    print("Voice: AssemblyAI Realtime Voice Agent")
    print("Press Ctrl+C to stop.\n")

    run_voice_agent(on_command=handle_voice_command)