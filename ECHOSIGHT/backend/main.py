import base64
import io
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import uvicorn
import ssl
import certifi

# Try to import vision/navigation modules; provide clear stubs if missing
_MISSING_VISION_DEPS = []

try:
    from navigation.engine import make_navigation_decision
except Exception as _err:
    try:
        from backend.navigation.engine import make_navigation_decision
    except Exception as _err2:
        make_navigation_decision = None
        _MISSING_VISION_DEPS.append(
            f"navigation.engine import error: {_err}; {getattr(_err2, 'args', _err2)}"
        )

try:
    from vision.depth import estimate_depth
except Exception as _err:
    try:
        from backend.vision.depth import estimate_depth
    except Exception as _err2:
        estimate_depth = None
        _MISSING_VISION_DEPS.append(
            f"vision.depth import error: {_err}; {getattr(_err2, 'args', _err2)}"
        )

try:
    from vision.detector import detect
except Exception as _err:
    try:
        from backend.vision.detector import detect
    except Exception as _err2:
        detect = None
        _MISSING_VISION_DEPS.append(
            f"vision.detector import error: {_err}; {getattr(_err2, 'args', _err2)}"
        )

try:
    from vision.scene import analyze_scene, describe_scene
except Exception as _err:
    try:
        from backend.vision.scene import analyze_scene, describe_scene
    except Exception as _err2:
        analyze_scene = None
        describe_scene = None
        _MISSING_VISION_DEPS.append(
            f"vision.scene import error: {_err}; {getattr(_err2, 'args', _err2)}"
        )


def _ensure_vision_available():
    """Raise a helpful error when vision modules are not available."""
    if _MISSING_VISION_DEPS:
        details = "; ".join(_MISSING_VISION_DEPS)
        raise RuntimeError(
            "Vision dependencies are not installed or failed to import. "
            "Install the needed packages (see backend/requirements.txt) or fix import errors. "
            f"Details: {details}"
        )
from threading import Lock
import time

# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Thread-safe latest-scene cache used to answer voice queries
LATEST_SCENE: dict = {
    "scene": [],
    "navigation": None,
    "description": "",
    "answer": "",
    "timestamp": 0,
}

LATEST_SCENE_LOCK = Lock()


def _json_safe(value):
    """Convert model output into values supported by JSON responses."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "item"):
        return _json_safe(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def update_latest_scene(scene, decision, description, answer=""):
    with LATEST_SCENE_LOCK:
        LATEST_SCENE["scene"] = scene
        LATEST_SCENE["navigation"] = decision
        LATEST_SCENE["description"] = description
        LATEST_SCENE["answer"] = answer
        LATEST_SCENE["timestamp"] = int(time.time())


# FASTAPI

app = FastAPI(
    title="EchoSight API",
    description="EchoSight computer vision and navigation backend",
    version="1.0.0",
)


# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# IMAGE VALIDATION

ALLOWED_IMAGE_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
    "GIF",
    "TIFF",
}


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Validate and decode an image from raw bytes.
    """

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))

        image.seek(0)
        image.load()

    except UnidentifiedImageError:
        raise ValueError(
            "Could not identify this image format. "
            "Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    except (OSError, SyntaxError) as error:
        raise ValueError(
            f"Could not read image: {error}. "
            "Please convert the image to JPG or PNG and try again."
        )

    except Exception as error:
        raise ValueError(f"Could not read image: {error}")

    image_format = (image.format or "").upper()

    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported image format: "
            f"{image_format or 'unknown'}. "
            "Please upload JPG, PNG, or WEBP."
        )

    try:
        return image.convert("RGB")

    except Exception as error:
        raise ValueError(
            f"Could not convert image to RGB: {error}"
        )


# ============================================================
# VISION PIPELINE
#
def analyze_image(image_path: Path):
    """
    Complete EchoSight vision pipeline.

    Image
       ↓
    YOLO object detection
       ↓
    Depth Anything V2
       ↓
    Spatial reasoning
       ↓
    Navigation decision
    """

    print(
        f"\n[ECHOSIGHT] Analyzing: {image_path.name}"
    )

    detections = detect(
        str(image_path)
    )

    depth_map = estimate_depth(
        str(image_path)
    )

    scene = analyze_scene(
        detections,
        depth_map,
    )

    decision = make_navigation_decision(
        scene
    )

    print("[ECHOSIGHT] Scene:")

    for obj in scene:
        print(
            f"  {obj['class']} | "
            f"confidence: {obj['confidence']:.2f} | "
            f"direction: {obj['direction']} | "
            f"distance: {obj['distance']} | "
            f"meters: {obj['distance_meters']:.2f}"
        )

    print(
        "[ECHOSIGHT] Navigation:",
        decision["message"],
    )

    # Update the global latest scene cache so voice queries
    # can be answered without re-running the full vision stack.
    description = describe_scene(scene)
    update_latest_scene(scene, decision, description)

    return scene, decision


# ============================================================
# ANALYZE IMAGE BYTES
#
def analyze_image_bytes(
    image_bytes: bytes,
    filename: str = "uploaded_image.jpg",
):
    """
    Convert an uploaded image to JPEG and run the
    EchoSight vision pipeline.
    """

    image = load_image_from_bytes(
        image_bytes
    )

    safe_name = (
        Path(filename)
        .stem
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    frame_path = (
        UPLOAD_DIR
        / f"{safe_name}_current.jpg"
    )

    image.save(
        frame_path,
        format="JPEG",
        quality=90,
    )

    return analyze_image(
        frame_path
    )


# ============================================================
# VOICE COMMAND HANDLER
# ============================================================

def handle_voice_command(command: str):
    """
    Analyze the current camera frame in response
    to an AssemblyAI voice-agent tool call.
    """

    command = command.lower().strip()

    current_frame = (
        UPLOAD_DIR / "current_frame.jpg"
    )

    if not current_frame.exists():
        return {
            "answer": (
                "I don't have a current camera image "
                "to analyze yet."
            ),
            "scene": [],
            "navigation": None,
            "description": "No current camera frame is available.",
        }

    scene, decision = analyze_image(
        current_frame
    )

    scene_phrases = (
        "what do you see",
        "what is ahead",
        "what's ahead",
        "what can you see",
        "what is in front",
        "what's in front",
        "what is around me",
        "what's around me",
        "what is on my left",
        "what's on my left",
        "what is on my right",
        "what's on my right",
        "how many people",
        "how many persons",
        "how many objects",
        "describe the scene",
        "describe what you see",
    )

    navigation_phrases = (
        "path",
        "navigate",
        "where should i go",
        "where do i go",
        "can i walk",
        "is it safe",
        "obstacle",
        "obstacles",
        "can i move",
        "which way should i go",
        "which direction",
        "should i move",
        "can i go forward",
        "can i go ahead",
        "what direction",
    )

    description = describe_scene(scene)

    if any(
        phrase in command
        for phrase in navigation_phrases
    ):
        answer = decision["message"]

    elif any(
        phrase in command
        for phrase in scene_phrases
    ):
        answer = description

    else:
        answer = description

    return {
        "answer": answer,
        "scene": scene,
        "navigation": decision,
        "description": description,
    }


# ============================================================
# ASSEMBLYAI TEMPORARY VOICE TOKEN
# ============================================================

@app.get("/api/voice-token")
async def voice_token():
    """
    Create a short-lived AssemblyAI Voice Agent token.

    IMPORTANT:
    The real AssemblyAI API key stays on the backend.
    The browser only receives the temporary token.
    """

    api_key = os.getenv(
        "ASSEMBLYAI_API_KEY"
    )

    if not api_key:
        return {
            "type": "error",
            "message": (
                "ASSEMBLYAI_API_KEY is not configured "
                "on the backend."
            ),
        }

    try:
        query = urllib.parse.urlencode(
            {
                "expires_in_seconds": "300",
            }
        )

        request = urllib.request.Request(
            (
                "https://agents.assemblyai.com/v1/token"
                f"?{query}"
            ),
            headers={
                "Authorization": (
                    f"Bearer {api_key}"
                ),
            },
            method="GET",
        )

        # Use a certifi-backed SSL context to avoid macOS / custom-Python
        # certificate verification issues when calling external HTTPS APIs.
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(
            request,
            timeout=15,
            context=ssl_ctx,
        ) as response:

            data = json.loads(response.read().decode("utf-8"))

        token = data.get("token")

        if not token:
            raise RuntimeError(
                "AssemblyAI did not return a token."
            )

        return {
            "token": token,
        }

    except Exception as error:

        print(
            "[Voice] Token error:",
            error,
        )

        return {
            "type": "error",
            "message": (
                "Could not create an AssemblyAI "
                "voice token."
            ),
        }


# ============================================================
# ASSEMBLYAI VISION TOOL
# ============================================================

@app.post("/api/voice/analyze")
async def voice_analyze(
    payload: dict = Body(...),
):
    """
    Endpoint called by the browser when AssemblyAI
    requests EchoSight visual analysis.

    Returns both the natural-language answer for the
    voice agent and the complete structured vision data
    for the frontend.
    """

    command = str(
        payload.get(
            "command",
            "",
        )
    ).strip()

    if not command:
        return {
            "type": "error",
            "message": (
                "No voice command was provided."
            ),
        }

    print(
        f"\n[Voice] Vision request: {command}"
    )

    try:
        current_frame = UPLOAD_DIR / "current_frame.jpg"

        if not current_frame.exists():
            result = {
                "answer": (
                    "I don't have a current camera image "
                    "to analyze yet."
                ),
                "scene": [],
                "navigation": None,
                "description": "No current camera frame is available.",
            }

        else:
            # Use the cached latest scene if available to avoid
            # re-running the heavy vision pipeline for each voice query.
            with LATEST_SCENE_LOCK:
                cached_ts = LATEST_SCENE.get("timestamp", 0)
                cached_scene = LATEST_SCENE.get("scene", [])
                cached_navigation = LATEST_SCENE.get("navigation", None)
                cached_description = LATEST_SCENE.get("description", "")

            # If we have a recent cached scene (e.g., < 60s), use it.
            if cached_ts and (int(time.time()) - cached_ts) < 60:
                scene = cached_scene
                decision = cached_navigation
                description = cached_description
                answer = description
            else:
                scene, decision = analyze_image(current_frame)
                description = describe_scene(scene)
                answer = description

            # Decide if the command is navigation-related.
            command_lower = command.lower()

            navigation_phrases = (
                "path",
                "navigate",
                "where should i go",
                "where do i go",
                "can i walk",
                "is it safe",
                "obstacle",
                "obstacles",
                "can i move",
                "which way should i go",
                "which direction",
                "should i move",
                "can i go forward",
                "can i go ahead",
                "what direction",
            )

            if any(phrase in command_lower for phrase in navigation_phrases):
                answer = (decision or {}).get("message") if decision else description

            result = {
                "answer": answer,
                "scene": scene,
                "navigation": decision,
                "description": description,
            }

        print(
            "[Voice] Vision result:",
            result["answer"],
        )

        return {
            "type": "tool_result",
            "result": json.dumps(result),
        }

    except Exception as error:

        print(
            "[Voice] Vision tool error:",
            error,
        )

        return {
            "type": "error",
            "message": (
                "EchoSight could not analyze "
                "the current scene."
            ),
        }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name": "EchoSight",
        "status": "online",
        "vision": (
            "YOLO + Depth Anything V2 "
            "+ Spatial Reasoning"
        ),
        "voice": "AssemblyAI Voice Agent",
        "image_support": [
            "JPEG",
            "PNG",
            "WEBP",
            "BMP",
            "GIF",
            "TIFF",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok"
    }


# ============================================================
# IMAGE UPLOAD
# ============================================================

@app.post("/api/vision/upload")
async def upload_image(
    file: UploadFile = File(...),
):
    """
    Analyze an uploaded image and make it the
    current frame used by the voice agent.
    """

    print(
        f"\n[Upload] Received: {file.filename}"
    )

    if file.content_type:

        if not file.content_type.startswith(
            "image/"
        ):

            return {
                "type": "error",
                "message": (
                    "Please upload an image file."
                ),
            }

    try:

        image_bytes = await file.read()

        if not image_bytes:

            return {
                "type": "error",
                "message": (
                    "The uploaded image is empty."
                ),
            }

        scene, decision = analyze_image_bytes(
            image_bytes,
            file.filename
            or "uploaded_image.jpg",
        )

        current_frame = (
            UPLOAD_DIR
            / "current_frame.jpg"
        )

        image = load_image_from_bytes(
            image_bytes
        )

        image.save(
            current_frame,
            format="JPEG",
            quality=90,
        )

        # Update latest scene cache from this upload.
        description = describe_scene(scene)
        update_latest_scene(scene, decision, description)

        return _json_safe({
            "type": "vision_result",
            "filename": file.filename,
            "scene": scene,
            "navigation": decision,
            "description": description,
        })

    except ValueError as error:

        return {
            "type": "error",
            "message": str(error),
        }

    except Exception as error:

        print(f"[Upload] Vision error: {error!r}")

        return {
            "type": "error",
            "message": (
                "Could not analyze the image. "
                "Check the Railway service logs for details."
            ),
        }


# ============================================================
# TEST CURRENT FRAME
# ============================================================

@app.get("/api/vision/test")
async def test_vision():
    """
    Analyze the current frame.

    Useful for testing the vision pipeline
    independently of the frontend.
    """

    current_frame = (
        UPLOAD_DIR / "current_frame.jpg"
    )

    if not current_frame.exists():

        return {
            "type": "error",
            "message": (
                "No image has been uploaded yet."
            ),
        }

    scene, decision = analyze_image(
        current_frame
    )

    return {
        "type": "vision_result",
        "scene": scene,
        "navigation": decision,
        "description": describe_scene(
            scene
        ),
    }


# ============================================================
# LIVE VISION WEBSOCKET
# ============================================================

@app.websocket("/ws/vision")
async def vision_websocket(
    websocket: WebSocket,
):
    """
    Live camera WebSocket.
    """

    await websocket.accept()

    print(
        "[WebSocket] Frontend connected."
    )

    try:

        while True:

            message = (
                await websocket.receive_json()
            )

            message_type = message.get(
                "type"
            )

            if message_type == "ping":

                await websocket.send_json(
                    {
                        "type": "pong"
                    }
                )

                continue

            if message_type != "frame":

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Unknown message type."
                        ),
                    }
                )

                continue

            image_base64 = message.get(
                "image"
            )

            if not image_base64:

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "No image provided."
                        ),
                    }
                )

                continue

            try:

                if "," in image_base64:

                    image_base64 = (
                        image_base64.split(
                            ",",
                            1,
                        )[1]
                    )

                image_bytes = (
                    base64.b64decode(
                        image_base64
                    )
                )

                image = load_image_from_bytes(
                    image_bytes
                )

            except Exception as error:

                print(
                    "[WebSocket] "
                    f"Image decode error: {error}"
                )

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Could not decode "
                            "camera image."
                        ),
                    }
                )

                continue

            frame_path = (
                UPLOAD_DIR
                / "current_frame.jpg"
            )

            image.save(
                frame_path,
                format="JPEG",
                quality=85,
            )

            try:

                scene, decision = (
                    analyze_image(
                        frame_path
                    )
                )

                result = _json_safe({
                    "type": "vision_result",
                    "scene": scene,
                    "navigation": decision,
                    "description": (
                        describe_scene(
                            scene
                        )
                    ),
                })

                await websocket.send_json(
                    result
                )

                # Update latest scene cache for voice queries.
                try:
                    description = describe_scene(scene)
                    update_latest_scene(scene, decision, description)
                except Exception:
                    pass

            except Exception as error:

                print(
                    "[WebSocket] "
                    f"Vision error: {error}"
                )

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": (
                            "Vision analysis "
                            "failed."
                        ),
                    }
                )

    except WebSocketDisconnect:

        print(
            "[WebSocket] "
            "Frontend disconnected."
        )

    except Exception as error:

        print(
            "[WebSocket] Connection error:",
            error,
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "          ECHOSIGHT BACKEND"
    )

    print(
        "========================================"
    )

    print(
        "HTTP:        http://localhost:8000"
    )

    print(
        "Vision WS:   ws://localhost:8000/ws/vision"
    )

    print(
        "Docs:        http://localhost:8000/docs"
    )

    print(
        "Upload:      POST /api/vision/upload"
    )

    print(
        "Voice token: GET  /api/voice-token"
    )

    print(
        "Voice tool:  POST /api/voice/analyze"
    )

    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )