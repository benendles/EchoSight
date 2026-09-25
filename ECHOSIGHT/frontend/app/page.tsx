"use client";

import {
  ChangeEvent,
  CSSProperties,
  DragEvent,
  SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const BACKEND_URL = (
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"
).replace(/\/$/, "");
const VISION_WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/vision`;
const ASSEMBLYAI_WS_URL = "wss://agents.assemblyai.com/v1/ws";

type VoiceState =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "error";

type Detection = {
  label?: string;
  class?: string;
  confidence?: number;
  distance?: number | string;
  distance_meters?: number;
  direction?: string;
  box?: number[];
  [key: string]: unknown;
};

type Navigation = {
  action?: string;
  direction?: string;
  distance?: number;
  message?: string;
  [key: string]: unknown;
};

type VisionResult = {
  type?: string;
  answer?: string;
  description?: string;
  scene_description?: string;
  scene?: Detection[];
  detections?: Detection[];
  navigation?: Navigation | null;
  result?: string | VisionResult;
  [key: string]: unknown;
};

type ConversationMessage = {
  id: number;
  role: "user" | "assistant" | "system";
  text: string;
};

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;

  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }

  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes.buffer;
}

function float32ToPCM16(input: Float32Array): ArrayBuffer {
  const output = new Int16Array(input.length);

  for (let i = 0; i < input.length; i++) {
    const sample = Math.max(-1, Math.min(1, input[i]));

    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }

  return output.buffer;
}

export default function Home() {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");

  const [connected, setConnected] = useState(false);

  const [voiceConnected, setVoiceConnected] = useState(false);

  const [cameraActive, setCameraActive] = useState(false);

  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [dragging, setDragging] = useState(false);

  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const [filename, setFilename] = useState("");

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });

  const [lastMessage, setLastMessage] = useState("Ready to assist.");

  const [conversation, setConversation] = useState<ConversationMessage[]>([]);

  const [scene, setScene] = useState("");

  const [navigation, setNavigation] = useState<Navigation | null>(null);

  const [sceneObjects, setSceneObjects] = useState<Detection[]>([]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const visionWsRef = useRef<WebSocket | null>(null);

  const assemblyWsRef = useRef<WebSocket | null>(null);

  const cameraStreamRef = useRef<MediaStream | null>(null);

  const microphoneStreamRef = useRef<MediaStream | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);

  const microphoneSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const imageUrlRef = useRef<string | null>(null);

  const analysisIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );

  const processingFrameRef = useRef(false);

  const pendingToolResultsRef = useRef<
    Array<{
      call_id: string;
      result: string;
    }>
  >([]);

  const playbackTimeRef = useRef(0);

  const messageIdRef = useRef(0);

  const mountedRef = useRef(true);

  const voiceStateRef = useRef<VoiceState>("idle");

  const addMessage = useCallback(
    (role: ConversationMessage["role"], text: string) => {
      if (!text.trim()) return;

      setConversation((prev) => [
        ...prev.slice(-30),
        {
          id: ++messageIdRef.current,
          role,
          text: text.trim(),
        },
      ]);
    },
    [],
  );

  /* VISION RESULT NORMALIZATION */

  const applyVisionResult = useCallback((rawResult: VisionResult) => {
    console.log("[EchoSight] RAW VISION RESULT:", rawResult);

    let result: VisionResult = rawResult;

    // Normalize every response shape used by EchoSight.
    if (rawResult?.result !== undefined) {
      if (typeof rawResult.result === "string") {
        try {
          result = JSON.parse(rawResult.result) as VisionResult;
        } catch {
          // AssemblyAI/tool responses can contain plain text.
          result = {
            ...rawResult,
            answer: rawResult.result,
          };
        }
      } else if (rawResult.result && typeof rawResult.result === "object") {
        result = rawResult.result as VisionResult;
      }
    }

    console.log("[EchoSight] NORMALIZED RESULT:", result);

    const objects = Array.isArray(result.scene)
      ? result.scene
      : Array.isArray(result.detections)
        ? result.detections
        : [];

    const description =
      typeof result.description === "string" && result.description.trim()
        ? result.description.trim()
        : typeof result.scene_description === "string" &&
            result.scene_description.trim()
          ? result.scene_description.trim()
          : typeof result.answer === "string"
            ? result.answer.trim()
            : "";

    const nav =
      result.navigation && typeof result.navigation === "object"
        ? result.navigation
        : null;

    console.log("[EchoSight] OBJECTS:", objects);
    console.log("[EchoSight] DESCRIPTION:", description);
    console.log("[EchoSight] NAVIGATION:", nav);

    setScene(description);
    setSceneObjects(objects);
    setNavigation(nav);
    setIsAnalyzing(false);

    if (typeof result.answer === "string" && result.answer.trim()) {
      setLastMessage(result.answer.trim());
    } else if (description) {
      setLastMessage(description);
    } else if (nav?.message) {
      setLastMessage(String(nav.message));
    } else if (objects.length > 0) {
      setLastMessage(
        `Detected ${objects.length} object${
          objects.length === 1 ? "" : "s"
        } in the scene.`,
      );
    } else {
      setLastMessage("Scene analyzed, but no visual information was returned.");
    }
  }, []);

  /* ANALYZE CURRENT FRAME */

  const analyzeCurrentFrame = useCallback(
    async (
      command = "Describe what is visible around me and identify obstacles or useful directions.",
    ) => {
      setIsAnalyzing(true);

      try {
        const response = await fetch(`${BACKEND_URL}/api/voice/analyze`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            command,
          }),
        });

        if (!response.ok) {
          throw new Error(`Vision request failed: ${response.status}`);
        }

        const rawResult = (await response.json()) as VisionResult;

        console.log("[EchoSight] /api/voice/analyze response:", rawResult);

        // All response normalization happens in one place.
        applyVisionResult(rawResult);

        return rawResult;
      } catch (error) {
        console.error("[EchoSight] Vision analysis error:", error);

        setIsAnalyzing(false);

        setLastMessage("I could not analyze the current camera frame.");

        return {
          error: "Vision analysis failed.",
        };
      }
    },
    [applyVisionResult],
  );

  /* CAMERA FRAME SENDING - Camera implementation intentionally left intact. */

  const sendFrame = useCallback(() => {
    if (processingFrameRef.current) {
      return;
    }

    const ws = visionWsRef.current;

    const video = videoRef.current;

    const canvas = canvasRef.current;

    if (
      !ws ||
      ws.readyState !== WebSocket.OPEN ||
      !video ||
      !canvas ||
      video.readyState < 2
    ) {
      return;
    }

    processingFrameRef.current = true;

    try {
      const maxWidth = 960;

      const scale = Math.min(1, maxWidth / Math.max(video.videoWidth, 1));

      canvas.width = Math.max(1, Math.round(video.videoWidth * scale));

      canvas.height = Math.max(1, Math.round(video.videoHeight * scale));

      const ctx = canvas.getContext("2d");

      if (!ctx) {
        processingFrameRef.current = false;

        return;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.72);

      ws.send(
        JSON.stringify({
          type: "frame",
          image: dataUrl.split(",")[1],
        }),
      );
    } catch (error) {
      console.error("[EchoSight] Frame send error:", error);

      processingFrameRef.current = false;
    }
  }, []);

  /* STOP CAMERA */

  const stopCamera = useCallback(() => {
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);

      analysisIntervalRef.current = null;
    }

    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());

    cameraStreamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  }, []);

  /* START CAMERA */

  const startCamera = useCallback(async () => {
    if (cameraActive) {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",

          width: {
            ideal: 1280,
          },

          height: {
            ideal: 720,
          },
        },

        audio: false,
      });

      cameraStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        await videoRef.current.play();
      }

      setCameraActive(true);

      setLastMessage("Camera active. EchoSight is watching the scene.");

      /*
       * Connect vision WebSocket.
       */

      if (
        !visionWsRef.current ||
        visionWsRef.current.readyState !== WebSocket.OPEN
      ) {
        const ws = new WebSocket(VISION_WS_URL);

        visionWsRef.current = ws;

        ws.onopen = () => {
          console.log("[EchoSight] Vision WebSocket connected.");

          setConnected(true);

          sendFrame();
        };

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);

            console.log("[EchoSight] Vision result:", message);

            if (message.type === "vision_result") {
              applyVisionResult(message as VisionResult);

              processingFrameRef.current = false;
            }
          } catch (error) {
            console.error("[EchoSight] Vision WS parse error:", error);

            processingFrameRef.current = false;

            setIsAnalyzing(false);
          }
        };

        ws.onerror = (event) => {
          console.error("[EchoSight] Vision WS error:", event);

          setConnected(false);

          setIsAnalyzing(false);

          setLastMessage(
            "Vision connection failed. Make sure the EchoSight backend is running.",
          );
        };

        ws.onclose = () => {
          console.log("[EchoSight] Vision WebSocket closed.");

          setConnected(false);

          setIsAnalyzing(false);

          processingFrameRef.current = false;
        };
      }

      /*
       * Continue analyzing frames.
       */

      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
      }

      analysisIntervalRef.current = setInterval(sendFrame, 1500);
    } catch (error) {
      console.error("[EchoSight] Camera error:", error);

      setLastMessage(
        "Camera permission was denied or the camera is unavailable.",
      );
    }
  }, [applyVisionResult, cameraActive, sendFrame]);

  /* ASSEMBLYAI MESSAGE HANDLER */

  const handleAssemblyAIMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);

        console.log("[AssemblyAI]", message);

        switch (message.type) {
          case "session.ready":
            setVoiceState("listening");

            setVoiceConnected(true);

            setLastMessage("Listening. Ask me what you see.");

            break;

          case "transcript.user.delta":
            if (message.text) {
              setLastMessage(message.text);
            }

            break;

          case "transcript.user":
            if (message.text) {
              addMessage("user", message.text);
            }

            break;

          case "reply.started":
            setVoiceState("speaking");

            setLastMessage("EchoSight is speaking...");

            break;

          case "reply.audio": {
            if (!message.data) {
              break;
            }

            const context = audioContextRef.current;

            if (!context) {
              break;
            }

            const pcm = new Int16Array(base64ToArrayBuffer(message.data));

            const audioBuffer = context.createBuffer(1, pcm.length, 24000);

            const channel = audioBuffer.getChannelData(0);

            for (let i = 0; i < pcm.length; i++) {
              channel[i] = pcm[i] / 32768;
            }

            const source = context.createBufferSource();

            source.buffer = audioBuffer;

            source.connect(context.destination);

            const now = context.currentTime;

            const startAt = Math.max(now + 0.005, playbackTimeRef.current);

            source.start(startAt);

            playbackTimeRef.current = startAt + audioBuffer.duration;

            source.onended = () => {
              if (
                mountedRef.current &&
                playbackTimeRef.current <= context.currentTime + 0.03
              ) {
                setVoiceState("listening");
              }
            };

            break;
          }

          case "transcript.agent":
            if (message.text) {
              addMessage("assistant", message.text);

              setLastMessage(message.text);
            }

            break;

          /*
           * AssemblyAI asks EchoSight
           * to analyze the current scene.
           */

          case "tool.call": {
            const callId = message.call_id;

            const name = message.name;

            const args = message.arguments || {};

            if (name !== "analyze_echo_sight_scene" || !callId) {
              break;
            }

            setVoiceState("thinking");

            const command =
              typeof args.command === "string"
                ? args.command
                : "Describe the current scene.";

            analyzeCurrentFrame(command).then((result) => {
              pendingToolResultsRef.current.push({
                call_id: callId,

                result: JSON.stringify(result),
              });
            });

            break;
          }

          /*
           * Send queued tool results
           * after reply.done.
           */

          case "reply.done": {
            const status = message.status;

            if (status === "interrupted") {
              pendingToolResultsRef.current = [];

              playbackTimeRef.current =
                audioContextRef.current?.currentTime || 0;

              setVoiceState("listening");

              break;
            }

            const ws = assemblyWsRef.current;

            if (
              ws?.readyState === WebSocket.OPEN &&
              pendingToolResultsRef.current.length
            ) {
              const pending = pendingToolResultsRef.current.splice(0);

              for (const item of pending) {
                ws.send(
                  JSON.stringify({
                    type: "tool.result",

                    call_id: item.call_id,

                    result: item.result,
                  }),
                );
              }
            }

            if (voiceState !== "speaking") {
              setVoiceState("listening");
            }

            break;
          }

          case "session.error":
            console.error("[EchoSight] AssemblyAI session error:", message);

            setVoiceState("error");

            setLastMessage(message.message || "AssemblyAI session error.");

            break;

          default:
            break;
        }
      } catch (error) {
        console.error("[EchoSight] AssemblyAI message error:", error);
      }
    },
    [addMessage, analyzeCurrentFrame, voiceState],
  );

  /* STOP VOICE */

  const stopVoice = useCallback(() => {
    processorRef.current?.disconnect();

    processorRef.current = null;

    microphoneSourceRef.current?.disconnect();

    microphoneSourceRef.current = null;

    microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());

    microphoneStreamRef.current = null;

    if (audioContextRef.current) {
      void audioContextRef.current.close();

      audioContextRef.current = null;
    }

    playbackTimeRef.current = 0;

    if (assemblyWsRef.current) {
      assemblyWsRef.current.close();

      assemblyWsRef.current = null;
    }

    pendingToolResultsRef.current = [];

    setVoiceConnected(false);

    setVoiceState("idle");
  }, []);

  /* START VOICE */

  const startVoice = useCallback(async () => {
    if (
      voiceState === "connecting" ||
      voiceState === "listening" ||
      voiceState === "thinking" ||
      voiceState === "speaking"
    ) {
      return;
    }

    setVoiceState("connecting");

    setLastMessage("Connecting to EchoSight voice assistant...");

    try {
      const tokenResponse = await fetch(`${BACKEND_URL}/api/voice-token`);

      if (!tokenResponse.ok) {
        throw new Error(`Token request failed: ${tokenResponse.status}`);
      }

      const tokenData = await tokenResponse.json();

      if (!tokenData.token) {
        throw new Error("No temporary AssemblyAI token returned.");
      }

      const ws = new WebSocket(
        `${ASSEMBLYAI_WS_URL}?token=${encodeURIComponent(tokenData.token)}`,
      );

      assemblyWsRef.current = ws;

      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: "session.update",

            session: {
              system_prompt:
                "You are EchoSight, an accessibility assistant for a blind or visually impaired person. Keep spoken answers concise, natural, and safety-focused. When the user asks what they see, what is ahead, where to go, or about obstacles, use the analyze_echo_sight_scene tool. Never invent visual information.",

              greeting: "EchoSight is ready. Tell me what you need help with.",

              output: {
                voice: "ivy",
              },

              tools: [
                {
                  type: "function",

                  name: "analyze_echo_sight_scene",

                  description:
                    "Analyze the current EchoSight camera frame with object detection, depth estimation, spatial reasoning, and navigation. Use this whenever the user asks about what they see, objects or people around them, obstacles, directions, or whether it is safe to move.",

                  parameters: {
                    type: "object",

                    properties: {
                      command: {
                        type: "string",

                        description:
                          "The user's visual or navigation question, in natural language.",
                      },
                    },

                    required: ["command"],
                  },
                },
              ],

              input: {
                turn_detection: {
                  vad_threshold: 0.45,

                  min_silence: 700,

                  max_silence: 3000,

                  interrupt_response: true,
                },
              },
            },
          }),
        );

        console.log("[EchoSight] AssemblyAI session.update sent");
      };

      ws.onmessage = handleAssemblyAIMessage;

      ws.onerror = (event) => {
        console.error("[EchoSight] AssemblyAI WebSocket error:", event);

        setVoiceState("error");

        setLastMessage("Voice connection error.");
      };

      ws.onclose = () => {
        setVoiceConnected(false);

        if (mountedRef.current) {
          setVoiceState("idle");
        }
      };
    } catch (error) {
      console.error("[EchoSight] Voice start error:", error);

      setVoiceState("error");

      setLastMessage("Could not connect to the voice assistant.");
    }
  }, [handleAssemblyAIMessage, voiceState]);

  useEffect(() => {
    voiceStateRef.current = voiceState;
  }, [voiceState]);

  /* MICROPHONE */

  useEffect(() => {
    if (!voiceConnected) {
      return;
    }

    let cancelled = false;

    const startMicrophone = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          video: false,
        });

        if (cancelled || !mountedRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        microphoneStreamRef.current = stream;

        const context = new AudioContext({
          sampleRate: 24000,
        });

        audioContextRef.current = context;

        await context.resume();

        if (cancelled) {
          await context.close();
          return;
        }

        const source = context.createMediaStreamSource(stream);

        const processor = context.createScriptProcessor(2048, 1, 1);

        const silentGain = context.createGain();

        silentGain.gain.value = 0;

        microphoneSourceRef.current = source;
        processorRef.current = processor;

        processor.onaudioprocess = (event) => {
          const ws = assemblyWsRef.current;

          if (
            !ws ||
            ws.readyState !== WebSocket.OPEN ||
            voiceStateRef.current === "speaking" ||
            voiceStateRef.current === "thinking"
          ) {
            return;
          }

          const input = event.inputBuffer.getChannelData(0);

          const pcm = float32ToPCM16(input);

          ws.send(
            JSON.stringify({
              type: "input.audio",
              audio: arrayBufferToBase64(pcm),
            }),
          );
        };

        source.connect(processor);

        processor.connect(silentGain);

        silentGain.connect(context.destination);

        setLastMessage("Microphone active. Speak naturally.");
      } catch (error) {
        console.error("[EchoSight] Microphone error:", error);

        if (mountedRef.current) {
          setVoiceState("error");

          setLastMessage("Microphone permission was denied or unavailable.");
        }
      }
    };

    void startMicrophone();

    return () => {
      cancelled = true;

      processorRef.current?.disconnect();
      processorRef.current = null;

      microphoneSourceRef.current?.disconnect();
      microphoneSourceRef.current = null;

      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());

      microphoneStreamRef.current = null;

      const context = audioContextRef.current;

      audioContextRef.current = null;

      if (context) {
        void context.close();
      }
    };
  }, [voiceConnected]);

  /* CLEANUP */

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      stopVoice();

      stopCamera();

      visionWsRef.current?.close();

      if (imageUrlRef.current) {
        URL.revokeObjectURL(imageUrlRef.current);
      }
    };
  }, [stopCamera, stopVoice]);

  /* IMAGE UPLOAD - Important: Upload already returns the CV analysis. We do NOT run analyzeCurrentFrame again. */
  const handleImageFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setLastMessage("Please select an image file.");

        return;
      }

      setIsAnalyzing(true);

      try {
        const url = URL.createObjectURL(file);

        if (imageUrlRef.current) {
          URL.revokeObjectURL(imageUrlRef.current);
        }

        imageUrlRef.current = url;

        setImageUrl(url);

        setFilename(file.name);

        const formData = new FormData();

        formData.append("file", file);

        const response = await fetch(`${BACKEND_URL}/api/vision/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.status}`);
        }

        /*
         * Get the actual CV response.
         */

        const rawResult = (await response.json()) as VisionResult;

        console.log("[EchoSight] Upload vision result:", rawResult);

        // The upload endpoint already performs:
        // image → YOLO → depth → spatial reasoning → navigation
        // Therefore we consume its result directly and DO NOT
        // run /api/voice/analyze a second time.
        applyVisionResult(rawResult);

        setLastMessage(`${file.name} loaded. Scene and navigation updated.`);
      } catch (error) {
        console.error("[EchoSight] Upload error:", error);

        setIsAnalyzing(false);

        setLastMessage("Could not analyze that image.");
      }
    },
    [applyVisionResult],
  );

  /* FILE INPUT */

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (file) {
      void handleImageFile(file);
    }
  };

  /* DRAG AND DROP */

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    setDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      void handleImageFile(file);
    }
  };

  /* IMAGE LOAD */

  const handleImageLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;

    setImageDimensions({
      width: img.naturalWidth,
      height: img.naturalHeight,
    });
  };

  /* CLEAR IMAGE */

  const clearImage = () => {
    if (imageUrlRef.current) {
      URL.revokeObjectURL(imageUrlRef.current);
    }

    imageUrlRef.current = null;

    setImageUrl(null);

    setFilename("");

    setImageDimensions({
      width: 0,
      height: 0,
    });

    setScene("");

    setSceneObjects([]);

    setNavigation(null);

    setLastMessage("Image cleared.");
  };

  /* NAVIGATION TEXT */

  const navigationText =
    navigation?.message ||
    navigation?.action ||
    navigation?.direction ||
    (navigation?.distance !== undefined
      ? `Continue with caution. Distance: ${String(navigation.distance)} m.`
      : "No navigation decision yet.");

  /* VOICE LABEL */

  const voiceLabel =
    voiceState === "listening"
      ? "Listening"
      : voiceState === "speaking"
        ? "Speaking"
        : voiceState === "thinking"
          ? "Thinking"
          : voiceState === "connecting"
            ? "Connecting"
            : voiceState === "error"
              ? "Error"
              : "Offline";

  const appStyle = {
    "--accent": "#49f08a",
  } as CSSProperties;

  /* UI */

  return (
    <main className="app-shell" style={appStyle}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">ES</div>

          <div>
            <h1>EchoSight</h1>

            <span>AI Vision Assistant</span>
          </div>
        </div>

        <div className="status-group">
          <div className={`status-pill ${cameraActive ? "active" : ""}`}>
            <span className="status-dot" />
            Camera {cameraActive ? "Active" : "Off"}
          </div>

          <div className={`status-pill ${voiceConnected ? "active" : ""}`}>
            <span className="status-dot" />
            Voice {voiceConnected ? "Connected" : "Offline"}
          </div>
        </div>
      </header>

      <section className="dashboard-grid">
        <div className="main-column">
          <section className="card camera-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">LIVE VISION</p>

                <h2>Camera View</h2>
              </div>

              <div className="heading-actions">
                <span className={`mini-status ${connected ? "green" : ""}`}>
                  <span className="status-dot" />

                  {connected ? "Vision online" : "Vision offline"}
                </span>

                {cameraActive ? (
                  <button className="btn btn-secondary" onClick={stopCamera}>
                    Stop camera
                  </button>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={() => void startCamera()}
                  >
                    Start camera
                  </button>
                )}
              </div>
            </div>

            <div className="camera-stage">
              {cameraActive ? (
                <video
                  ref={videoRef}
                  className="camera-video"
                  muted
                  playsInline
                />
              ) : imageUrl ? (
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="Uploaded scene"
                  className="camera-image"
                  onLoad={handleImageLoad}
                />
              ) : (
                <div className="camera-empty">
                  <div className="camera-icon">◉</div>

                  <strong>No camera feed</strong>

                  <span>Start the camera or upload an image to begin.</span>
                </div>
              )}

              <div className="camera-overlay top-left">LIVE</div>

              {isAnalyzing && (
                <div className="camera-overlay top-right">ANALYZING</div>
              )}

              <div className="camera-corner corner-tl" />
              <div className="camera-corner corner-tr" />
              <div className="camera-corner corner-bl" />
              <div className="camera-corner corner-br" />
            </div>

            <div className="camera-footer">
              <div className="status-message">
                <span className="green-dot" />

                {lastMessage}
              </div>

              <div className="image-meta">
                {filename || (cameraActive ? "Live camera" : "No image")}

                {imageDimensions.width > 0 &&
                  ` · ${imageDimensions.width}×${imageDimensions.height}`}
              </div>
            </div>
          </section>

          {/* ==================================================
              SCENE ANALYSIS
              ================================================== */}

          <div className="two-column-panels">
            <section className="card scene-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="eyebrow">COMPUTER VISION</p>

                  <h2>Scene Analysis</h2>
                </div>

                <span className="live-badge">AI</span>
              </div>

              <p className="scene-description">
                {scene ||
                  (sceneObjects.length > 0
                    ? `Detected ${sceneObjects.length} object${
                        sceneObjects.length === 1 ? "" : "s"
                      } in the scene.`
                    : "Waiting for a camera frame to analyze.")}
              </p>

              <div className="detection-list">
                {sceneObjects.length > 0 ? (
                  sceneObjects.map((object, index) => {
                    const objectName = object.class || object.label || "Object";

                    const confidence =
                      typeof object.confidence === "number"
                        ? `${Math.round(object.confidence * 100)}%`
                        : "—";

                    const distanceMeters =
                      typeof object.distance_meters === "number"
                        ? `${object.distance_meters.toFixed(2)} m`
                        : object.distance !== undefined
                          ? String(object.distance)
                          : "—";

                    return (
                      <div className="info-row" key={`${objectName}-${index}`}>
                        <div>
                          <strong>{objectName}</strong>

                          <small>
                            {object.direction || "center"} · {distanceMeters}
                          </small>
                        </div>

                        <span>{confidence}</span>
                      </div>
                    );
                  })
                ) : scene ? (
                  <div className="info-row">
                    <span>Scene status</span>

                    <strong>Analyzed</strong>
                  </div>
                ) : (
                  <div className="empty-row">No detections yet.</div>
                )}
              </div>
            </section>

            {/* ==================================================
                NAVIGATION
                ================================================== */}

            <section className="card navigation-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="eyebrow">SPATIAL ENGINE</p>

                  <h2>Navigation</h2>
                </div>

                <span className="direction-badge">NAV</span>
              </div>

              <div className="navigation-main">
                <div className="direction-arrow">→</div>

                <div>
                  <span className="nav-label">RECOMMENDATION</span>

                  <strong>{navigationText}</strong>
                </div>
              </div>

              {navigation?.distance !== undefined && (
                <div className="distance-row">
                  Distance
                  <strong>{String(navigation.distance)} m</strong>
                </div>
              )}
            </section>
          </div>

          {/* ==================================================
              UPLOAD
              ================================================== */}

          <section
            className="card upload-panel"
            onDragOver={(e) => {
              e.preventDefault();

              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="upload-copy">
              <p className="eyebrow">TEST IMAGE</p>

              <h2>Upload a scene</h2>

              <p>Drop an image here or select one from your computer.</p>
            </div>

            <div
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <span className="upload-icon">↑</span>

              <span>Choose image</span>

              <small>JPG, PNG, WEBP</small>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={handleFileInput}
            />

            {imageUrl && (
              <button className="btn btn-secondary" onClick={clearImage}>
                Clear
              </button>
            )}
          </section>
        </div>

        {/* ======================================================
            SIDEBAR
            ====================================================== */}

        <aside className="sidebar">
          <section className="card voice-card">
            <div className="voice-header">
              <div>
                <p className="eyebrow">ASSEMBLYAI</p>

                <h2>Voice Assistant</h2>
              </div>

              <span className={`voice-state ${voiceState}`}>{voiceLabel}</span>
            </div>

            <button
              className={`voice-button ${voiceConnected ? "active" : ""}`}
              onClick={() => (voiceConnected ? stopVoice() : void startVoice())}
            >
              <span className="voice-ring">
                <span className="voice-mic">●</span>
              </span>

              <strong>
                {voiceConnected ? "Stop listening" : "Start voice"}
              </strong>

              <small>
                {voiceConnected ? "Tap to disconnect" : "Talk to EchoSight"}
              </small>
            </button>

            <div className="voice-tip">
              Ask: “What is in front of me?”, “Is it safe to move?”, or “Where
              should I go?”
            </div>
          </section>

          {/* ==================================================
              CONVERSATION
              ================================================== */}

          <section className="card conversation-panel">
            <div className="panel-heading compact">
              <div>
                <p className="eyebrow">LIVE TRANSCRIPT</p>

                <h2>Conversation</h2>
              </div>

              <span className="message-count">{conversation.length}</span>
            </div>

            <div className="conversation-list">
              {conversation.length === 0 ? (
                <div className="conversation-empty">
                  Start the voice assistant to begin a conversation.
                </div>
              ) : (
                conversation.map((message) => (
                  <div key={message.id} className={`message ${message.role}`}>
                    <span className="message-role">
                      {message.role === "user" ? "YOU" : "ECHO"}
                    </span>

                    <p>{message.text}</p>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* ==================================================
              QUICK STATUS
              ================================================== */}

          <section className="card quick-panel">
            <p className="eyebrow">QUICK STATUS</p>

            <div className="stat-grid">
              <div>
                <span>Camera</span>

                <strong>{cameraActive ? "ON" : "OFF"}</strong>
              </div>

              <div>
                <span>Vision WS</span>

                <strong>{connected ? "OK" : "—"}</strong>
              </div>

              <div>
                <span>Voice</span>

                <strong>{voiceConnected ? "OK" : "—"}</strong>
              </div>

              <div>
                <span>Scene</span>

                <strong>
                  {sceneObjects.length > 0 || scene ? "READY" : "—"}
                </strong>
              </div>
            </div>
          </section>
        </aside>
      </section>

      <canvas ref={canvasRef} className="hidden-canvas" />
    </main>
  );
}
