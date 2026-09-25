# EchoSight

EchoSight is a computer-vision accessibility assistant built with a Next.js frontend and FastAPI backend. The backend runs YOLO11 and Depth Anything V2, then exposes scene analysis, navigation, and AssemblyAI voice-agent endpoints.

## Railway deployment

Deploy the `ECHOSIGHT` directory as the Railway service root.

1. Create a Railway service from this repository with the root directory set to `ECHOSIGHT`.
2. Railway will use `railway.toml` to install `backend/requirements.txt`, start FastAPI, and check `/health`.
3. Add the secret environment variable `ASSEMBLYAI_API_KEY` in Railway. Do not commit `.env` or put this key in the frontend.
4. Set `CORS_ORIGINS` to the deployed frontend origin, for example `https://echosight.vercel.app`.
5. Deploy the frontend separately, set `NEXT_PUBLIC_BACKEND_URL` to the Railway backend URL, and redeploy it so the value is included at build time.

The backend selects CUDA, Apple Metal, or CPU automatically. Railway normally uses CPU, so the first deployment should be treated as a latency test. The `/health` endpoint confirms that the process is running; uploading an image or opening the camera confirms that model inference and the WebSocket are working.

## Local development

Backend:

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

For local development, the frontend defaults to `http://localhost:8000`. To point it at another backend, create `frontend/.env.local` with `NEXT_PUBLIC_BACKEND_URL`.
