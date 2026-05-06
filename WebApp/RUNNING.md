# Running BioXplorer

## First Time Setup

### Backend
```bash
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp/api_server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install peft bitsandbytes
```

### Frontend
```bash
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp
npm install
```

---

## Starting the App

### 1. Start the backend (Terminal 1)
```bash
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp/api_server
source .venv/bin/activate
python -m uvicorn app:app --host 0.0.0.0 --port 5001
```
Wait until you see: `LLaMA fine-tuned model loaded successfully`

Check it is running:
```bash
curl http://localhost:5001/
# expected: {"message": "API is running"}
```

### 2. Start the frontend (Terminal 2)
```bash
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp
npm start
```
Opens automatically at http://localhost:3000

Navigate to http://localhost:3000/upload to use the app.

---

## After Making Changes

### Changed a frontend file (anything in `src/`)
**Nothing to do.** `npm start` watches for file changes and hot-reloads the browser automatically.

### Changed the backend (`api_server/app.py`)
Restart uvicorn:
```bash
# If running in foreground: Ctrl+C, then start again
python -m uvicorn app:app --host 0.0.0.0 --port 5001

# If running in background:
pkill -f "uvicorn app:app"
bash restart_api.sh
tail -f api_server.log   # watch for "LLaMA fine-tuned model loaded successfully"
```
Note: the model takes a few minutes to reload into GPU memory each restart.

### Added a new Python package
```bash
source .venv/bin/activate
pip install <package>
# then restart uvicorn as above
```

### Added a new npm package
```bash
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp
npm install <package>
# frontend hot-reloads automatically
```

---

## Stopping the App

```bash
# Stop backend
pkill -f "uvicorn app:app"

# Stop frontend
# Ctrl+C in the npm start terminal
```

---

## Deploying to Vercel (after frontend changes)

```bash
cd /p/realai/BioXplorer/LLama-BioXplorer/WebApp
npm run build
vercel --prod
```
Or push to GitHub if Vercel is connected to the repo — it will redeploy automatically.

The backend is never deployed to Vercel. It always runs on the GPU machine and is exposed via the Cloudflare tunnel.

---

## Cloudflare Tunnel (if backend needs to be publicly accessible)

```bash
cd /p/realai/BioXplorer/LLama-BioXplorer
./cloudflared tunnel --url http://localhost:5001
```
Copy the printed `https://*.trycloudflare.com` URL and set it in `WebApp/.env`:
```
REACT_APP_API_URL=https://<your-tunnel>.trycloudflare.com
```
Then restart `npm start` so it picks up the new env value.
