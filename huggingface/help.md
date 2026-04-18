# ResearchCopilot — Deployment Guide

**Architecture:**
- **Frontend** → Hugging Face Spaces (Gradio 6.12.0, free tier)
- **Backend** → Your GPU machine (FastAPI + Ollama + ChromaDB + DynamoDB)

---

## Part 1 — Expose Your GPU Backend Publicly

The HF Space needs to reach your backend over the internet. Use **ngrok** (easiest).

### Install ngrok (no sudo required)

```bash
# Download the binary directly into your home directory
cd ~
curl -Lo ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
unzip ngrok.zip
chmod +x ngrok
mkdir -p ~/.local/bin
mv ngrok ~/.local/bin/ngrok

# Add to PATH if not already there (add this line to ~/.bashrc too)
export PATH="$HOME/.local/bin:$PATH"

# Verify
ngrok version
```

### Authenticate

1. Sign up at https://ngrok.com (free)
2. Copy your authtoken from the dashboard
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

### Start your backend

```bash
cd /home/aditya/MLOps_Project
docker compose up -d                          # ChromaDB, DynamoDB, MLflow
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Tunnel port 8000

```bash
ngrok http 8000
```

Output will show:
```
Forwarding   https://abcd1234.ngrok-free.app -> http://localhost:8000
```

Copy that `https://abcd1234.ngrok-free.app` — this is your `BACKEND_URL`.

> Free ngrok URLs change every restart. For a stable URL use a paid ngrok plan.

---

## Part 2 — Deploy to Hugging Face Spaces

### Step 1 — Create a Space

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Set:
   - **SDK**: Gradio
   - **SDK version**: 6.12.0
   - **Visibility**: Public or Private

### Step 2 — Upload files

Upload the 4 files inside the `huggingface/` folder:
```
app.py
requirements.txt
README.md
help.md
```

**Via Git (recommended):**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/researchcopilot
cp /home/aditya/MLOps_Project/huggingface/* researchcopilot/
cd researchcopilot
git add .
git commit -m "Initial deploy"
git push
```

**Or drag-and-drop** via the HF web UI (Files tab → Upload files).

### Step 3 — Add BACKEND_URL secret

1. Go to your Space → **Settings** tab
2. Scroll to **Repository secrets**
3. Click **New secret**
   - Name: `BACKEND_URL`
   - Value: `https://abcd1234.ngrok-free.app`
4. Save → **Factory reboot** the Space

---

## Part 3 — Verify

1. Open your Space URL
2. The header should show ✅ **Backend Online**
3. Ask a question in Paper Discovery

**If Backend shows Offline:**
- Confirm `uvicorn` is running on your GPU machine
- Confirm `ngrok http 8000` is active
- Check `BACKEND_URL` secret has no trailing slash

---

## Part 4 — Keep Backend Running

```bash
cd /home/aditya/MLOps_Project
docker compose up -d
nohup python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
nohup ngrok http 8000 > /tmp/ngrok.log 2>&1 &
```

Get the current ngrok URL anytime:
```bash
curl -s http://localhost:4040/api/tunnels | python3 -c \
  "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

Update the HF Space secret with the new URL whenever ngrok restarts.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Backend Offline | Check ngrok is running; update BACKEND_URL if URL changed |
| CORS error | backend/main.py already has `allow_origins=["*"]` — no action needed |
| Timeout on search | Normal for slow arXiv — 60s timeout is set |
| BACKEND_URL not set warning | Add it in Space Settings → Secrets and reboot |

---

## File Structure

```
huggingface/              ← upload all 4 files to HF Space
├── app.py                ← complete Gradio app (all tabs in one file)
├── requirements.txt      ← only `requests` (Gradio pre-installed by HF)
├── README.md             ← HF Spaces config (frontmatter required)
└── help.md               ← this file

GPU Machine (stays local, not uploaded):
├── backend/              ← FastAPI + RAG pipeline
├── docker-compose.yml    ← ChromaDB, DynamoDB, MLflow
└── .env
```


```curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```


```http://127.0.0.1:4040```                       