#!/bin/bash
set -e
cd /home/aditya/MLOps_Project

echo "=== Step 0: Start Ollama port forwarder (0.0.0.0:11436 -> 127.0.0.1:11434) ==="
pkill -f "ollama_forward.py" 2>/dev/null || true
nohup python3 /tmp/ollama_forward.py > /tmp/ollama_forward.log 2>&1 &
sleep 1
if ! ss -tlnp | grep -q "11436"; then
  cat > /tmp/ollama_forward.py << 'PYEOF'
import socket, threading

def pipe(src, dst):
    try:
        while True:
            data = src.recv(4096)
            if not data: break
            dst.sendall(data)
    except: pass
    finally:
        for s in (src, dst):
            try: s.close()
            except: pass

def handle(client):
    try:
        server = socket.create_connection(('127.0.0.1', 11434))
        for a, b in [(client, server), (server, client)]:
            threading.Thread(target=pipe, args=(a, b), daemon=True).start()
    except: client.close()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 11436))
sock.listen(100)
print('Ollama forwarder: 0.0.0.0:11436 -> 127.0.0.1:11434', flush=True)
while True:
    c, _ = sock.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
PYEOF
  nohup python3 /tmp/ollama_forward.py > /tmp/ollama_forward.log 2>&1 &
  sleep 1
fi
echo "Forwarder running: $(ss -tlnp | grep 11436)"

echo "=== Step 1: Build Docker image ==="
docker build -t researchcopilot-backend:latest .

echo "=== Step 2: Import image into k3s ==="
docker save researchcopilot-backend:latest | sudo k3s ctr images import -

echo "=== Step 3: Apply k8s manifests ==="
sudo k3s kubectl apply -f k8s/namespace.yaml
sudo k3s kubectl apply -f k8s/configmap.yaml
sudo k3s kubectl apply -f k8s/deployment.yaml
sudo k3s kubectl apply -f k8s/service.yaml
sudo k3s kubectl apply -f k8s/hpa.yaml

echo "=== Step 4: Wait for pods to be ready ==="
sudo k3s kubectl rollout status deployment/researchcopilot-backend -n researchcopilot --timeout=120s

echo ""
echo "=== Status ==="
sudo k3s kubectl get pods -n researchcopilot
sudo k3s kubectl get hpa -n researchcopilot
echo ""
echo "Backend now available at: http://10.6.0.69:30800"
echo "Health check: curl http://10.6.0.69:30800/health"
