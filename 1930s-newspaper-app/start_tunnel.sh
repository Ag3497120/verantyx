#!/bin/bash

# ==========================================
# Verantyx Hybrid Tunneling (Plan A)
# ==========================================

echo "🧠 Booting Verantyx 1930s API (FastAPI)..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 3

echo "🌐 Checking Cloudflare Tunnel (cloudflared)..."
if ! command -v cloudflared &> /dev/null; then
    echo "cloudflared not found. Please install it with: brew install cloudflare/cloudflare/cloudflared"
    kill $FASTAPI_PID
    exit 1
fi

echo "🚀 Opening Tunnel to the World and intercepting the URL..."
# Run cloudflared in the background and pipe output to a log file
cloudflared tunnel --protocol http2 --url http://localhost:8000 2>&1 | tee tunnel.log &
TUNNEL_PID=$!

# Continuously read the log until the URL is generated
echo "🔍 Waiting for Cloudflare to assign a URL..."
TUNNEL_URL=""
while [ -z "$TUNNEL_URL" ]; do
    sleep 1
    # Extract the trycloudflare URL using regex
    TUNNEL_URL=$(grep -o 'https://[-a-zA-Z0-9]*\.trycloudflare\.com' tunnel.log | head -n 1)
done

echo "✅ Tunnel URL successfully extracted: $TUNNEL_URL"

# ==========================================
# 📝 Inject URL directly into index.html
# ==========================================
echo "🔄 Automatically updating index.html with the new URL..."

python3 -c "
import re
with open('public/index.html', 'r') as f:
    content = f.read()
# Replace the backendUrl ternary operator line with the new URL
new_content = re.sub(
    r'const backendUrl = window\.location\.protocol === \"file:\" \? \".*?\" : \"\";',
    'const backendUrl = window.location.protocol === \"file:\" ? \"$TUNNEL_URL\" : \"\";',
    content
)
with open('public/index.html', 'w') as f:
    f.write(new_content)
"

echo "✨ index.html has been successfully updated!"

# ==========================================
# 🚀 Auto-Deploy to Cloudflare Pages (Git Push)
# ==========================================
echo "📦 Committing and pushing the updated site to GitHub..."
git add public/index.html
git commit -m "Auto-update: Tunnel URL via DDNS Script"
git push

echo "✅ Push complete! Cloudflare Pages will now automatically deploy the latest URL."

# Wait for cloudflared to finish (it runs forever until Ctrl+C)
wait $TUNNEL_PID

echo "Shutting down..."
kill $FASTAPI_PID
rm -f tunnel.log
