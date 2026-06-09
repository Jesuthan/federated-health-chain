#!/bin/bash
# One-shot setup script for fedlearn-fabric
# Run this once inside GitHub Codespace or any Ubuntu machine with Docker installed.
# Usage: bash setup.sh

set -e  # stop on first error

echo ""
echo "============================================================"
echo "  fedlearn-fabric — Environment Setup"
echo "============================================================"
echo ""

# ── 1. Node.js 20 ─────────────────────────────────────────────
echo "[1/6] Checking Node.js..."
if node --version 2>/dev/null | grep -q "v1[89]\|v2[0-9]"; then
    echo "  Node.js $(node --version) already installed."
else
    echo "  Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "  Node: $(node --version) | npm: $(npm --version)"

# ── 2. IPFS (Kubo) ────────────────────────────────────────────
echo ""
echo "[2/6] Checking IPFS..."
if command -v ipfs &>/dev/null; then
    echo "  IPFS $(ipfs version) already installed."
else
    echo "  Installing IPFS Kubo v0.29.0..."
    # Extract in /tmp to avoid NTFS permission issues on /mnt drives
    cd /tmp
    wget -q https://dist.ipfs.tech/kubo/v0.29.0/kubo_v0.29.0_linux-amd64.tar.gz
    tar --warning=no-timestamp -xzf kubo_v0.29.0_linux-amd64.tar.gz
    sudo bash kubo/install.sh
    rm -rf kubo kubo_v0.29.0_linux-amd64.tar.gz
    cd - > /dev/null
    echo "  IPFS installed."
fi

if [ ! -d "$HOME/.ipfs" ]; then
    echo "  Initialising IPFS repo..."
    ipfs init
else
    echo "  IPFS repo already initialised."
fi

# ── 3. Hyperledger Fabric binaries + fabric-samples ───────────
echo ""
echo "[3/6] Checking Hyperledger Fabric..."
if command -v peer &>/dev/null; then
    echo "  Fabric peer already in PATH."
else
    echo "  Downloading Fabric 2.5.0 binaries + fabric-samples (~500MB)..."
    cd "$HOME"
    curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.0 1.5.5

    # Add to PATH if not already there
    if ! grep -q 'fabric-samples/bin' "$HOME/.bashrc"; then
        echo 'export PATH=$HOME/fabric-samples/bin:$PATH' >> "$HOME/.bashrc"
    fi
    export PATH="$HOME/fabric-samples/bin:$PATH"
    cd -
fi
echo "  Fabric peer: $(peer version 2>/dev/null | head -1 || echo 'check PATH')"

# ── 4. npm install — server ───────────────────────────────────
echo ""
echo "[4/6] Installing Node packages (server)..."
npm install --prefix "$(dirname "$0")/server" --silent
echo "  Done."

# ── 5. npm install — chaincode ────────────────────────────────
echo ""
echo "[5/6] Installing Node packages (chaincode)..."
npm install --prefix "$(dirname "$0")/chaincode/modelregistry" --silent
echo "  Done."

# ── 6. Python venv + packages ─────────────────────────────────
echo ""
echo "[6/6] Setting up Python virtual environment..."

VENV_DIR="$(dirname "$0")/venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
else
    echo "  venv already exists."
fi

# Activate and install
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet torch requests ipfshttpclient
echo "  Python packages installed inside venv."
echo "  Python: $(python --version)"
deactivate

# ── 7. direnv (auto-activate venv on cd) ──────────────────────
echo ""
echo "[+] Setting up direnv (auto venv activation)..."

if ! command -v direnv &>/dev/null; then
    sudo apt-get install -y direnv -qq
fi

# Hook direnv into bash if not already done
if ! grep -q 'direnv hook bash' "$HOME/.bashrc"; then
    echo 'eval "$(direnv hook bash)"' >> "$HOME/.bashrc"
fi

# Allow the .envrc in this project
direnv allow "$(dirname "$0")"
echo "  direnv configured. venv will activate automatically when you cd into this project."

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo ""
echo "  Terminal 1:  ipfs daemon"
echo ""
echo "  Terminal 2:  cd ~/fabric-samples/test-network"
echo "               ./network.sh up createChannel -ca"
echo "               ./network.sh deployCC \\"
echo "                 -ccn modelregistry \\"
echo "                 -ccp ~/fedlearn-fabric/chaincode/modelregistry \\"
echo "                 -ccl javascript"
echo ""
echo "  Terminal 3:  cd ~/fedlearn-fabric/server"
echo "               node enrollAdmin.js && node registerUser.js"
echo "               node server.js"
echo ""
echo "  Terminal 4:  cd ~/fedlearn-fabric"
echo "               source venv/bin/activate"
echo "               cd client"
echo "               python fl_client.py --sender Client1 --model covid --round 1"
echo "============================================================"
echo ""
