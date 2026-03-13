# FedHealth Chain — Setup & Run Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Your PC — Daily Startup](#your-pc--daily-startup)
3. [Fresh Setup — New PC or Friend's Laptop](#fresh-setup--new-pc-or-friends-laptop)
4. [Running FL Clients](#running-fl-clients)
5. [Dashboard](#dashboard)
6. [Troubleshooting](#troubleshooting)

---

## Project Overview

FedHealth Chain is a Federated Learning system backed by:
- **Hyperledger Fabric 2.5** — blockchain for storing FL update records
- **IPFS (Kubo v0.29.0)** — decentralised storage for model weight deltas
- **Python FL Client** — differential privacy (gradient clipping + Gaussian noise)
- **Express REST API** — bridge between Python clients and Fabric
- **Web Dashboard** — live view of all blockchain records at `http://localhost:3000`

---

## Your PC — Daily Startup

> Do this every time you restart your PC.

### Step 1 — Start Docker Desktop
- Open **Docker Desktop** from the Start menu
- Wait until the bottom left shows **"Engine running"**

### Step 2 — Open Ubuntu WSL
- Press `Win` key → search **Ubuntu** → open it

### Step 3 — Check if Fabric is already running
```bash
docker ps
```
- If you see `peer0.org1`, `peer0.org2`, `orderer` containers → **skip to Step 5**
- If the list is empty → continue to Step 4

### Step 4 — Start Fabric Network (only if containers are empty)
```bash
cd ~/fedlearn-fabric/fabric-samples/test-network
./network.sh up
export FABRIC_CFG_PATH=$HOME/fedlearn-fabric/fabric-samples/config/
./network.sh deployCC -ccn modelregistry -ccp ~/fedlearn-fabric/chaincode/modelregistry -ccl javascript -ccv 1.1 -ccs 2
bash scripts/setAnchorPeer.sh 1 mychannel
bash scripts/setAnchorPeer.sh 2 mychannel
cd ~/fedlearn-fabric/server
node enrollAdmin.js && node registerUser.js
```

### Step 5 — Start IPFS
Open a **new Ubuntu terminal** and run:
```bash
ipfs daemon
```
Leave this terminal open.

### Step 6 — Start REST Server
Open another **new Ubuntu terminal** and run:
```bash
cd ~/fedlearn-fabric/server
node server.js
```
Leave this terminal open.

### Step 7 — Open Dashboard
Open your browser and go to:
```
http://localhost:3000
```

---

## Fresh Setup — New PC or Friend's Laptop

> Do this once on any new machine.

### Prerequisites

**1 — Install WSL2 + Ubuntu**

Open PowerShell as Administrator:
```powershell
wsl --install -d Ubuntu
```
Restart the PC when prompted. Then open Ubuntu from the Start menu and create a username/password.

**2 — Install Docker Desktop**
- Download from [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- Install and launch it
- Go to **Settings → Resources → WSL Integration**
- Enable the toggle for **Ubuntu**
- Click **Apply & Restart**

---

### Clone the Project

Open Ubuntu WSL and run:
```bash
cd ~
git clone https://github.com/Jesuthan/federated-health-chain.git fedlearn-fabric
cd fedlearn-fabric
```

---

### Install Dependencies

Run each block one at a time:

**Node.js 20**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install nodejs -y
node --version
```

**npm packages**
```bash
cd ~/fedlearn-fabric
npm install --prefix server
npm install --prefix chaincode/modelregistry
```

**Python + venv**
```bash
sudo apt install python3.12-venv python3-pip jq -y
cd ~/fedlearn-fabric
python3 -m venv venv
source venv/bin/activate
pip install torch requests ipfshttpclient numpy
```

**IPFS**
```bash
cd /tmp
wget https://dist.ipfs.tech/kubo/v0.29.0/kubo_v0.29.0_linux-amd64.tar.gz
tar --warning=no-timestamp -xzf kubo_v0.29.0_linux-amd64.tar.gz
sudo cp kubo/ipfs /usr/local/bin/
ipfs init
```

**Hyperledger Fabric binaries + Docker images**
```bash
cd ~/fedlearn-fabric
curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.0 1.5.7
```
This downloads ~1GB of Docker images. Wait until it finishes.

**Docker permissions**
```bash
sudo usermod -aG docker $USER
newgrp docker
docker ps
```

---

### First-Time Fabric Setup

```bash
cd ~/fedlearn-fabric/fabric-samples/test-network
./network.sh up createChannel -c mychannel -ca
export FABRIC_CFG_PATH=$HOME/fedlearn-fabric/fabric-samples/config/
bash scripts/setAnchorPeer.sh 1 mychannel
bash scripts/setAnchorPeer.sh 2 mychannel
./network.sh deployCC -ccn modelregistry -ccp ~/fedlearn-fabric/chaincode/modelregistry -ccl javascript -ccv 1.1 -ccs 2
cd ~/fedlearn-fabric/server
node enrollAdmin.js && node registerUser.js
```

---

### Generate Dummy Models

```bash
cd ~/fedlearn-fabric
source venv/bin/activate
python models/generate_dummy_models.py
```

---

After this, follow the **Daily Startup** steps (Step 5 onwards) to start IPFS, the server, and open the dashboard.

---

## Running FL Clients

Activate the venv first:
```bash
cd ~/fedlearn-fabric
source venv/bin/activate
```

Run a client:
```bash
python client/fl_client.py --sender <hospital_name> --model <covid|skin> --round <number>
```

**Examples:**
```bash
# Hospital 1 submits covid model update for round 1
python client/fl_client.py --sender hospital1 --model covid --round 1

# Hospital 2 submits skin model update for round 1
python client/fl_client.py --sender hospital2 --model skin --round 1

# Hospital 3 submits covid model update for round 2
python client/fl_client.py --sender hospital3 --model covid --round 2
```

**Optional arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--arch` | `cnn4` | CNN architecture: `cnn2`, `cnn4`, `cnn6` |
| `--clip` | `1.0` | Differential privacy gradient clip norm |
| `--noise` | `0.1` | Differential privacy Gaussian noise scale |
| `--server` | `http://localhost:3000` | REST API URL |

---

## Dashboard

Open browser → **http://localhost:3000**

The dashboard shows:
- **Status** — Fabric blockchain, IPFS node, REST API (online/offline)
- **Stats** — total updates, number of hospitals, FL rounds, model types
- **Rounds Summary** — per-round breakdown of hospital submissions
- **Blockchain Records** — full table of all FL updates with IPFS CIDs and timestamps

The dashboard auto-refreshes every 10 seconds.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `docker ps` — permission denied | Run `sudo usermod -aG docker $USER && newgrp docker` |
| `npm install` — EPERM error | You are on `/mnt/d/` (Windows drive). Move to `~/` (Linux home) |
| IPFS version mismatch error | Already fixed in code — uses HTTP API directly |
| Fabric endorsement mismatch | Already fixed in chaincode — uses `getTxTimestamp()` |
| `node enrollAdmin.js` — connection profile not found | Check path has `fedlearn-fabric/fabric-samples/...` |
| Server returns 500 | Check server terminal for error. Restart with `node server.js` |
| Dashboard shows offline | Make sure server is running on port 3000 |
