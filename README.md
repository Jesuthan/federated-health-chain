# fedlearn-fabric

Federated Learning system using **Hyperledger Fabric** (blockchain) and **IPFS** for privacy-preserving, decentralised model training.

## System Overview

```
Hospital/Client
  ├─ trains CNN on local data        (data never leaves)
  ├─ computes weight delta
  ├─ clips + adds noise              (differential privacy)
  ├─ uploads delta ──────────────►  IPFS  →  returns CID
  └─ POST CID ───────────────────►  REST API  →  Fabric  →  on-chain record
```

---

## Prerequisites

| Tool | Version | Required for |
|---|---|---|
| Docker | Latest | Fabric peer/orderer containers |
| Node.js | 18+ | Server + chaincode |
| Python | 3.9+ | FL client |
| Go | 1.21+ | Fabric CLI tools |
| Git | Any | Cloning repos |

---

## Step 1 — Install IPFS

```bash
wget https://dist.ipfs.tech/kubo/v0.29.0/kubo_v0.29.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.29.0_linux-amd64.tar.gz
cd kubo && sudo bash install.sh
ipfs init
```

---

## Step 2 — Install Hyperledger Fabric

```bash
cd ~
curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.0 1.5.5
echo 'export PATH=$HOME/fabric-samples/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
peer version
```

---

## Step 3 — Install dependencies

```bash
# Chaincode
cd chaincode/modelregistry && npm install

# Server
cd ../../server && npm install

# Python client
cd ../client && pip install -r requirements.txt
```

---

## Step 4 — Generate test weights

```bash
cd models && python generate_dummy_models.py
```

---

## Running (4 terminals)

**Terminal 1 — IPFS daemon**
```bash
ipfs daemon
```

**Terminal 2 — Fabric network + chaincode**
```bash
cd ~/fabric-samples/test-network
./network.sh up createChannel -ca
./network.sh deployCC \
  -ccn modelregistry \
  -ccp ~/fedlearn-fabric/chaincode/modelregistry \
  -ccl javascript
```

**Terminal 3 — REST server**
```bash
cd ~/fedlearn-fabric/server
node enrollAdmin.js
node registerUser.js
node server.js
```

**Terminal 4 — FL clients**
```bash
cd ~/fedlearn-fabric/client
python fl_client.py --sender Client1 --model covid --round 1 --clip 1 --noise 0.1
python fl_client.py --sender Client2 --model covid --round 1
python fl_client.py --sender Client3 --model skin  --round 1
```

---

## Verify it works

```bash
curl http://localhost:3000/api/updates
curl http://localhost:3000/api/updates/round/1
curl http://localhost:3000/api/updates/sender/Client1
```

---

## REST API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/updates` | All model updates |
| POST | `/api/updates` | Submit new update |
| GET | `/api/updates/round/:round` | Filter by FL round |
| GET | `/api/updates/sender/:sender` | Filter by client |

**POST body example:**
```json
{
  "sender": "Client1",
  "modelType": "covid",
  "round": 1,
  "ipfsCID": "QmABC...",
  "clipValue": 1.0,
  "noiseScale": 0.1
}
```

**On-chain record example:**
```json
{
  "docType": "modelUpdate",
  "updateId": "update_Client1_round1_a1b2c3d4",
  "sender": "Client1",
  "modelType": "covid",
  "round": 1,
  "ipfsCID": "QmABC...",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "clipValue": 1.0,
  "noiseScale": 0.1
}
```

---

## CNN Architecture Comparison

| Arch | Layers | FC size | Speed | Accuracy |
|---|---|---|---|---|
| CNN2 | 2 conv | 64×16×16 → 128 | Fastest | Lowest |
| CNN4 | 4 conv | 128×16×16 → 256 | Balanced | Medium ← default |
| CNN6 | 6 conv | 256×16×16 → 512 | Slowest | Highest |

All architectures: **1-channel 64×64 input, 3 output classes**.

---

## FL Client Arguments

| Argument | Default | Description |
|---|---|---|
| `--sender` | required | Client identifier (e.g. Client1) |
| `--model` | required | `covid` or `skin` |
| `--arch` | `cnn4` | CNN architecture: `cnn2`, `cnn4`, `cnn6` |
| `--round` | required | FL round number |
| `--clip` | `1.0` | DP gradient clip value |
| `--noise` | `0.1` | DP Gaussian noise scale |
| `--server` | `http://localhost:3000` | REST server URL |

---

## Shutdown

```bash
# Stop Fabric network
cd ~/fabric-samples/test-network && ./network.sh down

# Stop IPFS daemon
Ctrl+C in Terminal 1
```

## Restarting Codespace

```bash
# Terminal 1
ipfs daemon

# Terminal 2
cd ~/fabric-samples/test-network
./network.sh up createChannel -ca

# Terminal 3
cd ~/fedlearn-fabric/server && node server.js
```

---

## What is NOT built yet

| Feature | Notes |
|---|---|
| **FedAvg aggregation** | A coordinator would fetch all CIDs from IPFS, average deltas, push new global model |
| **Real training loop** | Replace `simulate_local_training()` with real PyTorch training on your dataset |
| **Real model weights** | Replace dummy `.pth` files with pretrained weights |
| **API authentication** | REST server has no auth — add JWT or API keys for production |
