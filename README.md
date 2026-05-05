# fedlearn-fabric

Federated Learning system using **Hyperledger Fabric** (blockchain) and **IPFS** for privacy-preserving, decentralised model training across hospitals.

Two models supported: **COVID-19** (chest X-ray, 3 classes) and **Skin Cancer** (lesion images, 7 classes).

---

## System Overview

```
Hospital / Client
  ├─ pulls latest global model from IPFS  (--pull-global)
  ├─ trains CNN on local data             (data never leaves)
  ├─ computes weight delta
  ├─ clips + adds noise                   (differential privacy)
  ├─ uploads delta ──────────────────►   IPFS  ->  returns CID
  └─ POST CID ───────────────────────►   REST API  ->  Fabric  ->  on-chain record

Coordinator (auto-triggered when MIN_CLIENTS threshold reached)
  ├─ fetches all delta CIDs from blockchain
  ├─ downloads deltas from IPFS
  ├─ FedAvg: equal-weight average across all clients
  ├─ applies averaged delta to base model
  ├─ uploads new global model ─────────► IPFS  ->  returns CID
  └─ records global model CID ─────────► Fabric blockchain
```

---

## Project Structure

```
fedlearn-fabric/
├── chaincode/modelregistry/
│   └── index.js          <- Fabric smart contract
├── server/
│   ├── server.js         <- Express REST API + auto-aggregation trigger
│   ├── aggregator.py     <- FedAvg coordinator (spawned automatically)
│   ├── enrollAdmin.js
│   ├── registerUser.js
│   └── package.json
├── client/
│   ├── fl_client.py      <- Hospital FL client
│   └── requirements.txt
├── models/
│   ├── inspect_hf_models.py   <- Downloads real weights from HuggingFace
│   ├── covid_model.pth        <- Real pretrained COVID-19 weights (from HF)
│   └── skin_model.pth         <- Real pretrained Skin Cancer weights (from HF)
├── public/
│   └── index.html        <- Live dashboard (served at http://localhost:3000)
└── README.md
```

---

## Prerequisites

| Tool | Version | Required for |
|---|---|---|
| Docker | Latest | Fabric peer/orderer containers |
| Node.js | 18+ | Server + chaincode |
| Python | 3.9+ | FL client + aggregator |
| Go | 1.21+ | Fabric CLI tools |

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
```

---

## Step 3 — Install dependencies

```bash
# Python (client + aggregator)
cd client && pip install -r requirements.txt

# Chaincode
cd ../chaincode/modelregistry && npm install

# Server
cd ../../server && npm install
```

---

## Step 4 — Download real model weights

```bash
python models/inspect_hf_models.py
```

Downloads `covid_model.pth` and `skin_model.pth` from HuggingFace and prints architecture details.

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
MIN_CLIENTS=2 node server.js
```

**Terminal 4 — Hospital clients**
```bash
cd ~/fedlearn-fabric/client

# Round 1 (no global model yet)
python fl_client.py --sender Hospital1 --model covid --round 1
python fl_client.py --sender Hospital2 --model covid --round 1
# FedAvg triggers automatically once MIN_CLIENTS=2 updates arrive

# Round 2+ (pull the new global model first)
python fl_client.py --sender Hospital1 --model covid --round 2 --pull-global
python fl_client.py --sender Hospital2 --model covid --round 2 --pull-global
```

**Dashboard** — open `http://localhost:3000` in a browser to see live status.

---

## FL Client Arguments

| Argument | Default | Description |
|---|---|---|
| `--sender` | required | Hospital identifier e.g. Hospital1 |
| `--model` | required | `covid` or `skin` |
| `--round` | required | FL round number |
| `--clip` | `1.0` | DP gradient clip value |
| `--noise` | `0.1` | DP Gaussian noise scale |
| `--server` | `http://localhost:3000` | REST server URL |
| `--pull-global` | off | Pull latest global model before training (use from round 2+) |

---

## REST API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/updates` | All hospital updates |
| POST | `/api/updates` | Submit new update (triggers FedAvg when threshold reached) |
| GET | `/api/updates/round/:round` | Filter by FL round |
| GET | `/api/updates/sender/:sender` | Filter by hospital |
| POST | `/api/global-model` | Record aggregated global model (called by aggregator) |
| GET | `/api/global-model/:modelType/latest` | Get latest global model CID |
| GET | `/api/global-model/:modelType/round/:round` | Get global model for a specific round |

---

## Model Architectures

### COVID-19 (CovidCNN)
- Input: 224x224 grayscale (1 channel)
- 4 blocks of Conv2d -> ReLU -> MaxPool2d(2): 224->112->56->28->14
- Flatten: 256 * 14 * 14 = 50,176
- FC: Linear(50176, 128) -> Linear(128, 3)
- Output: raw logits for 3 classes
- State dict keys: `layers.0/3/6/9`, `fc1`, `fc2`

### Skin Cancer (SkinCNN)
- Input: 224x224 grayscale (1 channel)
- 2-layer CNN converted from Keras Sequential
- Conv2d(1,32) -> MaxPool(2) -> Conv2d(32,64) -> MaxPool(2): 224->112->56
- Flatten: 64 * 56 * 56 = 200,704
- FC: Linear(200704, 128) -> Linear(128, 7)
- Output: Softmax probabilities for 7 classes
- State dict keys: `conv1`, `conv2`, `fc1`, `fc2`

---

## FedAvg Aggregation

Triggered automatically by `server.js` when `MIN_CLIENTS` hospital updates arrive for the same round + model type.

```bash
# Manual trigger (if needed)
python server/aggregator.py --round 1 --model covid
python server/aggregator.py --round 1 --model skin
```

Set threshold via environment variable:
```bash
MIN_CLIENTS=3 node server.js
```

---

## Shutdown

```bash
# Stop Fabric
cd ~/fabric-samples/test-network && ./network.sh down

# Stop IPFS daemon
Ctrl+C in Terminal 1
```
