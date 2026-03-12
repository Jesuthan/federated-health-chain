# Federated Learning Project — Summary

## What Was Built

A **Federated Learning (FL) system** that combines:
- **Hyperledger Fabric** — blockchain to permanently record model updates
- **IPFS** — decentralised file storage for model weight deltas
- **Express REST API** — bridge between Python clients and Fabric
- **Python FL Client** — trains locally, applies differential privacy, submits update

---

## Project Structure

```
fedlearn-fabric/
├── chaincode/
│   └── modelregistry/
│       ├── index.js          ← Fabric smart contract (JavaScript)
│       └── package.json
├── server/
│   ├── enrollAdmin.js        ← Enroll Fabric CA admin identity
│   ├── registerUser.js       ← Register appUser identity
│   ├── server.js             ← Express REST API (port 3000)
│   └── package.json
├── client/
│   ├── fl_client.py          ← Python FL client
│   └── requirements.txt
├── models/
│   ├── generate_dummy_models.py  ← Creates test .pth weight files
│   ├── covid_model.pth           ← (add real weights here)
│   └── skin_model.pth            ← (add real weights here)
├── .gitignore
├── README.md
└── PROJECT_SUMMARY.md
```

---

## File-by-File Breakdown

### `chaincode/modelregistry/index.js`
Fabric smart contract. Stores and queries model update records on-chain.

Functions:
| Function | What it does |
|---|---|
| `storeUpdate()` | Save a model update record (sender, CID, round, DP params) |
| `getUpdate()` | Fetch one record by ID |
| `getAllUpdates()` | Return every stored record |
| `queryByRound()` | Filter records by FL round number |
| `queryBySender()` | Filter records by client name |

Each on-chain record looks like:
```json
{
  "sender": "Client1",
  "modelType": "covid",
  "round": 1,
  "ipfsCID": "QmABC...",
  "timestamp": "2024-...",
  "clipValue": 1.0,
  "noiseScale": 0.1
}
```
> Note: actual model weights live on IPFS, not on chain. Only the address (CID) is stored.

---

### `server/enrollAdmin.js`
Connects to Fabric CA and enrolls the `admin` identity into `server/wallet/`.
Run once before anything else.

### `server/registerUser.js`
Uses admin identity to register `appUser` into `server/wallet/`.
The REST server signs all transactions as `appUser`.

### `server/server.js`
Express REST API. Translates HTTP calls into Fabric transactions.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/updates` | All model updates |
| POST | `/api/updates` | Submit new update |
| GET | `/api/updates/round/:round` | Filter by FL round |
| GET | `/api/updates/sender/:sender` | Filter by client |

---

### `client/fl_client.py`
Python client each participant runs. Pipeline:

```
[1] Load model weights (.pth file)
[2] Simulate local training
[3] Compute weight delta (updated - original)
[4] Clip gradients        ← differential privacy
[5] Add Gaussian noise    ← differential privacy
[6] Upload delta to IPFS  → get CID
[7] POST CID to REST server → stored on Fabric blockchain
```

Run example:
```bash
python fl_client.py --sender Client1 --model covid --round 1 --clip 1 --noise 0.1
```

Models supported: `covid`, `skin`

---

### `models/generate_dummy_models.py`
Creates random PyTorch weights for `covid_model.pth` and `skin_model.pth`.
Use this for testing when you don't have real model weights.

```bash
python generate_dummy_models.py
```

---

## How the System Works End-to-End

```
Hospital / Client
│
├─ trains on local patient data (never leaves the hospital)
├─ computes weight delta
├─ applies differential privacy (clip + noise)
├─ uploads delta ──────────────────────► IPFS
│                                         └─ returns CID: "QmXyz..."
└─ POST /api/updates ─────────────────► server.js
                                          └─ Fabric chaincode
                                               └─ on-chain record saved
```

---

## What Is NOT Built (intentionally left out)

| Missing piece | Why |
|---|---|
| Model aggregation (FedAvg) | Would need a coordinator server to fetch all CIDs from IPFS, average the deltas, push new global model back |
| Real training loop | `simulate_local_training()` uses noise only — replace with real PyTorch training on your dataset |
| Real model weights | `covid_model.pth` / `skin_model.pth` need real pretrained weights |
| Authentication on REST API | No auth on the Express server currently |

---

## Running the Project (Codespace or Docker)

### What Codespace gives you free
- Docker, Node.js, Python, Git, Linux terminal

### What you must install
```bash
# IPFS
wget https://dist.ipfs.tech/kubo/v0.29.0/kubo_v0.29.0_linux-amd64.tar.gz
tar -xvzf kubo_v0.29.0_linux-amd64.tar.gz && cd kubo && sudo bash install.sh
ipfs init

# Fabric binaries + fabric-samples
cd ~/fedlearn-fabric
curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.0 1.5.5
echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```

### Startup order (4 terminals)

```bash
# Terminal 1
ipfs daemon

# Terminal 2
cd ~/fabric-samples/test-network
./network.sh up createChannel -ca
./network.sh deployCC -ccn modelregistry -ccp ~/fedlearn-fabric/chaincode/modelregistry -ccl javascript

# Terminal 3
cd ~/fedlearn-fabric/server
npm install
node enrollAdmin.js && node registerUser.js
node server.js

# Terminal 4
cd ~/fedlearn-fabric/client
pip install torch requests ipfshttpclient
python ../models/generate_dummy_models.py
python fl_client.py --sender Client1 --model covid --round 1
```

### Verify it works
```bash
curl http://localhost:3000/api/updates
```

---

## Current Status

| Component | Status |
|---|---|
| Chaincode | Done |
| REST Server | Done |
| FL Client | Done |
| Dummy model generator | Done |
| Real model weights | Not added yet |
| Model aggregation (FedAvg) | Not built yet |
| Connection to any external system | None — standalone |

---

## Next Steps (planned)

- User will provide a clean plan
- Code will be updated based on new requirements
