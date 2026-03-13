# FedHealth Chain — Complete Project Documentation

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Why Did We Build This?](#2-why-did-we-build-this)
3. [How Does It Work? (Simple Explanation)](#3-how-does-it-work-simple-explanation)
4. [System Architecture](#4-system-architecture)
5. [Technologies Used](#5-technologies-used)
6. [Project Structure](#6-project-structure)
7. [Key Concepts Explained](#7-key-concepts-explained)
8. [How Each Component Works](#8-how-each-component-works)
9. [The Full Flow — Step by Step](#9-the-full-flow--step-by-step)
10. [API Reference](#10-api-reference)
11. [Setup Guide](#11-setup-guide)
12. [Running the System](#12-running-the-system)
13. [Dashboard](#13-dashboard)
14. [Security & Privacy Features](#14-security--privacy-features)
15. [Limitations & Future Work](#15-limitations--future-work)

---

## 1. What Is This Project?

**FedHealth Chain** is a system that allows multiple hospitals to train an AI model together **without sharing their patient data**.

Instead of sending patient data to a central server (which is a privacy risk), each hospital trains the model on their own data locally, then only shares the **model updates** (what the model learned). These updates are stored on a **blockchain** so nobody can tamper with them.

Think of it like this:
> Each hospital bakes a cake using their own secret recipe. They only share the final taste rating — not the recipe itself. A blockchain records every rating so everyone can trust the process.

---

## 2. Why Did We Build This?

### The Problem
- Hospitals have sensitive patient data (X-rays, skin images, medical records)
- Privacy laws (like HIPAA, GDPR) prevent sharing this data
- But AI models need large amounts of data to be accurate
- A single hospital doesn't have enough data alone

### The Solution
- **Federated Learning** — train AI at each hospital separately, share only model updates
- **Blockchain** — record every update permanently so it can't be changed or faked
- **IPFS** — store model update files in a decentralised way (no single server)
- **Differential Privacy** — add mathematical noise to updates so individual patient data can't be reverse-engineered

---

## 3. How Does It Work? (Simple Explanation)

```
Hospital 1 (has COVID X-ray data)
    |
    | 1. Train model on local data
    | 2. Compute what changed (weight delta)
    | 3. Add privacy noise (differential privacy)
    | 4. Upload delta file to IPFS → gets a CID (address)
    | 5. Record CID on blockchain
    |
    v
Blockchain stores: "Hospital1 submitted update for Round 1, file is at IPFS CID: Qm..."

Hospital 2 (has skin lesion data)
    |
    | Same process...
    |
    v
Blockchain stores: "Hospital2 submitted update for Round 1, file is at IPFS CID: Qm..."

Central Aggregator (future)
    |
    | 1. Read all CIDs from blockchain
    | 2. Download delta files from IPFS
    | 3. Average the updates (FedAvg)
    | 4. Update the global model
```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│                                                             │
│  Hospital 1        Hospital 2        Hospital 3             │
│  (fl_client.py)    (fl_client.py)    (fl_client.py)         │
│       │                 │                 │                  │
└───────┼─────────────────┼─────────────────┼──────────────────┘
        │                 │                 │
        │    HTTP POST /api/updates         │
        ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      SERVER LAYER                            │
│                                                             │
│              Express REST API (server.js)                   │
│                   Port 3000                                  │
│                       │                                      │
└───────────────────────┼─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐              ┌────────────────────┐
│  IPFS Node    │              │ Hyperledger Fabric  │
│  Port 5001    │              │ Blockchain          │
│               │              │                    │
│ Stores model  │              │ Stores CID +        │
│ delta files   │              │ metadata on ledger  │
│ Returns CID   │              │                    │
└───────────────┘              └────────────────────┘
                                        │
                               ┌────────────────────┐
                               │  Smart Contract     │
                               │  (modelregistry)    │
                               │                    │
                               │ - storeUpdate()     │
                               │ - getAllUpdates()    │
                               │ - queryByRound()    │
                               └────────────────────┘
```

---

## 5. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Hyperledger Fabric** | 2.5.0 | Permissioned blockchain for storing FL records |
| **IPFS (Kubo)** | 0.29.0 | Decentralised file storage for model deltas |
| **Python** | 3.12 | FL client, model training, differential privacy |
| **PyTorch** | 2.10.0 | Deep learning framework for CNN models |
| **Node.js** | 20.x | REST API server |
| **Express.js** | 4.x | Web framework for the REST API |
| **Docker** | Latest | Running Fabric peer/orderer containers |
| **WSL2 (Ubuntu)** | 24.04 | Linux environment on Windows |

---

## 6. Project Structure

```
fedlearn-fabric/
│
├── chaincode/
│   └── modelregistry/
│       ├── index.js          ← Smart contract (Fabric chaincode)
│       └── package.json
│
├── server/
│   ├── server.js             ← Express REST API
│   ├── enrollAdmin.js        ← Enroll Fabric CA admin
│   ├── registerUser.js       ← Register app user
│   ├── wallet/               ← Fabric identity credentials
│   └── public/
│       └── index.html        ← Web dashboard
│
├── client/
│   ├── fl_client.py          ← Federated Learning client
│   └── requirements.txt
│
├── models/
│   ├── generate_dummy_models.py   ← Generate test model weights
│   ├── covid_model.pth            ← COVID X-ray model weights
│   └── skin_model.pth             ← Skin lesion model weights
│
├── fabric-samples/           ← Hyperledger Fabric test network
│   └── test-network/
│
├── SETUP_GUIDE.md            ← Quick setup reference
├── DOCUMENTATION.md          ← This file
└── README.md
```

---

## 7. Key Concepts Explained

### Federated Learning (FL)
Normal AI training: send all data to one central server → train model.
Federated Learning: keep data at each hospital → train locally → only share what the model learned (weight updates).

**Why it matters:** Patient data never leaves the hospital.

### Weight Delta
When a model trains, its internal numbers (weights) change. A "weight delta" is just the difference between the weights before and after training:
```
delta = weights_after_training - weights_before_training
```
This delta is what gets shared — not the raw data.

### Differential Privacy (DP)
Even weight deltas can leak private information. DP adds two protections:
1. **Gradient Clipping** — limits how much any single update can change the model (prevents one patient's data from dominating)
2. **Gaussian Noise** — adds random mathematical noise so the exact delta can't be traced back to specific patients

### Blockchain (Hyperledger Fabric)
A blockchain is a database where:
- Records can be **added** but never **deleted or changed**
- Every record has a **timestamp** from the network itself (not the client)
- Multiple organisations (Org1, Org2) must **agree** before a record is added (consensus)

Hyperledger Fabric is a **permissioned** blockchain — only authorised users can participate (unlike Bitcoin which is public).

### Smart Contract (Chaincode)
A smart contract is code that runs on the blockchain. Our smart contract (`modelregistry`) has these functions:
- `storeUpdate` — save a new FL update record
- `getAllUpdates` — get all records
- `queryByRound` — get all records for a specific FL round
- `queryBySender` — get all records from a specific hospital

### IPFS (InterPlanetary File System)
IPFS is a decentralised file storage system. When you upload a file:
- IPFS gives it a unique address called a **CID** (Content Identifier)
- The CID is based on the file's content — if the file changes, the CID changes
- This makes tampering detectable

We store the model delta file on IPFS and put only the CID on the blockchain.

### CID (Content Identifier)
Example: `QmcM6QSAXmkrVKJ4hUMwZMd1jCszxnxmLN2WxEp2GUjjKi`

This is like a fingerprint of the file. Anyone with this CID can download the exact same file from IPFS.

---

## 8. How Each Component Works

### FL Client (`client/fl_client.py`)

The Python client simulates a hospital participating in federated learning.

**What it does step by step:**
1. **Load model** — loads the CNN model weights from a `.pth` file
2. **Simulate local training** — pretends to train on local hospital data (in production, this would use real patient data)
3. **Compute weight delta** — calculates the difference between old and new weights
4. **Apply differential privacy** — clips gradients and adds Gaussian noise
5. **Upload to IPFS** — saves the delta as a file and uploads it to IPFS, gets back a CID
6. **Submit to blockchain** — sends the CID + metadata to the REST API, which records it on Fabric

**Model architectures available:**
| Architecture | Layers | Parameters | Use case |
|---|---|---|---|
| CNN2 | 2 conv layers | ~1M | Fast, less accurate |
| CNN4 | 4 conv layers | ~8M | Default, balanced |
| CNN6 | 6 conv layers | ~33M | Slower, more accurate |

---

### Smart Contract (`chaincode/modelregistry/index.js`)

Runs inside the Fabric blockchain. Stores and retrieves FL update records.

**Key design decisions:**
- Uses `ctx.stub.getTxTimestamp()` for timestamps — this comes from the blockchain network itself, so all peers agree on the same value (prevents endorsement mismatch)
- Checks for duplicate IDs before storing
- Returns JSON strings (Fabric requires string/bytes)

---

### REST API (`server/server.js`)

Express.js server that acts as the bridge between Python clients and the Fabric blockchain.

**Why do we need this?** The Fabric SDK is only available in Node.js and Java. Python clients can't talk to Fabric directly, so the Node.js server translates HTTP requests into Fabric transactions.

---

### Dashboard (`server/public/index.html`)

A single-page web app that shows the system status and all blockchain records in real time.

- Auto-refreshes every 10 seconds
- Shows Fabric, IPFS, and API status
- Displays all FL updates in a table with IPFS CIDs

---

## 9. The Full Flow — Step by Step

Here is exactly what happens when a hospital runs the FL client:

```
1. fl_client.py starts
   └── Reads COVID model from models/covid_model.pth

2. Simulates local training
   └── Slightly modifies model weights (pretending to train on data)

3. Computes weight delta
   └── delta = new_weights - old_weights

4. Applies differential privacy
   ├── Calculates global L2 norm of all gradients
   ├── Clips norm to max value (default: 1.0)
   └── Adds Gaussian noise (default scale: 0.1)

5. Uploads delta to IPFS
   ├── Saves delta as a .pt file (PyTorch format)
   ├── POST http://localhost:5001/api/v0/add
   └── Receives CID: "QmcM6QSAXmkrVKJ4..."

6. Submits to blockchain via REST API
   ├── POST http://localhost:3000/api/updates
   │   Body: { sender, modelType, round, ipfsCID, clipValue, noiseScale }
   │
   └── server.js receives the request
       ├── Generates unique ID: "update_hospital1_round1_1b905e16"
       ├── Calls Fabric smart contract: storeUpdate(id, sender, ...)
       │
       └── Smart contract runs on BOTH peers (Org1 and Org2)
           ├── Checks for duplicate IDs
           ├── Gets timestamp from blockchain network
           ├── Creates record JSON
           ├── Writes to ledger
           └── Both peers must agree → transaction committed ✓

7. Response returned to fl_client.py
   └── Prints success with CID and Update ID
```

---

## 10. API Reference

Base URL: `http://localhost:3000`

### GET /health
Check if server is running.

**Response:**
```json
{ "status": "ok", "timestamp": "2026-03-13T04:26:17.000Z" }
```

---

### GET /api/updates
Get all FL update records from blockchain.

**Response:**
```json
{
  "success": true,
  "count": 2,
  "updates": [
    {
      "updateId": "update_hospital1_round1_1b905e16",
      "sender": "hospital1",
      "modelType": "covid",
      "round": 1,
      "ipfsCID": "QmcM6QSAXmkrVKJ4hUMwZMd1jCszxnxmLN2WxEp2GUjjKi",
      "timestamp": "2026-03-13T04:26:17.000Z",
      "clipValue": 1,
      "noiseScale": 0.1
    }
  ]
}
```

---

### POST /api/updates
Submit a new FL update.

**Request body:**
```json
{
  "sender": "hospital1",
  "modelType": "covid",
  "round": 1,
  "ipfsCID": "QmcM6QSAXmkrVKJ4...",
  "clipValue": 1.0,
  "noiseScale": 0.1
}
```

**Response (201):**
```json
{
  "message": "Model update stored on blockchain",
  "id": "update_hospital1_round1_1b905e16",
  "sender": "hospital1",
  "modelType": "covid",
  "round": 1,
  "ipfsCID": "QmcM6QSAXmkrVKJ4..."
}
```

---

### GET /api/updates/round/:round
Get all updates for a specific FL round.

Example: `GET /api/updates/round/1`

---

### GET /api/updates/sender/:sender
Get all updates from a specific hospital.

Example: `GET /api/updates/sender/hospital1`

---

## 11. Setup Guide

See [SETUP_GUIDE.md](./SETUP_GUIDE.md) for detailed setup instructions including:
- Setting up on your own PC
- Setting up on a friend's laptop from scratch
- Troubleshooting common errors

---

## 12. Running the System

### Start everything (3 terminals needed)

**Terminal 1 — Fabric Network**
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

**Terminal 2 — IPFS**
```bash
ipfs daemon
```

**Terminal 3 — REST Server**
```bash
cd ~/fedlearn-fabric/server
node server.js
```

### Run FL clients

```bash
cd ~/fedlearn-fabric
source venv/bin/activate

# Round 1
python client/fl_client.py --sender hospital1 --model covid --round 1
python client/fl_client.py --sender hospital2 --model skin --round 1
python client/fl_client.py --sender hospital3 --model covid --round 1

# Round 2
python client/fl_client.py --sender hospital1 --model covid --round 2
python client/fl_client.py --sender hospital2 --model skin --round 2
```

---

## 13. Dashboard

Open browser → **http://localhost:3000**

### What you see:

**Status Bar**
- Fabric Blockchain: Online/Offline
- IPFS Node: Online + version
- REST API: Online/Offline

**Statistics**
- Total updates submitted
- Number of hospitals participating
- Number of FL rounds completed
- Number of model types

**Rounds Summary**
- Cards showing which hospitals submitted in each round

**Blockchain Records Table**
- Full list of all FL updates
- Update ID, sender, model type, round, IPFS CID, DP parameters, timestamp

---

## 14. Security & Privacy Features

### Differential Privacy
Every model update is protected with:
- **Gradient Clipping** (default norm = 1.0): Prevents any single data point from having too much influence
- **Gaussian Noise** (default scale = 0.1): Makes it mathematically impossible to reverse-engineer individual patient records

### Blockchain Immutability
- Once an update is recorded on Fabric, it **cannot be changed or deleted**
- Every record has a network-generated timestamp (not client-provided)
- Both Org1 and Org2 must endorse every transaction

### IPFS Content Addressing
- Model delta files are addressed by their content hash (CID)
- If anyone tampers with a file, the CID changes — making tampering detectable
- Files are stored independently of any central server

### Permissioned Network
- Only enrolled users with valid cryptographic credentials can submit updates
- The Fabric CA (Certificate Authority) manages who can participate

---

## 15. Limitations & Future Work

### Current Limitations
| Limitation | Description |
|------------|-------------|
| Simulated training | The FL client simulates training — it doesn't use real patient data |
| No aggregation | The system records updates but doesn't yet aggregate them into a global model |
| Single channel | All hospitals use one Fabric channel — in production, separate channels per consortium |
| Local IPFS | IPFS runs locally — in production, it should run on a distributed network |
| No authentication | The REST API has no login — anyone on localhost can submit |

### Future Improvements
- **FedAvg aggregation** — download all deltas from IPFS, compute weighted average, update global model
- **Real model integration** — connect to a HuggingFace pretrained model (e.g., medical imaging transformer)
- **Multi-channel Fabric** — separate channel per hospital group for stronger privacy
- **IPFS cluster** — run IPFS on multiple nodes across hospitals for true decentralisation
- **REST API authentication** — JWT tokens or mutual TLS for hospital authentication
- **Automated rounds** — orchestrator that triggers new FL rounds automatically
- **Model evaluation** — measure accuracy improvement across rounds

---

*FedHealth Chain — Built with Hyperledger Fabric 2.5, IPFS Kubo, PyTorch, and Express.js*
