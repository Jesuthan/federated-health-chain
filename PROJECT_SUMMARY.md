# Federated Learning Project — Summary

## What Is Built

A full **Federated Learning (FL) system** combining:

- **Hyperledger Fabric** — blockchain records every model update and global model CID
- **IPFS** — decentralised storage for weight deltas and global models
- **Express REST API** — bridge between Python clients/aggregator and Fabric
- **Python FL Client** — loads real model, trains locally, applies DP, submits update
- **FedAvg Aggregator** — fetches deltas from IPFS, averages them, pushes new global model
- **Live Dashboard** — browser UI showing round status, hospital updates, global model CIDs

---

## Project Structure

```
fedlearn-fabric/
├── chaincode/modelregistry/
│   └── index.js               <- Fabric smart contract
├── server/
│   ├── server.js              <- Express REST API (auto-triggers FedAvg)
│   ├── aggregator.py          <- FedAvg coordinator
│   ├── enrollAdmin.js
│   ├── registerUser.js
│   └── package.json
├── client/
│   ├── fl_client.py           <- Hospital FL client (CovidCNN + SkinCNN)
│   └── requirements.txt
├── models/
│   ├── inspect_hf_models.py   <- Downloads real weights from HuggingFace
│   ├── covid_model.pth        <- Real COVID-19 weights (downloaded from HF)
│   └── skin_model.pth         <- Real Skin Cancer weights (downloaded from HF)
├── public/
│   └── index.html             <- Live dashboard at http://localhost:3000
└── README.md
```

---

## Component Status

| Component | Status | Notes |
|---|---|---|
| Fabric chaincode | Done | storeUpdate, storeGlobalModel, queryByRoundAndModel, getLatestGlobalModel |
| REST server | Done | 8 endpoints, auto-triggers FedAvg at MIN_CLIENTS threshold |
| FedAvg aggregator | Done | server/aggregator.py — architecture-agnostic tensor averaging |
| FL client | Done | CovidCNN + SkinCNN, --pull-global flag, DP (clip + noise) |
| Real COVID weights | Done | Downloaded from HuggingFace, strict load verified (0 mismatches) |
| Real Skin weights | Pending | HF repo gated — needs owner to remove gated access |
| Dashboard | Done | public/index.html — auto-refreshes every 5s |
| Real training loop | Intentional placeholder | simulate_local_training() — hospitals replace with their own dataset |

---

## Model Architectures

### CovidCNN — matched exactly from HuggingFace state_dict

```
Input: 1 x 224 x 224 (grayscale)
layers.0  Conv2d(1, 32, 3)   + ReLU + MaxPool2d(2)   -> 112x112
layers.3  Conv2d(32, 64, 3)  + ReLU + MaxPool2d(2)   -> 56x56
layers.6  Conv2d(64, 128, 3) + ReLU + MaxPool2d(2)   -> 28x28
layers.9  Conv2d(128,256, 3) + ReLU + MaxPool2d(2)   -> 14x14
fc1       Linear(50176, 128)   [256 * 14 * 14 = 50176]
fc2       Linear(128, 3)
Output: raw logits, 3 classes
```

### SkinCNN — 2-layer Keras CNN converted to PyTorch

```
Input: 1 x 224 x 224 (grayscale)
conv1     Conv2d(1, 32, 3) + ReLU + MaxPool2d(2)     -> 112x112
conv2     Conv2d(32, 64, 3) + ReLU + MaxPool2d(2)    -> 56x56
fc1       Linear(200704, 128)   [64 * 56 * 56 = 200704]
fc2       Linear(128, 7)
Output: Softmax probabilities, 7 classes
NOTE: key names (conv1/conv2/fc1/fc2) to be confirmed once HF repo is ungated
```

---

## Full Round Flow

```
Round N start:
  Hospital1  ->  pull global model CID from blockchain
                 download model from IPFS
                 train locally on patient data
                 compute delta (updated - original)
                 apply DP (clip + Gaussian noise)
                 upload delta to IPFS  ->  CID_1
                 POST CID_1 to /api/updates

  Hospital2  ->  (same steps)  ->  CID_2
                 POST CID_2 to /api/updates

server.js sees updateCount >= MIN_CLIENTS (2):
  ->  spawns aggregator.py --round N --model covid

aggregator.py:
  ->  fetch CID_1, CID_2 from blockchain
  ->  download delta_1, delta_2 from IPFS
  ->  FedAvg: avg_delta = mean(delta_1, delta_2)
  ->  load base model (previous global or HF pretrained)
  ->  global_weights = base + avg_delta
  ->  upload global_weights to IPFS  ->  GLOBAL_CID
  ->  POST GLOBAL_CID to /api/global-model  ->  blockchain record

Round N+1:
  Hospital1  ->  GET /api/global-model/covid/latest  ->  GLOBAL_CID
                 download global model from IPFS
                 train locally  ->  ...
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/api/updates` | All hospital updates |
| POST | `/api/updates` | Submit update (auto-triggers FedAvg at threshold) |
| GET | `/api/updates/round/:round` | Filter by FL round |
| GET | `/api/updates/sender/:sender` | Filter by hospital |
| POST | `/api/global-model` | Record global model (called by aggregator) |
| GET | `/api/global-model/:modelType/latest` | Latest global model CID |
| GET | `/api/global-model/:modelType/round/:round` | Global model for specific round |

---

## One Remaining Step

The Skin Cancer HuggingFace repo (`sanjulamaduranga/BFL_Healthcare_skincancer`) has
gated access enabled. Once the repo owner removes gating, run:

```bash
python models/inspect_hf_models.py
```

This downloads `skin_model.pth` and prints the real layer key names.
If the keys differ from `conv1/conv2/fc1/fc2`, update `SkinCNN` in `client/fl_client.py`
to match — same way `CovidCNN` was matched from the real state_dict.
