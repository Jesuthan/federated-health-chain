#!/usr/bin/env python3
"""
FedProx / FedAvg Aggregator
============================
Auto-triggered by server.js when MIN_CLIENTS updates arrive for a round.

Pipeline:
  [1] Query blockchain for all updates in this round + model type
  [2] Download each client delta from IPFS
  [3] Weighted aggregation  (FedProx-aware weighted FedAvg)
  [4] Load base model  (previous global round OR HF pretrained weights)
  [5] Apply averaged delta → new global model weights
  [6] Real accuracy evaluation on synthetic balanced test set
  [7] Upload global model to IPFS → CID
  [8] Record global model CID on blockchain
  [9] POST accuracy + privacy metrics to REST server

Algorithm note
--------------
Server-side aggregation for FedProx is identical to FedAvg: a weighted average
of client deltas.  The FedProx contribution is in the CLIENT training step
(proximal term constrains local updates).  Server uses sample-count weighting
so hospitals with more patients contribute proportionally.

Usage (auto-called by server.js, or manually for testing):
    python server/aggregator.py --round 1 --model covid --algo fedprox --mu 0.01
    python server/aggregator.py --round 1 --model covid --algo fedavg
"""

import argparse
import math
import os
import sys
import tempfile
from datetime import datetime, timezone

import ipfshttpclient
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODELS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models')
DEFAULT_SERVER = os.environ.get('FL_SERVER_URL', 'http://localhost:3000')
IPFS_ADDR      = os.environ.get('IPFS_ADDR', '/ip4/127.0.0.1/tcp/5001')

# ─── Model Architectures (mirrors fl_client.py) ────────────────────────────────

class CovidCNN(nn.Module):
    """4-block CNN matching sanjulamaduranga/BFL_Healthcare_covid_19 weights."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(256 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layers(x).view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


class SkinCNN(nn.Module):
    """2-block CNN for HAM10000 skin lesion classification (7 classes)."""
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1   = nn.Linear(64 * 56 * 56, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


MODEL_CLASS = {'covid': CovidCNN, 'skin': SkinCNN}
NUM_CLASSES = {'covid': 3,        'skin': 7}

# ─── Synthetic Test Data (mirrors fl_client.py generate_test_data) ─────────────

def generate_test_data(model_type: str, n_samples: int = 60) -> TensorDataset:
    """
    Balanced synthetic test set — same generation logic as fl_client.py.
    Fixed seed=999 ensures the same test set every round for fair comparison.
    """
    torch.manual_seed(999)
    num_classes = NUM_CLASSES[model_type]
    per_class   = max(1, n_samples // num_classes)
    labels_list = []
    for c in range(num_classes):
        labels_list.extend([c] * per_class)
    labels = torch.tensor(labels_list, dtype=torch.long)
    n      = len(labels)

    images = torch.randn(n, 1, 224, 224) * 0.15
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        brightness = (c / (num_classes - 1)) * 1.2 - 0.6
        images[mask] += brightness
        quad = c % 4
        h, w = 224, 224
        mid_h, mid_w = h // 2, w // 2
        regions = [
            (slice(0, mid_h), slice(0, mid_w)),
            (slice(0, mid_h), slice(mid_w, w)),
            (slice(mid_h, h), slice(0, mid_w)),
            (slice(mid_h, h), slice(mid_w, w)),
        ]
        r = regions[quad]
        images[mask, :, r[0], r[1]] += 0.4

    return TensorDataset(images, labels)


# ─── Real Accuracy Evaluation ──────────────────────────────────────────────────

def evaluate_global_model(state_dict: dict, model_type: str, n_test: int = 60) -> float:
    """
    Run a real forward pass on a synthetic balanced test set.

    Uses the same image generation as the clients (brightness + spatial quadrant
    per class) so the model's learned representations transfer directly.
    Fixed seed=999 ensures identical test set across every FL round so accuracy
    numbers are strictly comparable round-over-round.

    Returns accuracy in [0, 1].
    """
    model = MODEL_CLASS[model_type]()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Note: {len(missing)} key(s) not in state_dict — using random init for those")
    model.eval()

    test_data  = generate_test_data(model_type, n_test)
    loader     = DataLoader(test_data, batch_size=16, shuffle=False)
    correct    = total = 0

    with torch.no_grad():
        for x, y in loader:
            _, pred = model(x).max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()

    accuracy = correct / total if total > 0 else 0.0

    # Per-class breakdown
    num_classes  = NUM_CLASSES[model_type]
    all_x, all_y = test_data.tensors
    per_class_acc = []
    with torch.no_grad():
        logits = model(all_x)
        preds  = logits.argmax(dim=1)
    for c in range(num_classes):
        mask = all_y == c
        if mask.sum() > 0:
            cls_acc = preds[mask].eq(all_y[mask]).float().mean().item()
            per_class_acc.append(f"class{c}={cls_acc*100:.0f}%")
    print(f"  Per-class accuracy: {' | '.join(per_class_acc)}")

    return round(accuracy, 4)


# ─── Blockchain / REST helpers ─────────────────────────────────────────────────

def fetch_round_updates(round_num: int, model_type: str, server_url: str) -> list:
    resp = requests.get(f"{server_url}/api/updates/round/{round_num}", timeout=30)
    resp.raise_for_status()
    all_updates = resp.json().get('updates', [])
    return [u for u in all_updates if u['modelType'] == model_type]


def record_global_model(round_num, model_type, cid, client_count,
                        accuracy, algorithm, mu, server_url) -> dict:
    payload = {
        'round':       round_num,
        'modelType':   model_type,
        'ipfsCID':     cid,
        'clientCount': client_count,
        'accuracy':    accuracy,
        'algorithm':   algorithm,
        'mu':          mu,
    }
    resp = requests.post(f"{server_url}/api/global-model", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def post_metrics(round_num, model_type, accuracy, algorithm, mu,
                 client_count, epsilon, server_url) -> dict:
    """POST accuracy + privacy metrics to the REST server for dashboard charts."""
    payload = {
        'round':       round_num,
        'modelType':   model_type,
        'accuracy':    accuracy,
        'algorithm':   algorithm,
        'mu':          mu,
        'clientCount': client_count,
        'epsilon':     epsilon,
        'timestamp':   datetime.now(timezone.utc).isoformat(),
    }
    resp = requests.post(f"{server_url}/api/metrics", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()

# ─── IPFS helpers ──────────────────────────────────────────────────────────────

def _ipfs_connect():
    try:
        return ipfshttpclient.connect(IPFS_ADDR)
    except Exception:
        raise RuntimeError(f"Cannot connect to IPFS at {IPFS_ADDR}. Run: ipfs daemon")


def download_from_ipfs(cid: str) -> dict:
    client = _ipfs_connect()
    try:
        raw = client.cat(cid)
    finally:
        client.close()
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        return torch.load(tmp_path, map_location='cpu', weights_only=False)
    finally:
        os.unlink(tmp_path)


def upload_to_ipfs(state_dict: dict) -> str:
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(state_dict, tmp_path)
        client = _ipfs_connect()
        try:
            result = client.add(tmp_path)
        finally:
            client.close()
        return result['Hash']
    finally:
        os.unlink(tmp_path)

# ─── Aggregation ───────────────────────────────────────────────────────────────

def weighted_fedavg(deltas: list, sample_counts: list) -> dict:
    """
    Weighted FedAvg aggregation.

    w_global = Σ_i  (n_i / Σ_j n_j)  ·  w_i

    Hospitals with larger local datasets contribute more to the global model,
    reflecting their greater statistical representation.  Falls back to equal
    weighting if all counts are identical (standard FedAvg).
    """
    total   = sum(sample_counts) or len(deltas)
    weights = [n / total for n in sample_counts]

    keys = set(deltas[0].keys())
    avg  = {}
    for key in keys:
        stacked  = torch.stack([d[key].float() for d in deltas if key in d])
        w_tensor = torch.tensor(
            weights[:len(deltas)], dtype=torch.float32
        ).view(-1, *([1] * (stacked.dim() - 1)))
        avg[key] = (stacked * w_tensor).sum(dim=0)

    equal = all(s == sample_counts[0] for s in sample_counts)
    print(f"  Weighting: {'equal (all clients same size)' if equal else 'proportional to sample count'}")
    for i, (_, w) in enumerate(zip([None] * len(deltas), weights)):
        print(f"    client {i+1}: {sample_counts[i]} samples  weight={w:.3f}")
    return avg

# ─── Privacy metrics ───────────────────────────────────────────────────────────

def compute_epsilon(clip: float, noise_scale: float,
                    sample_count: int, delta: float = 1e-5) -> float:
    """
    Privacy budget ε per FL round (Gaussian mechanism, moments accountant).

    ε ≈ (clip / (noise_scale · n)) · √(2 · ln(1.25 / δ))

    where n = local sample count, δ = 1e-5 (standard for medical applications).
    Total ε over T rounds scales as ≈ ε_round · √T (advanced composition).
    """
    if noise_scale <= 0 or sample_count <= 0:
        return float('inf')
    sensitivity = clip / sample_count
    eps = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / noise_scale
    return round(eps, 6)

# ─── Base model loader ─────────────────────────────────────────────────────────

def load_base_model(model_type: str, round_num: int, server_url: str) -> dict:
    """
    Load starting weights for this aggregation round.
    Priority: previous global model (round N-1) → HF pretrained weights.
    """
    if round_num > 1:
        try:
            resp = requests.get(
                f"{server_url}/api/global-model/{model_type}/latest", timeout=15
            )
            if resp.ok:
                cid = resp.json().get('ipfsCID')
                if cid:
                    print(f"  Using previous global model  CID: {cid}")
                    return download_from_ipfs(cid)
        except Exception as exc:
            print(f"  Warning — could not fetch previous global model: {exc}")

    base_path = os.path.abspath(os.path.join(MODELS_DIR, f'{model_type}_model.pth'))
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Base model not found: {base_path}\n"
            "Run:  python models/inspect_hf_models.py"
        )
    print(f"  Using HF pretrained base: {base_path}")
    return torch.load(base_path, map_location='cpu', weights_only=False)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FedProx/FedAvg Aggregator')
    parser.add_argument('--round',  required=True, type=int)
    parser.add_argument('--model',  required=True, choices=['covid', 'skin'])
    parser.add_argument('--server', default=DEFAULT_SERVER)
    parser.add_argument('--algo',   default='fedprox', choices=['fedprox', 'fedavg'],
                        help='Algorithm label for metrics (default: fedprox)')
    parser.add_argument('--mu',     default=0.01, type=float,
                        help='FedProx μ used on clients (for metrics logging)')
    args = parser.parse_args()

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  {'FedProx' if args.algo == 'fedprox' else 'FedAvg'} Aggregator")
    print(f"  Model : {args.model.upper()}  |  Round : {args.round}")
    if args.algo == 'fedprox':
        print(f"  mu    : {args.mu}")
    print(f"  Server: {args.server}")
    print(sep)

    # [1] Fetch update records from blockchain
    print(f"\n[1/9] Fetching round {args.round} updates for '{args.model}'…")
    updates = fetch_round_updates(args.round, args.model, args.server)
    if not updates:
        print("  No updates found — nothing to aggregate.")
        sys.exit(0)
    print(f"  Found {len(updates)} client update(s):")
    for u in updates:
        print(f"    {u['sender']:<22}  CID: {u['ipfsCID']}")

    # [2] Download deltas from IPFS
    print(f"\n[2/9] Downloading deltas from IPFS…")
    deltas, sample_counts, noise_scales = [], [], []
    for u in updates:
        print(f"  Downloading — {u['sender']}")
        delta = download_from_ipfs(u['ipfsCID'])
        deltas.append(delta)
        sample_counts.append(int(u.get('sampleCount', 1000)))
        noise_scales.append(float(u.get('noiseScale', 0.1)))
        print(f"    {len(delta)} parameter tensors  samples={sample_counts[-1]}")

    # [3] Weighted FedAvg
    print(f"\n[3/9] Weighted aggregation ({args.algo.upper()})…")
    avg_delta = weighted_fedavg(deltas, sample_counts)
    print(f"  Aggregated {len(avg_delta)} parameter tensors across {len(deltas)} clients")

    # [4] Load base model
    print(f"\n[4/9] Loading base model for round {args.round}…")
    base = load_base_model(args.model, args.round, args.server)
    print(f"  Base: {len(base)} parameter tensors")

    # [5] Apply averaged delta → global model
    print(f"\n[5/9] Applying averaged delta to base model…")
    global_weights = {}
    missing = []
    for key in base:
        if key in avg_delta:
            global_weights[key] = base[key] + avg_delta[key]
        else:
            global_weights[key] = base[key]
            missing.append(key)
    if missing:
        print(f"  {len(missing)} key(s) not in delta — kept base values")
    print(f"  Global model assembled  ({len(global_weights)} tensors)")

    # [6] Real accuracy evaluation
    print(f"\n[6/9] Evaluating global model on synthetic test set…")
    accuracy = evaluate_global_model(global_weights, args.model)
    print(f"  Accuracy: {accuracy*100:.2f}%  ({args.algo.upper()}"
          + (f"  mu={args.mu}" if args.algo == 'fedprox' else "") + ")")

    # [7] Upload global model to IPFS
    print(f"\n[7/9] Uploading global model to IPFS…")
    global_cid = upload_to_ipfs(global_weights)
    print(f"  Global model CID: {global_cid}")

    # [8] Record on blockchain
    print(f"\n[8/9] Recording on blockchain…")
    result = record_global_model(
        round_num=args.round, model_type=args.model, cid=global_cid,
        client_count=len(updates), accuracy=accuracy,
        algorithm=args.algo, mu=args.mu, server_url=args.server,
    )
    print(f"  Recorded: {result.get('id', 'N/A')}")

    # [9] Post metrics for dashboard
    print(f"\n[9/9] Posting metrics to dashboard…")
    avg_samples = int(sum(sample_counts) / len(sample_counts))
    avg_clip    = sum(float(u.get('clipValue', 1.0)) for u in updates) / len(updates)
    avg_noise   = sum(noise_scales) / len(noise_scales)
    epsilon     = compute_epsilon(avg_clip, avg_noise, avg_samples)
    try:
        post_metrics(
            round_num=args.round, model_type=args.model, accuracy=accuracy,
            algorithm=args.algo, mu=args.mu, client_count=len(updates),
            epsilon=epsilon, server_url=args.server,
        )
        print(f"  Metrics posted  epsilon={epsilon:.4f}  accuracy={accuracy*100:.2f}%")
    except Exception as exc:
        print(f"  Warning — could not post metrics: {exc}")

    print(f"\n{sep}")
    print(f"  Aggregation complete!")
    print(f"  Model         : {args.model.upper()}")
    print(f"  Round         : {args.round}")
    print(f"  Algorithm     : {args.algo.upper()}" + (f"  mu={args.mu}" if args.algo == 'fedprox' else ""))
    print(f"  Clients merged: {len(updates)}")
    print(f"  Accuracy      : {accuracy*100:.2f}%")
    print(f"  Privacy eps   : {epsilon:.4f}  (per round, delta=1e-5)")
    print(f"  Global CID    : {global_cid}")
    print(f"{sep}\n")


if __name__ == '__main__':
    main()
