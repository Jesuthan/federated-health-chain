#!/usr/bin/env python3
"""
Federated Learning Client  —  FedProx / FedAvg  (Real Training)
================================================================
Pipeline:
  [0] (optional) Pull latest global model from blockchain → IPFS
  [1] Load pretrained model weights
  [2] Real local training — FedProx (default) or FedAvg
      Uses real PyTorch gradient descent on synthetic non-IID hospital data.
      Replace generate_hospital_data() with your real DataLoader when available.
  [3] Compute weight delta  (updated - original)
  [4] Apply differential privacy  (global L2 clip + Gaussian noise)
  [5] Upload delta to IPFS → CID
  [6] POST CID to REST server → stored on Fabric blockchain

Non-IID data simulation
-----------------------
Each hospital has a skewed class distribution (dominant class per hospital ID).
This reproduces the real-world scenario where Hospital A has mostly COVID-positive
patients, Hospital B has mostly healthy patients, etc.  FedProx's proximal term
prevents local models from over-fitting to their dominant class.

Algorithm
---------
FedProx local objective:  min  F_i(w)  +  (μ/2) · ||w - w_global||²

The proximal term (μ/2)||w - w_global||² is added to the cross-entropy loss
during every gradient step.  This constrains the local model from drifting too
far from the global model — the key advantage over FedAvg on non-IID data.
"""

import argparse
import os
import tempfile

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Real data root — folder relative to this script's location
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# ─── Model Architectures ───────────────────────────────────────────────────────

class CovidCNN(nn.Module):
    """4-block CNN for COVID-19 chest X-ray classification (3 classes)."""
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


MODEL_CLASS    = {'covid': CovidCNN}
NUM_CLASSES    = {'covid': 3}
CLASS_NAMES    = {
    'covid': ['COVID-19', 'Normal', 'Viral Pneumonia'],
}


# ─── Synthetic Data Generation ─────────────────────────────────────────────────

def generate_hospital_data(model_type: str, n_samples: int, hospital_id: int) -> TensorDataset:
    """
    Generate synthetic non-IID hospital data.

    Each hospital has a DOMINANT CLASS reflecting real-world data heterogeneity:
      Hospital 1 → mostly COVID-positive  (class 0 = 70%)
      Hospital 2 → mostly Normal          (class 1 = 70%)
      Hospital 3 → mostly Viral Pneumonia (class 2 = 70%)  etc.

    Class-specific image patterns (brightness + spatial region) give the model
    real signal to learn from, so gradient descent produces meaningful updates.

    Replace this function with a real torch DataLoader on your hospital dataset.
    The FL pipeline (delta, DP, IPFS, blockchain) is identical for real data.
    """
    torch.manual_seed(hospital_id * 42)
    num_classes    = NUM_CLASSES[model_type]
    dominant_class = (hospital_id - 1) % num_classes

    # Build non-IID label distribution: 70% dominant, rest shared equally
    minority_frac = 0.30 / max(num_classes - 1, 1)
    counts = []
    for c in range(num_classes):
        frac = 0.70 if c == dominant_class else minority_frac
        counts.append(max(1, int(n_samples * frac)))
    # Trim / pad to exactly n_samples
    while sum(counts) > n_samples:
        counts[dominant_class] -= 1
    while sum(counts) < n_samples:
        counts[dominant_class] += 1

    labels_list = []
    for c, cnt in enumerate(counts):
        labels_list.extend([c] * cnt)
    labels = torch.tensor(labels_list, dtype=torch.long)

    img_size = 224
    images = torch.randn(n_samples, 1, img_size, img_size) * 0.15

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        brightness = (c / (num_classes - 1)) * 1.2 - 0.6
        images[mask] += brightness
        quad = c % 4
        h, w = img_size, img_size
        mid_h, mid_w = h // 2, w // 2
        regions = [
            (slice(0, mid_h),   slice(0, mid_w)),    # top-left
            (slice(0, mid_h),   slice(mid_w, w)),     # top-right
            (slice(mid_h, h),   slice(0, mid_w)),     # bottom-left
            (slice(mid_h, h),   slice(mid_w, w)),     # bottom-right
        ]
        r = regions[quad]
        images[mask, :, r[0], r[1]] += 0.4

    return TensorDataset(images, labels)


def generate_test_data(model_type: str, n_samples: int = 40) -> TensorDataset:
    """
    Balanced test set (equal class distribution) for accuracy evaluation.
    Uses a fixed seed (seed=999) for reproducibility across rounds.
    """
    torch.manual_seed(999)
    num_classes  = NUM_CLASSES[model_type]
    per_class    = max(1, n_samples // num_classes)
    labels_list  = []
    for c in range(num_classes):
        labels_list.extend([c] * per_class)
    labels = torch.tensor(labels_list, dtype=torch.long)
    n      = len(labels)

    img_size = 224
    images = torch.randn(n, 1, img_size, img_size) * 0.15
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        brightness = (c / (num_classes - 1)) * 1.2 - 0.6
        images[mask] += brightness
        quad = c % 4
        h, w = img_size, img_size
        mid_h, mid_w = h // 2, w // 2
        regions = [
            (slice(0, mid_h),   slice(0, mid_w)),
            (slice(0, mid_h),   slice(mid_w, w)),
            (slice(mid_h, h),   slice(0, mid_w)),
            (slice(mid_h, h),   slice(mid_w, w)),
        ]
        r = regions[quad]
        images[mask, :, r[0], r[1]] += 0.4

    return TensorDataset(images, labels)


# ─── Real Data Loaders ────────────────────────────────────────────────────────

# Expected folder layout:
#   data/hospital_1/covid/COVID-19/img.jpg
#   data/hospital_1/covid/Normal/img.jpg
#   data/hospital_1/covid/Viral_Pneumonia/img.jpg
#   data/test/covid/COVID-19/img.jpg  (shared evaluation set)

_TRANSFORMS = {}

def _get_transform(model_type: str = 'covid'):
    if model_type not in _TRANSFORMS:
        size = 224
        _TRANSFORMS[model_type] = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    return _TRANSFORMS[model_type]


def load_hospital_data(model_type: str, hospital_id: int, batch_size: int = 8):
    """
    Load real hospital training data from data/hospital_{id}/{model_type}/.
    Returns (DataLoader, n_samples) or (None, 0) if the folder doesn't exist.
    Sub-folders = class names (ImageFolder convention).
    """
    if not TORCHVISION_AVAILABLE:
        return None, 0
    folder = os.path.abspath(os.path.join(DATA_DIR, f'hospital_{hospital_id}', model_type))
    if not os.path.isdir(folder):
        return None, 0
    dataset = ImageFolder(folder, transform=_get_transform(model_type))
    if len(dataset) == 0:
        return None, 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"  Real data loaded: {folder}")
    print(f"  Classes: {dataset.classes}")
    dist = {cls: sum(1 for _, l in dataset.samples if l == i)
            for i, cls in enumerate(dataset.classes)}
    print(f"  Distribution: {dist}")
    return loader, len(dataset)


def load_test_data(model_type: str, batch_size: int = 16):
    """
    Load shared evaluation set from data/test/{model_type}/.
    Returns DataLoader or None if not present.
    """
    if not TORCHVISION_AVAILABLE:
        return None
    folder = os.path.abspath(os.path.join(DATA_DIR, 'test', model_type))
    if not os.path.isdir(folder):
        return None
    dataset = ImageFolder(folder, transform=_get_transform(model_type))
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ─── Real Training ─────────────────────────────────────────────────────────────

def train_fedprox(model: nn.Module, dataloader: DataLoader,
                  global_weights: dict, mu: float,
                  epochs: int = 2, lr: float = 0.01) -> nn.Module:
    """
    Real FedProx local training.

    Minimises:  L(w)  =  CrossEntropy(w)  +  (μ/2) · ||w - w_global||²

    The proximal term (μ/2)||w - w_global||² penalises deviations from the
    global model, directly addressing client drift on non-IID medical data.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = correct = total = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs  = model(batch_x)
            ce_loss  = criterion(outputs, batch_y)

            # FedProx proximal term
            prox = torch.tensor(0.0)
            for name, param in model.named_parameters():
                if name in global_weights:
                    diff = param - global_weights[name].to(param.device)
                    prox = prox + diff.norm() ** 2
            loss = ce_loss + (mu / 2.0) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += ce_loss.item()
            _, predicted = outputs.max(1)
            total   += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1}/{epochs}  CE={total_loss/len(dataloader):.4f}  "
              f"prox={prox.item():.4f}  train_acc={acc*100:.1f}%")
    return model


def train_fedavg(model: nn.Module, dataloader: DataLoader,
                 epochs: int = 2, lr: float = 0.01) -> nn.Module:
    """Standard local SGD — FedAvg baseline (no proximal constraint)."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = correct = total = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(dataloader):.4f}  "
              f"train_acc={acc*100:.1f}%")
    return model


# ─── Model loading ─────────────────────────────────────────────────────────────

def load_weights(model_type: str, state_dict: dict = None) -> nn.Module:
    """Load model with pretrained weights if available, else random init."""
    model = MODEL_CLASS[model_type]()
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys (using random init for those)")
    return model


def load_pretrained(model_type: str) -> nn.Module:
    """Load pretrained weights from models/{model_type}_model.pth."""
    weight_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_type}_model.pth')
    )
    if not os.path.exists(weight_file):
        print(f"  Warning: pretrained weights not found at {weight_file}")
        print("  Using random initialisation. Place pretrained weights in models/ folder.")
        return load_weights(model_type)

    state = torch.load(weight_file, map_location='cpu', weights_only=False)
    model = load_weights(model_type, state)
    print(f"  Pretrained weights loaded: {weight_file}")
    return model


# ─── IPFS / Blockchain helpers ─────────────────────────────────────────────────

def pull_global_model(model_type: str, server_url: str):
    """[0] Download latest global model from blockchain → IPFS."""
    import ipfshttpclient
    try:
        resp = requests.get(f"{server_url}/api/global-model/{model_type}/latest", timeout=15)
        resp.raise_for_status()
        cid       = resp.json().get('ipfsCID')
        round_num = resp.json().get('round', '?')
        print(f"  Latest global model: round {round_num}  CID: {cid}")
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print("  No global model yet — using pretrained HF weights.")
            return None
        raise

    try:
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
    except Exception:
        raise RuntimeError("Cannot connect to IPFS.")
    raw = client.cat(cid)
    client.close()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        state = torch.load(tmp_path, map_location='cpu', weights_only=False)
        model = load_weights(model_type, state)
        model.eval()
        print(f"  Global model loaded ({len(state)} tensors)")
        return model
    finally:
        os.unlink(tmp_path)


def compute_delta(original: dict, updated: dict) -> dict:
    """[3] Weight delta = updated - original (trainable params only)."""
    return {k: updated[k] - original[k] for k in original if k in updated}


def apply_dp(delta: dict, clip: float, noise: float) -> dict:
    """[4] (ε,δ)-DP: global L2 clip + Gaussian noise."""
    norm  = torch.cat([v.flatten() for v in delta.values()]).norm().item()
    scale = min(1.0, clip / (norm + 1e-8))
    print(f"  L2 norm={norm:.4f}  clip_scale={scale:.4f}  sigma={noise}")
    clipped = {k: v * scale for k, v in delta.items()}
    return   {k: v + torch.randn_like(v) * noise for k, v in clipped.items()}


def upload_ipfs(delta: dict) -> str:
    """[5] Serialise delta and upload to IPFS. Returns CID."""
    import ipfshttpclient
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(delta, tmp_path)
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        cid    = client.add(tmp_path)['Hash']
        client.close()
        print(f"  Uploaded to IPFS  CID: {cid}")
        return cid
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def submit_blockchain(sender, model_type, round_num, cid,
                      clip, noise, samples, algo, server_url) -> dict:
    """[6] POST CID + metadata to Fabric REST server."""
    resp = requests.post(f"{server_url}/api/updates", json={
        'sender': sender, 'modelType': model_type, 'round': round_num,
        'ipfsCID': cid, 'clipValue': clip, 'noiseScale': noise,
        'sampleCount': samples, 'algorithm': algo,
    }, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get('aggregating'):
        print("  Aggregation threshold reached — FedProx aggregation triggered!")
    else:
        print(f"  Waiting for {result.get('minClients',2) - result.get('updateCount',1)} more hospital(s)")
    return result


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FL Client — FedProx/FedAvg')
    parser.add_argument('--sender',      required=True)
    parser.add_argument('--model',       required=True,  choices=['covid'])
    parser.add_argument('--round',       required=True,  type=int)
    parser.add_argument('--clip',        default=1.0,    type=float)
    parser.add_argument('--noise',       default=0.1,    type=float)
    parser.add_argument('--server',      default='http://localhost:3000')
    parser.add_argument('--pull-global', action='store_true')
    parser.add_argument('--algo',        default='fedprox', choices=['fedprox', 'fedavg'])
    parser.add_argument('--mu',          default=0.01,   type=float)
    parser.add_argument('--samples',     default=32,     type=int,
                        help='Local training samples per hospital (default: 32)')
    parser.add_argument('--epochs',      default=2,      type=int)
    parser.add_argument('--lr',          default=0.01,   type=float)
    parser.add_argument('--hospital-id', default=1,      type=int,
                        help='Hospital ID determines non-IID class distribution (default: 1)')
    args = parser.parse_args()

    # Derive hospital_id from sender name if not set explicitly
    hospital_id = args.hospital_id
    try:
        # e.g. "Hospital2" → 2
        import re
        m = re.search(r'\d+', args.sender)
        if m:
            hospital_id = int(m.group())
    except Exception:
        pass

    sep = '=' * 64
    print(f"\n{sep}")
    print("  Federated Learning Client  (Real Training)")
    print(f"  Hospital     : {args.sender}  (id={hospital_id})")
    print(f"  Model        : {args.model.upper()}")
    print(f"  Round        : {args.round}")
    algo_str = f"{args.algo.upper()}" + (f"  mu={args.mu}" if args.algo == 'fedprox' else "  (baseline)")
    print(f"  Algorithm    : {algo_str}")
    print(f"  Training     : {args.samples} samples, {args.epochs} epochs, lr={args.lr}")
    print(f"  DP           : clip={args.clip}  sigma={args.noise}")
    print(sep)

    # [0] Pull global model
    base_model = None
    if args.pull_global:
        print("\n[0/6] Pulling latest global model…")
        base_model = pull_global_model(args.model, args.server)

    # [1] Load starting weights
    print("\n[1/6] Loading model…")
    if base_model is not None:
        model = base_model
    elif args.round > 1:
        # Rounds 2+: train from previous global model so deltas are consistent
        try:
            pulled = pull_global_model(args.model, args.server)
            model = pulled if pulled is not None else load_pretrained(args.model)
        except Exception as exc:
            print(f"  Warning — could not pull global model ({exc}), using pretrained.")
            model = load_pretrained(args.model)
    else:
        model = load_pretrained(args.model)

    original_state = {n: p.data.clone() for n, p in model.named_parameters()}

    # [2] Load training data — real if available, synthetic fallback
    print(f"\n[2/6] Loading training data for {args.sender}…")
    real_loader, real_n = load_hospital_data(args.model, hospital_id, batch_size=min(8, args.samples))

    if real_loader is not None:
        dataloader = real_loader
        print(f"  Using REAL data  ({real_n} samples)")
    else:
        print(f"  Real data not found — using synthetic non-IID data")
        dataset    = generate_hospital_data(args.model, args.samples, hospital_id)
        dataloader = DataLoader(dataset, batch_size=min(8, args.samples), shuffle=True)
        labels     = dataset.tensors[1]
        class_names = CLASS_NAMES[args.model]
        dist = {class_names[c]: int((labels == c).sum()) for c in range(NUM_CLASSES[args.model])}
        print(f"  Class distribution (non-IID): {dist}")

    print(f"\n  Training with {args.algo.upper()}…")
    if args.algo == 'fedprox':
        model = train_fedprox(model, dataloader, original_state, args.mu, args.epochs, args.lr)
    else:
        model = train_fedavg(model, dataloader, args.epochs, args.lr)

    updated_state = {n: p.data for n, p in model.named_parameters()}

    # [3] Delta
    print("\n[3/6] Computing weight delta…")
    delta = compute_delta(original_state, updated_state)
    print(f"  {len(delta)} parameter tensors")

    # [4] Differential privacy
    print("\n[4/6] Applying differential privacy…")
    delta = apply_dp(delta, args.clip, args.noise)

    # [5] IPFS upload
    print("\n[5/6] Uploading delta to IPFS…")
    cid = upload_ipfs(delta)

    # [6] Blockchain
    print("\n[6/6] Recording on Fabric blockchain…")
    result = submit_blockchain(
        args.sender, args.model, args.round,
        cid, args.clip, args.noise, args.samples, args.algo, args.server,
    )

    print(f"\n{sep}")
    print("  Round complete!")
    print(f"  Algorithm : {args.algo.upper()}" + (f"  mu={args.mu}" if args.algo == 'fedprox' else ""))
    print(f"  IPFS CID  : {cid}")
    print(f"  Update ID : {result.get('id', 'N/A')}")
    print(f"{sep}\n")


if __name__ == '__main__':
    main()
