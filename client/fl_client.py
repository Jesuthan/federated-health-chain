#!/usr/bin/env python3
"""
Federated Learning Client
=========================
Pipeline:
  [0] (optional) Pull latest global model from IPFS via blockchain
  [1] Load model weights (.pth)
  [2] Simulate local training  ← replace with real training on hospital data
  [3] Compute weight delta (updated - original)
  [4] Apply differential privacy (global clip + Gaussian noise)
  [5] Upload delta to IPFS → get CID
  [6] POST CID to REST server → stored on Fabric blockchain

Architecture note
-----------------
CovidCNN and SkinCNN below are PLACEHOLDERS based on what we know:
  - COVID : custom CNN, 224×224 grayscale, 3 classes
  - Skin  : 2-layer CNN (converted from Keras), 224×224 grayscale, 7 classes

Run  python models/inspect_hf_models.py  first.  That script downloads the
real weights, prints every layer name + shape, and writes
models/generated_archs.py with auto-generated class code.  Replace these
placeholder classes with the generated ones so state_dict keys match exactly.
"""

import argparse
import os
import tempfile

import requests
import torch
import torch.nn as nn

# ─── Model Architectures ───────────────────────────────────────────────────────
#
# Input for both real models: 1 channel (grayscale), 224×224
# After 2× MaxPool2d(2):  224 → 112 → 56
#
# IMPORTANT: run inspect_hf_models.py and replace these classes with the
# auto-generated ones from models/generated_archs.py if the layer names or
# sizes differ.


class CovidCNN(nn.Module):
    """
    Real COVID-19 model architecture — matched exactly from HuggingFace state_dict.
    4-block CNN: each block is Conv2d -> ReLU -> MaxPool2d(2).
    Input: 224x224 grayscale (1 channel).  224->112->56->28->14 after 4 pools.
    Flatten: 256 * 14 * 14 = 50176.  Output: 3 classes.
    State dict keys: layers.0/3/6/9 (conv weights), fc1, fc2.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 224->112
            nn.Conv2d(32,  64,  3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112->56
            nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56->28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
        )
        self.fc1 = nn.Linear(256 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layers(x).view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


class SkinCNN(nn.Module):
    """
    Skin Cancer model -- 2-layer CNN converted from Keras Sequential to PyTorch.
    Input: 224x224 grayscale (1 channel).  224->112->56 after 2 MaxPool2d(2).
    Flatten: 64 * 56 * 56 = 200704.  Output: 7 classes (Softmax probabilities).
    State dict keys: conv1, conv2, fc1, fc2.
    NOTE: key names will be verified against the real weights once the HF repo
    is ungated. Load uses strict=False until confirmed.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1   = nn.Linear(64 * 56 * 56, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(F.relu(self.conv1(x)))  # 224->112
        x = self.pool(F.relu(self.conv2(x)))  # 112->56
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


# Map used by --model and --arch flags
MODEL_CLASS = {
    'covid': CovidCNN,
    'skin':  SkinCNN,
}

# ─── Pipeline steps ────────────────────────────────────────────────────────────

def pull_global_model(model_type: str, server_url: str) -> nn.Module:
    """
    [0] Download the latest global model weights from blockchain → IPFS.
    Returns a freshly initialised model loaded with the global weights.
    Falls back to local HF weights if no global model exists yet (round 1).
    """
    import ipfshttpclient

    try:
        resp = requests.get(
            f"{server_url}/api/global-model/{model_type}/latest",
            timeout=15,
        )
        resp.raise_for_status()
        cid = resp.json().get('ipfsCID')
        round_num = resp.json().get('round', '?')
        print(f"  Latest global model: round {round_num}, CID: {cid}")
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print("  No global model on blockchain yet — using local HF weights as base.")
            return None
        raise

    try:
        client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
    except Exception:
        raise RuntimeError(
            "Cannot connect to IPFS. Make sure IPFS daemon is running: ipfs daemon"
        )

    raw = client.cat(cid)
    client.close()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        state = torch.load(tmp_path, map_location='cpu', weights_only=False)
        model = MODEL_CLASS[model_type]()
        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"  Global model loaded ({len(state)} parameter tensors)")
        return model
    finally:
        os.unlink(tmp_path)


def load_local_model(model_type: str) -> nn.Module:
    """
    [1] Load model weights from models/{model_type}_model.pth.
    These are the HuggingFace pretrained weights downloaded by inspect_hf_models.py.
    """
    weight_file = os.path.join(
        os.path.dirname(__file__), '..', 'models', f'{model_type}_model.pth'
    )
    weight_file = os.path.abspath(weight_file)

    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"Model weights not found: {weight_file}\n"
            f"Run first:  python models/inspect_hf_models.py"
        )

    model = MODEL_CLASS[model_type]()
    state = torch.load(weight_file, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded local HF weights from {weight_file}")
    return model


def simulate_local_training(model: nn.Module) -> nn.Module:
    """
    [2] Mimic one round of local SGD by nudging weights with small noise.
    REPLACE THIS with real PyTorch training on your hospital's local dataset.
    """
    print("  Simulating local training…")
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.01
    return model


def compute_delta(original_state: dict, updated_state: dict) -> dict:
    """[3] Per-layer weight delta: updated - original."""
    return {
        name: updated_state[name] - original_state[name]
        for name in original_state
        if name in updated_state
    }


def apply_differential_privacy(delta: dict, clip_value: float, noise_scale: float) -> dict:
    """
    [4] DP via global sensitivity clipping + Gaussian noise.

    Step 1 — CLIPPING:
      Compute global L2 norm across all layers.
      Scale = min(1.0, clip / (norm + eps)).  Multiply every tensor by scale.

    Step 2 — NOISE:
      Add Gaussian N(0, noise_scale) to every tensor.
    """
    all_tensors = torch.cat([v.flatten() for v in delta.values()])
    global_norm = all_tensors.norm().item()
    scale       = min(1.0, clip_value / (global_norm + 1e-8))
    print(f"  Global L2 norm: {global_norm:.4f}  |  clip scale: {scale:.4f}")

    clipped = {k: v * scale for k, v in delta.items()}
    noisy   = {k: v + torch.randn_like(v) * noise_scale for k, v in clipped.items()}
    print(f"  Gaussian noise added (scale={noise_scale})")
    return noisy


def upload_to_ipfs(delta: dict) -> str:
    """[5] Save delta to a temp file and upload to IPFS. Returns the CID."""
    import ipfshttpclient

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(delta, tmp_path)

        try:
            client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        except Exception:
            raise RuntimeError(
                "Cannot connect to IPFS. Make sure IPFS daemon is running: ipfs daemon"
            )

        result = client.add(tmp_path)
        client.close()
        cid = result['Hash']
        print(f"  Uploaded to IPFS — CID: {cid}")
        return cid
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def submit_to_blockchain(sender, model_type, round_num, cid,
                         clip_value, noise_scale, server_url) -> dict:
    """[6] POST the CID + metadata to the Fabric REST server."""
    payload = {
        'sender':     sender,
        'modelType':  model_type,
        'round':      round_num,
        'ipfsCID':    cid,
        'clipValue':  clip_value,
        'noiseScale': noise_scale,
    }
    try:
        resp = requests.post(f"{server_url}/api/updates", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach REST API at {server_url}. "
            "Make sure server is running: node server.js"
        )

    resp.raise_for_status()
    result = resp.json()

    if result.get('aggregating'):
        print(f"  Aggregation triggered — MIN_CLIENTS ({result.get('minClients')}) reached!")
    else:
        remaining = result.get('minClients', 0) - result.get('updateCount', 0)
        print(f"  Waiting for {remaining} more hospital(s) before aggregation.")

    return result

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--sender',      required=True,
                        help='Hospital identifier, e.g. Hospital1')
    parser.add_argument('--model',       required=True, choices=['covid', 'skin'],
                        help='Model type')
    parser.add_argument('--round',       required=True, type=int,
                        help='FL round number')
    parser.add_argument('--clip',        default=1.0, type=float,
                        help='DP gradient clip value (default 1.0)')
    parser.add_argument('--noise',       default=0.1, type=float,
                        help='DP noise scale (default 0.1)')
    parser.add_argument('--server',      default='http://localhost:3000',
                        help='REST server URL')
    parser.add_argument('--pull-global', action='store_true',
                        help='Pull latest global model from blockchain before training '
                             '(recommended for rounds > 1)')
    args = parser.parse_args()

    sep = '=' * 60
    print(f"\n{sep}")
    print("  Federated Learning Client")
    print(f"  Sender      : {args.sender}")
    print(f"  Model       : {args.model.upper()}")
    print(f"  Round       : {args.round}")
    print(f"  Clip / Noise: {args.clip} / {args.noise}")
    print(f"  Server      : {args.server}")
    print(f"  Pull global : {args.pull_global}")
    print(sep)

    # [0] Pull global model (if requested and available)
    base_model = None
    if args.pull_global:
        print("\n[0/6] Pulling latest global model from blockchain…")
        base_model = pull_global_model(args.model, args.server)

    # [1] Load starting weights
    print("\n[1/6] Loading model…")
    if base_model is not None:
        model = base_model
        print("  Using pulled global model as base.")
    else:
        model = load_local_model(args.model)

    original_state = {name: param.data.clone() for name, param in model.named_parameters()}

    # [2] Local training
    print("\n[2/6] Local training…")
    model = simulate_local_training(model)
    updated_state = {name: param.data for name, param in model.named_parameters()}

    # [3] Compute delta
    print("\n[3/6] Computing weight delta…")
    delta = compute_delta(original_state, updated_state)
    print(f"  Delta computed for {len(delta)} parameter tensors")

    # [4] Differential privacy
    print("\n[4/6] Applying differential privacy…")
    delta = apply_differential_privacy(delta, args.clip, args.noise)

    # [5] Upload delta to IPFS
    print("\n[5/6] Uploading delta to IPFS…")
    cid = upload_to_ipfs(delta)

    # [6] Store CID on blockchain
    print("\n[6/6] Storing CID on Fabric blockchain…")
    result = submit_to_blockchain(
        sender=args.sender,
        model_type=args.model,
        round_num=args.round,
        cid=cid,
        clip_value=args.clip,
        noise_scale=args.noise,
        server_url=args.server,
    )

    print(f"\n{sep}")
    print("  FL round complete!")
    print(f"  IPFS CID  : {cid}")
    print(f"  Update ID : {result.get('id', 'N/A')}")
    print(f"{sep}\n")


if __name__ == '__main__':
    main()
