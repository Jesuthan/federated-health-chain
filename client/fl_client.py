#!/usr/bin/env python3
"""
Federated Learning Client
=========================
Pipeline:
  [1] Load model weights (.pth)
  [2] Simulate local training
  [3] Compute weight delta (updated - original)
  [4] Apply differential privacy (global clip + Gaussian noise)
  [5] Upload delta to IPFS → get CID
  [6] POST CID to REST server → stored on Fabric blockchain
"""

import argparse
import os
import tempfile

import requests
import torch
import torch.nn as nn

# ─── CNN Architectures ─────────────────────────────────────────────────────────
# All architectures: input = 1-channel 64x64, output = 3 classes


class CNN2(nn.Module):
    """2-conv-layer baseline. Fast, lower accuracy."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # → 16x16
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


class CNN4(nn.Module):
    """4-conv-layer model. Best balance of speed and accuracy (default)."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # → 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # → 16x16
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


class CNN6(nn.Module):
    """6-conv-layer deep model. Highest capacity."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # → 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # → 16x16
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),                  # → still 16x16
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


ARCH_MAP = {'cnn2': CNN2, 'cnn4': CNN4, 'cnn6': CNN6}

# ─── Pipeline steps ────────────────────────────────────────────────────────────

def load_model(model_type: str, arch: str = 'cnn4') -> nn.Module:
    """
    Load model weights from ../models/{model_type}_model.pth.
    Raises FileNotFoundError if the checkpoint does not exist.
    Replace simulate_local_training() with real training on your dataset.
    """
    weight_file = os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_type}_model.pth')
    weight_file = os.path.abspath(weight_file)

    if not os.path.exists(weight_file):
        raise FileNotFoundError(
            f"Model weights not found: {weight_file}\n"
            f"Run: python models/generate_dummy_models.py"
        )

    model = ARCH_MAP[arch]()
    state = torch.load(weight_file, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded {arch.upper()} weights from {weight_file}")
    return model


def simulate_local_training(model: nn.Module) -> nn.Module:
    """
    Mimic one round of local SGD by nudging weights with small noise.
    Replace this with real PyTorch training on your local dataset.
    """
    print("  Simulating local training…")
    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.01
    return model


def compute_delta(original_model: nn.Module, updated_weights: dict) -> dict:
    """Compute per-layer weight delta: updated - original."""
    delta = {}
    for name, orig_param in original_model.named_parameters():
        delta[name] = updated_weights[name] - orig_param.data
    return delta


def apply_differential_privacy(delta: dict, clip_value: float, noise_scale: float) -> dict:
    """
    DP via global sensitivity clipping + Gaussian noise.

    Step 1 — CLIPPING:
      Flatten all layer tensors into one vector, compute global L2 norm,
      then scale = min(1.0, clip / (norm + eps)).
      Multiply every layer tensor by scale.

    Step 2 — NOISE:
      Add Gaussian noise N(0, noise_scale) to every layer tensor.
    """
    # Step 1: global L2 norm across all layers
    all_tensors = torch.cat([v.flatten() for v in delta.values()])
    global_norm = all_tensors.norm().item()
    scale = min(1.0, clip_value / (global_norm + 1e-8))
    print(f"  Global L2 norm: {global_norm:.4f}  |  clip scale: {scale:.4f}")

    clipped = {k: v * scale for k, v in delta.items()}

    # Step 2: add Gaussian noise
    noisy = {k: v + torch.randn_like(v) * noise_scale for k, v in clipped.items()}
    print(f"  Gaussian noise added (scale={noise_scale})")

    return noisy


def upload_to_ipfs(delta: dict) -> str:
    """
    Save the delta dict to a temp file with torch.save, then upload to IPFS.
    Returns the IPFS CID string.
    Raises RuntimeError if the IPFS daemon is unreachable.
    """
    import ipfshttpclient

    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    try:
        torch.save(delta, tmp.name)
        tmp.close()

        try:
            client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
        except Exception:
            raise RuntimeError(
                "Cannot connect to IPFS. Make sure IPFS daemon is running: ipfs daemon"
            )

        result = client.add(tmp.name)
        client.close()
        cid = result['Hash']
        print(f"  Uploaded to IPFS — CID: {cid}")
        return cid
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def submit_to_blockchain(sender, model_type, round_num, cid, clip_value, noise_scale, server_url) -> dict:
    """POST the CID + metadata to the Fabric REST server."""
    payload = {
        'sender': sender,
        'modelType': model_type,
        'round': round_num,
        'ipfsCID': cid,
        'clipValue': clip_value,
        'noiseScale': noise_scale,
    }
    try:
        resp = requests.post(f"{server_url}/api/updates", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach REST API at {server_url}. Make sure server is running: node server.js"
        )

    resp.raise_for_status()
    return resp.json()

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--sender', required=True,                              help='Client identifier, e.g. Client1')
    parser.add_argument('--model',  required=True, choices=['covid', 'skin'],   help='Model type')
    parser.add_argument('--arch',   default='cnn4', choices=['cnn2','cnn4','cnn6'], help='CNN architecture (default: cnn4)')
    parser.add_argument('--round',  required=True, type=int,                    help='FL round number')
    parser.add_argument('--clip',   default=1.0,   type=float,                  help='DP gradient clip value (default 1.0)')
    parser.add_argument('--noise',  default=0.1,   type=float,                  help='DP noise scale (default 0.1)')
    parser.add_argument('--server', default='http://localhost:3000',            help='REST server URL')
    args = parser.parse_args()

    sep = '=' * 60
    print(f"\n{sep}")
    print("  Federated Learning Client")
    print(f"  Sender : {args.sender}")
    print(f"  Model  : {args.model}  |  Arch: {args.arch.upper()}")
    print(f"  Round  : {args.round}")
    print(f"  Clip   : {args.clip}  |  Noise: {args.noise}")
    print(f"  Server : {args.server}")
    print(sep)

    # [1] Load model
    print("\n[1/6] Loading model…")
    model = load_model(args.model, args.arch)
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}

    # [2] Local training
    print("\n[2/6] Local training…")
    model = simulate_local_training(model)
    updated_state = {name: param.data for name, param in model.named_parameters()}

    # [3] Compute delta
    print("\n[3/6] Computing weight delta…")
    delta = compute_delta(model, updated_state)
    # Re-derive delta from clones
    delta = {name: updated_state[name] - original_state[name] for name in original_state}

    # [4] Differential privacy
    print("\n[4/6] Applying differential privacy…")
    delta = apply_differential_privacy(delta, args.clip, args.noise)

    # [5] Upload to IPFS
    print("\n[5/6] Uploading to IPFS…")
    cid = upload_to_ipfs(delta)

    # [6] Store on blockchain
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
