#!/usr/bin/env python3
"""
FedAvg Aggregator
=================
Auto-triggered by server.js when MIN_CLIENTS updates arrive for the same
round + model type.  Architecture-agnostic: works purely on tensor dicts.

Pipeline:
  [1] Query blockchain for all updates in this round + model type
  [2] Download each client delta from IPFS
  [3] FedAvg: equal-weight average across all client deltas
  [4] Load base model (previous global round OR HF pretrained weights)
  [5] Apply averaged delta → new global model weights
  [6] Upload global model to IPFS → get CID
  [7] Record global model CID on blockchain via REST server

Usage (called automatically by server.js, or manually):
    python server/aggregator.py --round 1 --model covid
    python server/aggregator.py --round 1 --model skin
"""

import argparse
import os
import sys
import tempfile

import ipfshttpclient
import requests
import torch

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DEFAULT_SERVER = os.environ.get('FL_SERVER_URL', 'http://localhost:3000')
IPFS_ADDR = os.environ.get('IPFS_ADDR', '/ip4/127.0.0.1/tcp/5001')

# ─── Blockchain helpers ────────────────────────────────────────────────────────

def fetch_round_updates(round_num: int, model_type: str, server_url: str) -> list:
    """Return all blockchain update records for this round + model type."""
    resp = requests.get(
        f"{server_url}/api/updates/round/{round_num}",
        timeout=30,
    )
    resp.raise_for_status()
    all_updates = resp.json().get('updates', [])
    return [u for u in all_updates if u['modelType'] == model_type]


def record_global_model(round_num: int, model_type: str, cid: str,
                        client_count: int, server_url: str) -> dict:
    """POST the new global model CID to the REST server."""
    payload = {
        'round':       round_num,
        'modelType':   model_type,
        'ipfsCID':     cid,
        'clientCount': client_count,
    }
    resp = requests.post(
        f"{server_url}/api/global-model",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ─── IPFS helpers ──────────────────────────────────────────────────────────────

def _ipfs_connect():
    try:
        return ipfshttpclient.connect(IPFS_ADDR)
    except Exception:
        raise RuntimeError(
            f"Cannot connect to IPFS at {IPFS_ADDR}. "
            "Make sure the IPFS daemon is running: ipfs daemon"
        )


def download_from_ipfs(cid: str) -> dict:
    """Download a .pt tensor dict from IPFS by CID."""
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
    """Save a state_dict to a temp file and upload to IPFS. Returns CID."""
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

# ─── FedAvg ────────────────────────────────────────────────────────────────────

def fedavg(deltas: list) -> dict:
    """
    Equal-weight FedAvg: for every parameter key, stack all client tensors
    and take the mean.  Architecture-agnostic — works on any state_dict shape.
    """
    keys = set(deltas[0].keys())
    avg = {}
    for key in keys:
        stacked = torch.stack([d[key].float() for d in deltas if key in d])
        avg[key] = stacked.mean(dim=0)
    return avg

# ─── Base model ────────────────────────────────────────────────────────────────

def load_base_model(model_type: str, round_num: int, server_url: str) -> dict:
    """
    Load the starting weights for this aggregation round.

    Priority:
      1. Previous global model (round N-1) from blockchain → IPFS
      2. HuggingFace pretrained weights (models/{model_type}_model.pth)

    The base model + averaged delta = new global model.
    """
    if round_num > 1:
        try:
            resp = requests.get(
                f"{server_url}/api/global-model/{model_type}/latest",
                timeout=15,
            )
            if resp.ok:
                cid = resp.json().get('ipfsCID')
                if cid:
                    print(f"  Using previous global model (CID: {cid})")
                    return download_from_ipfs(cid)
        except Exception as exc:
            print(f"  Warning — could not fetch previous global model: {exc}")

    base_path = os.path.abspath(
        os.path.join(MODELS_DIR, f'{model_type}_model.pth')
    )
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Base model not found: {base_path}\n"
            f"Run first:  python models/inspect_hf_models.py"
        )
    print(f"  Using HF pretrained base: {base_path}")
    return torch.load(base_path, map_location='cpu', weights_only=False)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FedAvg Aggregator')
    parser.add_argument('--round',  required=True, type=int,
                        help='FL round number to aggregate')
    parser.add_argument('--model',  required=True, choices=['covid', 'skin'],
                        help='Model type')
    parser.add_argument('--server', default=DEFAULT_SERVER,
                        help='REST server URL (default: http://localhost:3000)')
    args = parser.parse_args()

    sep = '=' * 60
    print(f"\n{sep}")
    print(f"  FedAvg Aggregator")
    print(f"  Model : {args.model.upper()}  |  Round : {args.round}")
    print(f"  Server: {args.server}")
    print(sep)

    # [1] Fetch update records
    print(f"\n[1/7] Fetching round {args.round} updates for '{args.model}'...")
    updates = fetch_round_updates(args.round, args.model, args.server)
    if not updates:
        print("  No updates found — nothing to aggregate. Exiting.")
        sys.exit(0)
    print(f"  Found {len(updates)} client update(s):")
    for u in updates:
        print(f"    {u['sender']:20s}  CID: {u['ipfsCID']}")

    # [2] Download all deltas from IPFS
    print(f"\n[2/7] Downloading deltas from IPFS...")
    deltas = []
    for u in updates:
        print(f"  Downloading delta — sender: {u['sender']}")
        delta = download_from_ipfs(u['ipfsCID'])
        deltas.append(delta)
        print(f"    {len(delta)} parameter tensors loaded")

    # [3] FedAvg
    print(f"\n[3/7] Running FedAvg over {len(deltas)} client(s)...")
    avg_delta = fedavg(deltas)
    print(f"  Averaged {len(avg_delta)} parameter tensors (equal weighting)")

    # [4] Load base model
    print(f"\n[4/7] Loading base model for round {args.round}...")
    base = load_base_model(args.model, args.round, args.server)
    print(f"  Base model has {len(base)} parameter tensors")

    # [5] Apply averaged delta → global model
    print(f"\n[5/7] Applying averaged delta to base model...")
    global_weights = {}
    missing_in_delta = []
    for key in base:
        if key in avg_delta:
            global_weights[key] = base[key] + avg_delta[key]
        else:
            global_weights[key] = base[key]
            missing_in_delta.append(key)

    if missing_in_delta:
        print(f"  Warning: {len(missing_in_delta)} key(s) not in delta "
              f"(kept base values): {missing_in_delta[:3]}{'...' if len(missing_in_delta)>3 else ''}")
    print(f"  Global model assembled ({len(global_weights)} tensors)")

    # [6] Upload global model to IPFS
    print(f"\n[6/7] Uploading global model to IPFS...")
    global_cid = upload_to_ipfs(global_weights)
    print(f"  Global model CID: {global_cid}")

    # [7] Record on blockchain
    print(f"\n[7/7] Recording global model CID on blockchain...")
    result = record_global_model(
        round_num=args.round,
        model_type=args.model,
        cid=global_cid,
        client_count=len(updates),
        server_url=args.server,
    )
    print(f"  Recorded: {result.get('id', 'N/A')}")

    print(f"\n{sep}")
    print(f"  Aggregation complete!")
    print(f"  Model type       : {args.model.upper()}")
    print(f"  Round            : {args.round}")
    print(f"  Clients merged   : {len(updates)}")
    print(f"  Global model CID : {global_cid}")
    print(f"  Hospitals can now pull the new global model via:")
    print(f"    python client/fl_client.py --pull-global --model {args.model} ...")
    print(f"{sep}\n")


if __name__ == '__main__':
    main()
