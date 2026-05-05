#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
HuggingFace Model Inspector
============================
Downloads both real model weights from HuggingFace, prints every layer
name + shape, and auto-generates matching PyTorch architecture code.

Run this FIRST -- before any FL pipeline steps.

Usage:
    pip install huggingface_hub
    python models/inspect_hf_models.py

Outputs:
    models/covid_model.pth       ← real pretrained weights
    models/skin_model.pth        ← real pretrained weights
    models/generated_archs.py    ← auto-generated PyTorch classes to paste
                                    into client/fl_client.py
"""

import os
import shutil
import tempfile

import torch
from huggingface_hub import hf_hub_download, list_repo_files

# ─── Config ────────────────────────────────────────────────────────────────────

HF_REPOS = {
    'covid': 'sanjulamaduranga/BFL_Healthcare_covid_19',
    'skin':  'sanjulamaduranga/BFL_Healthcare_skincancer',
}

NUM_CLASSES = {
    'covid': 3,
    'skin':  7,
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Helpers ───────────────────────────────────────────────────────────────────

def find_pth_file(repo_id: str) -> str:
    """
    Return the best .pth filename in the repo.
    Prefers 'global_model.pth' (aggregated base) over client_model_*.pth.
    Falls back to the first .pth found.
    """
    files = list(list_repo_files(repo_id))
    pth_files = [f for f in files if f.endswith('.pth')]
    if not pth_files:
        raise FileNotFoundError(
            f"No .pth file in {repo_id}.\nAll files: {files}"
        )
    # Prefer the global/base model if available
    preferred = ['global_model.pth', 'skin_model.pth', 'model.pth']
    for name in preferred:
        if name in pth_files:
            print(f"  Using preferred file: {name}")
            return name
    if len(pth_files) > 1:
        print(f"  Multiple .pth files found: {pth_files}")
        print(f"  Using: {pth_files[0]}")
    return pth_files[0]


def download_model(name: str, repo_id: str) -> dict:
    """Download .pth from HuggingFace and save as models/{name}_model.pth."""
    filename = find_pth_file(repo_id)
    print(f"  Downloading '{filename}' from {repo_id} ...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False,
        )
        dest = os.path.join(OUT_DIR, f'{name}_model.pth')
        shutil.copy2(local, dest)

    state = torch.load(dest, map_location='cpu', weights_only=False)
    print(f"  Saved -> {dest}")
    return state


def inspect(state: dict, name: str):
    """Print every layer key + tensor shape."""
    print(f"\n  Layers ({len(state)} parameters):")
    for key, tensor in state.items():
        print(f"    {key:<55} shape = {list(tensor.shape)}")


# ─── Architecture code generator ───────────────────────────────────────────────

def _shape_to_conv(weight_shape) -> str:
    out_ch, in_ch, kH, kW = weight_shape
    pad = (kH - 1) // 2
    return f"nn.Conv2d({in_ch}, {out_ch}, {kH}, padding={pad})"


def _shape_to_linear(weight_shape) -> str:
    out_f, in_f = weight_shape
    return f"nn.Linear({in_f}, {out_f})"


def generate_arch_code(name: str, state: dict, num_classes: int) -> str:
    """
    Attempt to reconstruct a sequential PyTorch model from state_dict keys.
    Works for simple sequential Conv->ReLU->Pool->Linear architectures.
    The generated code is a STARTING POINT -- verify against the printed layers.
    """
    class_name = f"{name.title()}CNN"
    lines = [
        f"class {class_name}(nn.Module):",
        f"    # Auto-generated from state_dict -- verify layer names match exactly",
        f"    def __init__(self, num_classes={num_classes}):",
        "        super().__init__()",
    ]

    feature_layers = []
    classifier_layers = []

    prev_key_base = None
    for key, tensor in state.items():
        parts = key.split('.')
        layer_base = '.'.join(parts[:-1])  # e.g. "features.0"
        param_type = parts[-1]            # "weight" or "bias"

        if param_type != 'weight':
            continue
        if layer_base == prev_key_base:
            continue
        prev_key_base = layer_base

        shape = list(tensor.shape)
        is_conv   = len(shape) == 4
        is_linear = len(shape) == 2

        # Decide features vs classifier bucket by key prefix
        bucket = feature_layers if (
            'features' in layer_base or 'conv' in layer_base.lower()
        ) else classifier_layers

        if is_conv:
            bucket.append(f"            {_shape_to_conv(shape)}, nn.ReLU(),")
        elif is_linear:
            bucket.append(f"            {_shape_to_linear(shape)}, nn.ReLU(),")

    # Remove trailing ReLU from last classifier layer
    if classifier_layers:
        last = classifier_layers[-1]
        classifier_layers[-1] = last.replace(", nn.ReLU()", "")

    if feature_layers:
        lines.append("        self.features = nn.Sequential(")
        lines.extend(feature_layers)
        lines.append("        )")
    if classifier_layers:
        lines.append("        self.classifier = nn.Sequential(")
        lines.extend(classifier_layers)
        lines.append("        )")

    # Try to infer flatten size from last conv weight
    conv_keys = [k for k in state if 'weight' in k and len(state[k].shape) == 4]
    if conv_keys:
        last_conv_shape = list(state[conv_keys[-1]].shape)
        out_ch = last_conv_shape[0]
        lines.append(f"        # NOTE: update _flat_size if your input is not 224x224")
        lines.append(f"        # After pooling, spatial dim depends on MaxPool count")

    lines += [
        "",
        "    def forward(self, x):",
        "        x = self.features(x)",
        "        x = x.view(x.size(0), -1)",
        "        return self.classifier(x)",
    ]
    return '\n'.join(lines)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    sep = '=' * 65
    arch_blocks = []

    for name, repo_id in HF_REPOS.items():
        print(f"\n{sep}")
        print(f"  {name.upper()} MODEL  --  {repo_id}")
        print(sep)

        try:
            state = download_model(name, repo_id)
        except Exception as exc:
            msg = str(exc)
            if '401' in msg or 'gated' in msg.lower() or 'restricted' in msg.lower():
                print(f"  AUTH ERROR: repo is gated -- login first:")
                print(f"    huggingface-cli login")
                print(f"  Or set environment variable:  HF_TOKEN=your_token")
            else:
                print(f"  ERROR: {exc}")
            continue

        inspect(state, name)

        code = generate_arch_code(name, state, NUM_CLASSES[name])
        arch_blocks.append(f"# ── {name.upper()} ──\n{code}")

        print(f"\n  Auto-generated class:\n")
        print(code)

    # Write generated architectures to a helper file
    out_path = os.path.join(OUT_DIR, 'generated_archs.py')
    header = (
        "# Auto-generated by inspect_hf_models.py\n"
        "# Paste these classes into client/fl_client.py and verify\n"
        "# layer names match your state_dict keys exactly.\n\n"
        "import torch.nn as nn\n\n"
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n\n'.join(arch_blocks) + '\n')

    print(f"\n{sep}")
    print(f"  generated_archs.py written -> {out_path}")
    print(f"  Next steps:")
    print(f"    1. Check the printed layer names above")
    print(f"    2. Copy classes from generated_archs.py into client/fl_client.py")
    print(f"    3. Verify the flatten size (spatial dim after all MaxPool layers)")
    print(f"    4. Run the FL pipeline\n")


if __name__ == '__main__':
    main()
