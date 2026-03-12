#!/usr/bin/env python3
"""
Generate dummy model checkpoints for testing.
Replace these with real pretrained weights for production.

Usage:
    python generate_dummy_models.py
"""

import os
import torch
import torch.nn as nn


# CNN4 architecture (default in fl_client.py)
class CNN4(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


out_dir = os.path.dirname(os.path.abspath(__file__))

for name in ('covid_model.pth', 'skin_model.pth'):
    model = CNN4(num_classes=3)
    out_path = os.path.join(out_dir, name)
    torch.save(model.state_dict(), out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path}  ({size_kb:.1f} KB)")

print("\nDone — replace with real pretrained weights for production.")
