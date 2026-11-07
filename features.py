from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from data import network_input

JUDGE = Path("models") / "classifier" / "input128_nall_seed0" / "best_checkpoint.pt"


class Features(nn.Module):
    """phi (PROTOCOL 8): 512-d global-pool output of an ImageNet ResNet-18, never fine-tuned.
    Takes uint8 images at 128x128."""

    def __init__(self):
        super().__init__()
        self.net = models.resnet18(weights="IMAGENET1K_V1")
        self.net.fc = nn.Identity()
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.net(network_input(x))


def load_judge(num_classes, device):
    """The 4b classifier, input 128, seed 0 (PROTOCOL 6). Takes uint8 images at 128x128."""
    if not JUDGE.exists():
        raise FileNotFoundError(f"judge not found: {JUDGE} (run: python train_classifier.py resolution --input 128)")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net.load_state_dict(torch.load(JUDGE, map_location=device))
    return net.to(device).eval()


@torch.no_grad()
def apply(net, images, batch=256, preprocess=network_input):
    """Run `net` over a uint8 image tensor in batches; images can be on CPU or GPU."""
    device = next(net.parameters()).device
    out = []
    for x in images.split(batch):
        x = x.to(device)
        out.append(net(preprocess(x) if preprocess else x).cpu())
    return torch.cat(out)


def frechet_distance(f1, f2):
    """Fréchet distance between Gaussians fitted to two feature sets (N, d).
    ||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2)), with tr((S1 S2)^(1/2)) computed as the
    sum of square roots of the eigenvalues of S1^(1/2) S2 S1^(1/2) (symmetric, stable)."""
    f1, f2 = np.asarray(f1, np.float64), np.asarray(f2, np.float64)
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    w, V = np.linalg.eigh(s1)
    r1 = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(r1 @ s2 @ r1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)
