# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.transforms import v2

from data import PBCDataset, rare_classes, subsample
from metrics import confusion, metrics

# %%
# ---------- settings ----------
INPUT        = sys.argv[1]       # "native": 360 -> 224    "128": 360 -> 128 -> 224
DEVICE       = sys.argv[2]       # "cuda:0" or "cuda:1"
N            = 0                 # images kept per rare class, 0 = all
SEEDS        = [0, 1, 2, 3, 4]
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 1e-4

ROOT = Path.home() / "Globule" / "barcelona"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
device = torch.device(DEVICE)

# %%
size = 224 if INPUT == "native" else 128

train_tf = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomChoice([v2.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
    v2.ColorJitter(0.1, 0.1, 0.1, 0.02),
    v2.Resize(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])
eval_tf = v2.Compose([
    v2.ToImage(),
    v2.Resize(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])

train_set = PBCDataset(ROOT, "train", size, train_tf)
valid_set = PBCDataset(ROOT, "valid", size, eval_tf)
classes = train_set.classes
C = len(classes)
rare = rare_classes(train_set.labels)

validloader = DataLoader(valid_set, batch_size=128, shuffle=False,
                         num_workers=8, pin_memory=True, persistent_workers=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# %%
for SEED in SEEDS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ---------- where things go ----------
    run_name    = f"input{INPUT}_n{N if N else 'all'}_seed{SEED}"
    model_dir   = Path("models")  / "classifier" / run_name
    results_dir = Path("results") / "classifier" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt     = model_dir   / "best_checkpoint.pt"
    log_file = results_dir / "training_log.csv"

    # ---------- training subset ----------
    idx = subsample(train_set.labels, rare, N, SEED) if N else torch.arange(len(train_set))
    counts = torch.bincount(train_set.labels[idx], minlength=C).tolist()

    g = torch.Generator()
    g.manual_seed(SEED)
    trainloader = DataLoader(Subset(train_set, idx.tolist()), batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=8, generator=g, worker_init_fn=seed_worker,
                             pin_memory=True, persistent_workers=True)

    # ---------- write down what this run was ----------
    config = f"""Model      : ResNet-18, ImageNet weights
Started    : {datetime.now():%Y-%m-%d %H:%M:%S}
Device     : {device}

Input pipeline   : {"360 -> 224" if INPUT == "native" else "360 -> 128 -> 224"}
Rare classes     : {", ".join(classes[c] for c in rare)}
Images per rare  : {N if N else "all"}
Learning rate    : {LR}
Weight decay     : {WEIGHT_DECAY}
Batch size       : {BATCH_SIZE}
Epochs           : {EPOCHS}
Random seed      : {SEED}

Training images per class:
""" + "\n".join(f"  {c:14}{k}" for c, k in zip(classes, counts)) + "\n"
    (results_dir / "configuration.txt").write_text(config)
    print(config)

    # ---------- model ----------
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, C)
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    # ---------- training ----------
    with open(log_file, "w") as f:
        f.write("epoch,lr,train_loss,train_acc,valid_loss,valid_acc,valid_macro_f1,"
                "valid_balanced_acc,valid_mcc," + ",".join(f"recall_{c}" for c in classes) + "\n")

    print(f"{'epoch':>5} | {'train':>7} {'acc':>6} | {'valid':>7} {'acc':>6} {'F1':>6} |")
    best_f1 = -1.0

    for epoch in range(EPOCHS):
        t0 = time.time()
        lr = optimiser.param_groups[0]["lr"]

        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimiser.step()
            tr_loss += loss.item() * len(labels)
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_n += len(labels)
        scheduler.step()

        model.eval()
        va_loss, va_n = 0.0, 0
        cm = torch.zeros(C, C, dtype=torch.long)
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                va_loss += F.cross_entropy(outputs, labels, reduction="sum").item()
                va_n += len(labels)
                cm += confusion(outputs.argmax(1), labels, C).cpu()
        m = metrics(cm)

        row = [epoch + 1, lr, tr_loss / tr_n, tr_correct / tr_n, va_loss / va_n,
               m["accuracy"], m["macro_f1"], m["balanced_accuracy"], m["mcc"]] + m["recall"]
        with open(log_file, "a") as f:
            f.write(",".join(f"{v:.6g}" for v in row) + "\n")

        improved = m["macro_f1"] > best_f1
        if improved:
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), ckpt)
            np.savetxt(results_dir / "confusion_best.csv", cm.numpy(), fmt="%d",
                       delimiter=",", header=",".join(classes))

        print(f"{epoch + 1:>5} | {tr_loss / tr_n:7.4f} {tr_correct / tr_n:6.4f} "
              f"| {va_loss / va_n:7.4f} {m['accuracy']:6.4f} {m['macro_f1']:6.4f} | "
              f"{time.time() - t0:.0f}s{'  * saved' if improved else ''}", flush=True)
