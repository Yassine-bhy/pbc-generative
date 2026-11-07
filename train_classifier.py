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

# %% [markdown]
# # Downstream classifier (PROTOCOL 4, 4b, 5, 6)
#
# One script for every classifier run of the study. The classifier itself is identical
# everywhere (PROTOCOL 4); only the training data and the loss change.
#
# | arm          | runs                                    | configurations                  |
# |--------------|-----------------------------------------|---------------------------------|
# | `resolution` | 4b, uncut set, seeds 0–4                | `--input native` or `--input 128` |
# | `baseline`   | N ∈ {50, 150, 400} × replicates 0–9     | none, weighted, oversampling    |
# | `cvae`       | same                                    | S ∈ {500, 1000, 2000} synthetic |
# | `cgan`       | same                                    | same                            |
#
# Examples:
#
#     python train_classifier.py baseline --device cuda:0 --replicates 0 1 2 3 4
#     python train_classifier.py cgan --device cuda:1 --n 50
#
# A finished run leaves a `done` file and is skipped next time, so after a crash the same
# command just continues. The test set is never loaded here (D5).

# %%
import argparse
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2

from data import ROOT, MEAN, STD, PBCDataset, SyntheticDataset, rare_classes, subsample, class_weights
from metrics import confusion, metrics

# %%
# ---------- settings ----------
parser = argparse.ArgumentParser()
parser.add_argument("arm", choices=["resolution", "baseline", "cvae", "cgan"])
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--input", choices=["native", "128"], default="128", help="resolution arm only")
parser.add_argument("--n", type=int, nargs="+", default=[50, 150, 400])
parser.add_argument("--replicates", type=int, nargs="+", default=list(range(10)))
args = parser.parse_args()

ARM          = args.arm
INPUT        = args.input if ARM == "resolution" else "128"   # "native": 360 -> 224   "128": 360 -> 128 -> 224
EPOCHS       = 30
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 1e-4
WORKERS      = 8

CONFIGS = {
    "resolution": ["none"],
    "baseline":   ["none", "weighted", "oversampling"],    # PROTOCOL 5
    "cvae":       [500, 1000, 2000],                        # PROTOCOL 6: synthetic images per rare class
    "cgan":       [500, 1000, 2000],
}

device = torch.device(args.device)

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
                         num_workers=WORKERS, pin_memory=True, persistent_workers=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------- list of runs: (name, seed, N, configuration) ----------
if ARM == "resolution":
    # same folder names as the first 4b runs, so 01_resolution_cost.py and the judge path still work
    runs = [(f"input{INPUT}_nall_seed{s}", s, 0, "none") for s in range(5)]
else:
    # replicate outermost: by morning, finished replicates are complete across N and configurations
    runs = [(f"{ARM}/n{n}_rep{r}/{cfg if isinstance(cfg, str) else f's{cfg}'}", r, n, cfg)
            for r in args.replicates for n in args.n for cfg in CONFIGS[ARM]]

print(f"{len(runs)} runs, arm {ARM}, device {device}")

# %%
for run_name, SEED, N, CFG in runs:
    model_dir   = Path("models")  / "classifier" / run_name
    results_dir = Path("results") / "classifier" / run_name
    if (results_dir / "done").exists():
        print("already done:", run_name)
        continue
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt     = model_dir   / "best_checkpoint.pt"
    log_file = results_dir / "training_log.csv"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ---------- training data ----------
    idx = subsample(train_set.labels, rare, N, SEED) if N else torch.arange(len(train_set))
    real_labels = train_set.labels[idx]
    data = Subset(train_set, idx.tolist())
    n_synth = 0
    if ARM in ("cvae", "cgan"):
        synth_dir = Path("synthetic") / ARM / f"n{N}_rep{SEED}"
        if not (synth_dir / "done").exists():
            print(f"skipped, synthetic set not ready: {synth_dir}")
            continue
        synth = SyntheticDataset(synth_dir, classes, CFG, train_tf)
        data = ConcatDataset([data, synth])
        n_synth = CFG
        all_labels = torch.cat([real_labels, synth.labels])
    else:
        all_labels = real_labels
    counts = torch.bincount(all_labels, minlength=C)

    g = torch.Generator()
    g.manual_seed(SEED)
    if CFG == "oversampling":
        # each image drawn with probability ∝ 1 / its class count: every class equally often
        sampler = WeightedRandomSampler((1 / counts.double())[all_labels], num_samples=len(all_labels),
                                        replacement=True, generator=g)
        trainloader = DataLoader(data, batch_size=BATCH_SIZE, sampler=sampler,
                                 num_workers=WORKERS, worker_init_fn=seed_worker,
                                 pin_memory=True, persistent_workers=True)
    else:
        trainloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=WORKERS, generator=g, worker_init_fn=seed_worker,
                                 pin_memory=True, persistent_workers=True)
    weight = class_weights(all_labels, C).to(device) if CFG == "weighted" else None

    # ---------- write down what this run was ----------
    config = f"""Model      : ResNet-18, ImageNet weights
Started    : {datetime.now():%Y-%m-%d %H:%M:%S}
Device     : {device}

Arm              : {ARM}
Configuration    : {CFG if ARM not in ("cvae", "cgan") else f"{CFG} synthetic images per rare class"}
Input pipeline   : {"360 -> 224" if INPUT == "native" else "360 -> 128 -> 224"}
Rare classes     : {", ".join(classes[c] for c in rare)}
Images per rare  : {N if N else "all"}
Replicate / seed : {SEED}
Learning rate    : {LR}
Weight decay     : {WEIGHT_DECAY}
Batch size       : {BATCH_SIZE}
Epochs           : {EPOCHS}

Training images per class (real + synthetic):
""" + "\n".join(f"  {c:14}{k}" for c, k in zip(classes, counts.tolist())) + "\n"
    if weight is not None:
        config += "\nClass weights:\n" + "\n".join(f"  {c:14}{w:.4f}" for c, w in zip(classes, weight.tolist())) + "\n"
    (results_dir / "configuration.txt").write_text(config)
    (results_dir / "train_images.txt").write_text(
        "\n".join(f"{classes[train_set.labels[i]]}/{train_set.names[i]}" for i in idx.tolist()) + "\n")
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
                "valid_balanced_acc,valid_mcc," + ",".join(f"recall_{c}" for c in classes) + ","
                + ",".join(f"f1_{c}" for c in classes) + "\n")

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
            loss = F.cross_entropy(outputs, labels, weight=weight)
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
                va_loss += F.cross_entropy(outputs, labels, reduction="sum").item()   # unweighted: comparable across configurations
                va_n += len(labels)
                cm += confusion(outputs.argmax(1), labels, C).cpu()
        m = metrics(cm)

        row = [epoch + 1, lr, tr_loss / tr_n, tr_correct / tr_n, va_loss / va_n,
               m["accuracy"], m["macro_f1"], m["balanced_accuracy"], m["mcc"]] + m["recall"] + m["f1"]
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

    (results_dir / "done").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\nbest valid macro-F1 {best_f1:.6f}\n")
    del trainloader
