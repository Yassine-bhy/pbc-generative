# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00 — Data audit: PBC Barcelona
#
# Purpose: describe the dataset exactly as received, find problems in it, and fix
# them by rules stated in advance, **before any model is trained**.
#
# Everything here is deterministic (it reads and counts files; no randomness, no
# training), so re-running this notebook reproduces every number below.
#
# **Note on the test set.** This notebook reads the test files to count them and
# to check for duplicates. That is not evaluation: no model is scored on the test
# set and nothing is tuned from it. The rule "test is used once, at the end"
# concerns model evaluation and is untouched.
#
# **Output:** `excluded.txt`, the list of training images removed by the
# duplicate rule (section 5). Decisions that follow from this audit are frozen
# in `PROTOCOL.md`.

# %%
from pathlib import Path
from collections import Counter
import hashlib
import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path.home() / "Globule" / "barcelona"
SPLITS = ["train", "valid", "test"]
CLASSES = sorted(p.name for p in (ROOT / "train").iterdir() if p.is_dir())
print(CLASSES)

# %% [markdown]
# ## 1. Structure and counts
#
# For each split: number of images per class, and the share of the split each
# class represents. If the shares are the same in all three splits, the split
# was *stratified* (done class by class).

# %%
counts = {s: {c: len(list((ROOT / s / c).glob("*.jpg"))) for c in CLASSES} for s in SPLITS}
totals = {s: sum(counts[s].values()) for s in SPLITS}

print(f"{'class':14}" + "".join(f"{s:>8}{'%':>7}" for s in SPLITS))
for c in CLASSES:
    print(f"{c:14}" + "".join(f"{counts[s][c]:>8}{100 * counts[s][c] / totals[s]:>7.1f}" for s in SPLITS))
print(f"{'total':14}" + "".join(f"{totals[s]:>8}{100 * totals[s] / sum(totals.values()):>7.1f}" for s in SPLITS))

largest = max(counts["train"].values())
smallest = min(counts["train"].values())
print(f"\nimbalance ratio (train): {largest} / {smallest} = {largest / smallest:.2f}")

# %% [markdown]
# ## 2. Formats, sizes, colour mode

# %%
for s in SPLITS:
    found = Counter()
    for f in (ROOT / s).rglob("*"):
        if f.is_file():
            with Image.open(f) as im:
                found[(f.suffix, im.size, im.mode)] += 1
    print(s, dict(found))

# %% [markdown]
# ## 3. Subtypes
#
# The file-name prefix (before `_`) is a cell subtype code. Classes made of
# several subtypes need care when subsampled: a small random draw can
# under-represent a subtype.

# %%
for s in SPLITS:
    print(f"== {s}")
    for c in CLASSES:
        prefixes = Counter(f.name.split("_")[0] for f in (ROOT / s / c).glob("*.jpg"))
        print(f"   {c:14}", dict(prefixes))

# %% [markdown]
# ## 4. Exact duplicates
#
# Two files are exact duplicates if their bytes are identical (same md5 hash).
# Three kinds, each a different problem:
#
# - **within-split**: same class, same split. The image counts double in training.
# - **cross-split**: same class, different splits. Leakage: the model is evaluated
#   on an image it trained on.
# - **label-conflict**: same image, two classes. Label noise: at least one label is wrong.
#
# Exact duplicates only. Near-duplicates (same cell, slightly different crop) need
# a feature-space check, done later.

# %%
groups = {}
for f in sorted(ROOT.rglob("*.jpg")):
    groups.setdefault(hashlib.md5(f.read_bytes()).hexdigest(), []).append(f)
dups = [v for v in groups.values() if len(v) > 1]


def kind(v):
    if len({f.parent.name for f in v}) > 1:
        return "label-conflict"
    if len({f.parts[-3] for f in v}) > 1:
        return "cross-split"
    return "within-split"


print(len(dups), "duplicate groups:", dict(Counter(kind(v) for v in dups)))
for v in sorted(dups, key=kind):
    print(f"{kind(v):15}", [str(f.relative_to(ROOT)) for f in v])

# %% [markdown]
# ## 5. Exclusion rule
#
# This rule was written after seeing the first 10 of the 17 duplicate groups, and
# before seeing the full list. It depends only on the *kind* of duplicate, never on
# any model result.
#
# 1. Valid and test are never modified.
# 2. A label conflict, or a group with any copy outside train: every train copy is removed.
# 3. Same-class duplicates entirely inside train: the first copy (alphabetical) is kept, the rest removed.
# 4. A group with no train copy (valid ↔ test) is left as it is and recorded as a known flaw.
#
# Nothing is deleted from disk. The removed files are listed in `excluded.txt`,
# which `PBCDataset` reads.

# %%
excluded = []
for v in dups:
    train = [f for f in v if f.parts[-3] == "train"]
    if kind(v) == "within-split" and len(train) == len(v):
        excluded += train[1:]
    else:
        excluded += train

lines = sorted(str(f.relative_to(ROOT)) for f in excluded)
Path("excluded.txt").write_text("\n".join(lines) + "\n")
print(len(lines), "training images excluded")
print(dict(Counter(l.split("/")[1] for l in lines)))

for v in dups:
    if not any(f.parts[-3] == "train" for f in v):
        print("known flaw, kept:", [str(f.relative_to(ROOT)) for f in v])

assert len(lines) == 17

# %% [markdown]
# ## 6. Clean counts
#
# Data is loaded through `PBCDataset` (from `data.py`), which applies `excluded.txt`.
# Train must lose exactly 17 images; valid and test must be unchanged.

# %%
from data import PBCDataset

for s in SPLITS:
    ds = PBCDataset(ROOT, s, 128)
    print(f"{s:6} {len(ds):>6}", dict(zip(ds.classes, torch.bincount(ds.labels).tolist())))
    if s == "train":
        small = ds

assert len(small) == totals["train"] - 17

# %% [markdown]
# ## 7. Visual check: native resolution vs 128×128 vs 64×64
#
# One cell per class, at native resolution (360×360), at 128×128 and at 64×64.
# Purpose: check by eye whether the features that separate classes (granules,
# nucleus shape) remain visible. This check decided the working resolution (PROTOCOL D4).

# %%
out = Path("results") / "data_audit"
out.mkdir(parents=True, exist_ok=True)

sizes = [360, 128, 64]
fig, axes = plt.subplots(len(sizes), len(CLASSES), figsize=(2 * len(CLASSES), 2.2 * len(sizes)))
for c, cls in enumerate(small.classes):
    i = (small.labels == c).nonzero()[0].item()
    full = v2.functional.center_crop(read_image(str(ROOT / "train" / cls / small.names[i])), 360)
    for r, s in enumerate(sizes):
        img = v2.functional.resize(full, [s, s], antialias=True)
        axes[r, c].imshow(img.permute(1, 2, 0))
        axes[r, c].axis("off")
    axes[0, c].set_title(cls, fontsize=9)
for r, s in enumerate(sizes):
    axes[r, 0].text(-0.15, 0.5, str(s), transform=axes[r, 0].transAxes, rotation=90, va="center")
fig.savefig(out / "resolution_check.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Findings
#
# **Format.** 16,891 JPEG images, all 360×363, RGB. 8 classes.
#
# **Split.** Train / valid / test = 14,441 / 1,605 / 845, i.e. 85.5 / 9.5 / 5 %.
# Class shares are identical across splits, so the split is stratified. It was
# done image by image: cells from the same blood smear can appear in several splits.
#
# **Imbalance.** Natural imbalance is mild: largest / smallest = 2803 / 1038 ≈ 2.7
# (neutrophil / lymphocyte). Imbalance must therefore be *induced* by subsampling.
#
# **Subtypes.** `ig` = metamyelocytes (`mmy`), myelocytes (`my`), promyelocytes
# (`pmy`), unequal. `neutrophil` = band (`bne`) and segmented (`sne`), about half
# each. Every other class has a single subtype. The number after the prefix is an
# arbitrary ID: slide-level grouping is **not recoverable** from the release, so
# leakage must be checked by image similarity.
#
# **Duplicates.** 17 exact-duplicate groups: 11 within-split, 5 cross-split
# (1 train↔test, 3 train↔valid, 1 valid↔test), 1 label conflict
# (neutrophil / eosinophil). 17 training images excluded; train = 14,424.
# Known flaw kept: one `ig` image present in both valid and test.
#
# **Rare classes.** Basophil (1035 after cleaning) and lymphocyte (1038) are the two
# smallest classes in train, and both are single-subtype, so subsampling them
# reduces the number of examples without removing a subtype.
#
# **Precision of the test set.** The rare classes have 61 test images each. Recall
# moves in steps of $1/61 \approx 1.6\%$, and if the true recall is $p = 0.9$ its
# standard error is
#
# $$
# \begin{aligned}
# \sqrt{\frac{p(1-p)}{n}}
# &= \sqrt{\frac{0.9 \times 0.1}{61}} \\
# &= \sqrt{0.001475} \\
# &\approx 0.038
# \end{aligned}
# $$
#
# so a single run cannot detect a change of a few points in rare-class recall.
# Results must be averaged over several seeds and tested statistically.
#
# **Visual check (section 7).** At 64×64, shape, size, colour and nucleus outline
# survive, so eosinophils, monocytes, platelets and erythroblasts remain easy to
# recognise. Texture does not: the dark granules covering the basophil disappear,
# and it becomes a smooth dark blob close to the lymphocyte and the ig. Since
# basophil is a rare class and granules identify it, 64×64 loses the feature that
# defines one of the two rare classes. At 128×128 the granule texture is still visible.
#
# **Resolution.** Every image, real or synthetic, passes through 128×128 so that
# both are on equal footing. The cost of this bottleneck is measured by the first
# baseline experiment (same classifier at native resolution and at 128×128).
# Fallback to 64×64 is governed by a numerical rule fixed in PROTOCOL D4.
