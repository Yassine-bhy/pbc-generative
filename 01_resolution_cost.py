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
# # 01 — Resolution cost (PROTOCOL 4b)
#
# Same classifier, uncut training set, seeds 0–4, two input pipelines:
#
# - **native:** 360 → 224
# - **128:** 360 → 128 → 224
#
# The difference between the two measures the detail lost at 128×128.
# All numbers are on the validation set, at the epoch with the best validation
# macro-F1 (PROTOCOL 4). Test scores are computed at the end of the study (D5).

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("results") / "classifier"
CONDITIONS = ["native", "128"]
SEEDS = [0, 1, 2, 3, 4]

out = Path("results") / "resolution_cost"
out.mkdir(parents=True, exist_ok=True)


def run_dir(cond, seed):
    return RESULTS / f"input{cond}_nall_seed{seed}"


logs = {c: [np.genfromtxt(run_dir(c, s) / "training_log.csv", delimiter=",", names=True)
            for s in SEEDS] for c in CONDITIONS}
classes = [n[len("recall_"):] for n in logs["128"][0].dtype.names if n.startswith("recall_")]
best = {c: [d[np.argmax(d["valid_macro_f1"])] for d in logs[c]] for c in CONDITIONS}

# %% [markdown]
# ## 1. Scores at the best epoch
#
# Mean ± standard deviation over the 5 seeds. The last column is native minus 128,
# computed seed by seed, then averaged.

# %%
rows = ["valid_macro_f1", "valid_balanced_acc", "valid_mcc", "valid_acc"] + [f"recall_{k}" for k in classes]

print(f"{'':24}{'native':>18}{'128':>18}{'native - 128':>20}")
for r in rows:
    a = np.array([b[r] for b in best["native"]])
    b = np.array([b[r] for b in best["128"]])
    d = a - b
    print(f"{r:24}{a.mean():>10.4f} ± {a.std(ddof=1):.4f}"
          f"{b.mean():>10.4f} ± {b.std(ddof=1):.4f}"
          f"{d.mean():>+12.4f} ± {d.std(ddof=1):.4f}")

print()
for c in CONDITIONS:
    print(f"best epoch, {c:>6}:", [int(b["epoch"]) for b in best[c]])

# %% [markdown]
# ## 2. Training curves
#
# Mean over seeds (line) and range over seeds (band). Training loss is measured on
# augmented images, validation loss on clean ones, so the two are not strictly
# comparable in level. What matters is the shape: a validation loss that rises while
# training loss keeps falling is overfitting.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for c in CONDITIONS:
    epochs = logs[c][0]["epoch"]
    for ax, key in zip(axes, ["train_loss", "valid_loss", "valid_macro_f1"]):
        v = np.stack([d[key] for d in logs[c]])
        ax.plot(epochs, v.mean(0), label=c)
        ax.fill_between(epochs, v.min(0), v.max(0), alpha=0.2)
for ax, title in zip(axes, ["training loss", "validation loss", "validation macro-F1"]):
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(out / "curves.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Per-class recall over training
#
# Rare classes (basophil, lymphocyte) are the ones to watch.

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for ax, k in zip(axes.flat, classes):
    for c in CONDITIONS:
        epochs = logs[c][0]["epoch"]
        v = np.stack([d[f"recall_{k}"] for d in logs[c]])
        ax.plot(epochs, v.mean(0), label=c)
        ax.fill_between(epochs, v.min(0), v.max(0), alpha=0.2)
    ax.set_title(k)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
fig.savefig(out / "recall_per_class.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Confusion matrices at the best epoch
#
# Each row is normalised to sum to 1 (row = true class, column = predicted class),
# then averaged over seeds. The right panel is native minus 128: positive on the
# diagonal means native recognises that class better; positive off the diagonal
# means native makes that particular mistake more often.

# %%
cms = {c: np.mean([np.loadtxt(run_dir(c, s) / "confusion_best.csv", delimiter=",") for s in SEEDS], axis=0)
       for c in CONDITIONS}
norm = {c: cms[c] / cms[c].sum(1, keepdims=True) for c in CONDITIONS}
panels = [(norm["native"], "native"), (norm["128"], "128"), (norm["native"] - norm["128"], "native - 128")]

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, (m, title) in zip(axes, panels):
    diff = title == "native - 128"
    lim = np.abs(m).max() if diff else 1
    ax.imshow(m, cmap="RdBu" if diff else "Blues", vmin=-lim if diff else 0, vmax=lim)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{m[i, j]:+.3f}" if diff else f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
plt.tight_layout()
fig.savefig(out / "confusion.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Findings

# %% [markdown]
# **Resolution cost.** At full training data, the 128×128 bottleneck has no measurable
# cost. Validation macro-F1 is 0.9935 ± 0.0011 (native) and 0.9933 ± 0.0011 (128); the
# paired difference is +0.0002 ± 0.0020 over 5 seeds (standard error 0.0009). Per-class
# differences go in both directions and amount to a few images: basophil recall 1.000 vs
# 0.998 corresponds to one misclassified basophil in one seed.
#
# **Generalisation.** Training loss falls to about 0.002; validation loss plateaus near
# 0.035 from epoch 18 and does not rise, so no overfitting is visible within 30 epochs.
# Best epochs fall between 20 and 30, where scores differ only in the fourth decimal.
#
# **Residual errors** are concentrated between ig, neutrophil and monocyte, and from
# monocyte to lymphocyte: biologically adjacent cell types (metamyelocytes mature into
# band neutrophils).
#
# **Ceiling.** With all training data the classifier reaches 99.3% macro-F1 and 100%
# recall on both rare classes in the native condition. There is no room for synthetic
# data to help at this level; the effect of synthetic data can only be studied under
# induced imbalance (PROTOCOL 2).
#
# **Limitation.** The cost was measured at full data. It may be larger when rare classes
# have few examples, where fine detail could matter more.

# %%
