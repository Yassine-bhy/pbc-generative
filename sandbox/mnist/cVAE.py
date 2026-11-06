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

# %%
import struct
from os.path import join
import os
from pathlib import Path
from datetime import datetime
import textwrap
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# %%
# ----------------------------------------------------------------------
# 1. What is being trained, and with which hyper-parameters
# ----------------------------------------------------------------------
MODEL_KEY   = "cVAE"                              # folder name under models/ and results/
MODEL_TITLE = "Conditional Variational Autoencoder (cVAE)"    # readable name, used inside files

SEED         = 0
LATENT_DIM   = 64
BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 1e-3
LOSS         = "mse"      # "mse" or "bce"
RECON_WEIGHT = 8.0        # for mse: = 1/(2*sigma^2). for bce: leave at 1.0
TAG          = ""         # optional note, e.g. "3conv"
NUM_CLASSES = 10

# ----------------------------------------------------------------------
# 2. One single source of truth for the run identity.
#    Each entry is (short key for the folder name, readable label, value).
# ----------------------------------------------------------------------
PARAMS = [
    ("loss",   "Reconstruction loss",   LOSS),
    ("weight", "Reconstruction weight", RECON_WEIGHT),
    ("latent", "Latent dimension",      LATENT_DIM),
    ("lr",     "Learning rate",         LR),
    ("batch",  "Batch size",            BATCH_SIZE),
    ("epochs", "Epochs",                EPOCHS),
    ("seed",   "Random seed",           SEED),
]
if TAG:
    PARAMS.append(("tag", "Note", TAG))

# Expansions used only for humans, never in a path.
READABLE = {
    "bce": "binary cross-entropy",
    "mse": "mean squared error",
}


def compact(value):
    """Short form for a value inside a folder name: 0.001, 32, bce."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def readable(value):
    """Long form for a value inside a file: 'binary cross-entropy'."""
    return READABLE.get(value, compact(value))


RUN_DIR_NAME = "_".join(f"{key}={compact(value)}" for key, _, value in PARAMS)
RUN_TITLE    = " · ".join(f"{label}: {readable(value)}" for _, label, value in PARAMS)

# ----------------------------------------------------------------------
# 3. Create the folders and name the files
# ----------------------------------------------------------------------
MODEL_DIR   = Path("models")  / MODEL_KEY / RUN_DIR_NAME
RESULTS_DIR = Path("results") / MODEL_KEY / RUN_DIR_NAME
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CKPT        = MODEL_DIR   / "best_checkpoint.pt"
LOG         = RESULTS_DIR / "training_log.csv"
FIG_SAMPLES = RESULTS_DIR / "generated_samples.png"
FIG_CURVES  = RESULTS_DIR / "loss_curves.png"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# 4. A plain-language description of the run, written into both folders
# ----------------------------------------------------------------------
label_width = max(len(label) for _, label, _ in PARAMS)
config_text = "\n".join(
    [
        f"Model      : {MODEL_TITLE}",
        f"Started    : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Device     : {device}",
        "",
        "Parameters",
        "----------",
    ]
    + [f"{label:<{label_width}} : {readable(value)}" for _, label, value in PARAMS]
) + "\n"

for folder in (MODEL_DIR, RESULTS_DIR):
    (folder / "configuration.txt").write_text(config_text)

print(config_text)


def vae_loss(logits, target, mu, logvar):
    if LOSS == "mse":
        recon = ((torch.sigmoid(logits) - target) ** 2).flatten(1).sum(1).mean()
    else:
        recon = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none").flatten(1).sum(1).mean()
    kl = (0.5 * (logvar.exp() + mu**2 - 1 - logvar)).sum(1).mean()
    return RECON_WEIGHT * recon + kl, recon, kl


# %%
class MnistArrayDataset(Dataset):
    def __init__(self, images_filepath, labels_filepath, transform=None):
        self.transform = transform
        images, labels = self.read_images_labels(images_filepath, labels_filepath)
        self.images = images
        self.labels = torch.from_numpy(labels.astype("int64"))

    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = np.frombuffer(file.read(), dtype=np.uint8).copy()

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows, cols).copy()
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# %%
tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

batch_size = BATCH_SIZE

input_path = "./Data"
training_images_filepath = join(input_path, "train-images.idx3-ubyte")
training_labels_filepath = join(input_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels.idx1-ubyte")

train_view = MnistArrayDataset(training_images_filepath, training_labels_filepath, tf)
valid_view = MnistArrayDataset(training_images_filepath, training_labels_filepath, tf)
test_set = MnistArrayDataset(test_images_filepath, test_labels_filepath, tf)

n = len(train_view)
train_idx = list(range(0, n * 9 // 10))
val_idx = list(range(n * 9 // 10, n))

g = torch.Generator()
g.manual_seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


trainloader = DataLoader(train_view, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx),
                         num_workers=4, generator=g, worker_init_fn=seed_worker,
                         pin_memory=True, persistent_workers=True)

validloader = DataLoader(valid_view, batch_size=batch_size, sampler=val_idx,
                         num_workers=4, generator=g, worker_init_fn=seed_worker,
                         pin_memory=True, persistent_workers=True)

testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)


# %%
class cVAE(nn.Module):
    
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc = nn.Linear(num_classes, 28 * 28)
        self.fc_mu = nn.Linear(self.num_classes + 64 * 7 * 7, self.latent_dim)
        self.fc_logvar = nn.Linear(self.num_classes + 64 * 7 * 7, self.latent_dim)
        self.fc3 = nn.Linear(latent_dim + self.num_classes, 64 * 7 * 7)
        self.convtrans1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def onehot(self, y):
        return F.one_hot(y, num_classes = self.num_classes).float()
    
    def encode(self, x, y):
        ymap = self.fc(self.onehot(y))
        ymap = torch.unflatten(ymap, 1, [1, 28, 28])
        h = torch.cat([x, ymap], dim = 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, y):
        h = torch.cat([z, self.onehot(y)], dim = 1)
        h = self.fc3(h)
        h = torch.unflatten(h, 1, [64, 7, 7])
        h = F.relu(self.convtrans1(h))
        return self.convtrans2(h)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x, y):
        mu, _ = self.encode(x, y)
        return torch.sigmoid(self.decode(mu, y))

    @torch.no_grad()
    def sample(self, y, device):
        z = torch.randn(len(y), self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z, y.to(device)))


# %%
torch.manual_seed(SEED)
model = cVAE(LATENT_DIM, NUM_CLASSES).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=LR)

# %%
best_validation_loss = float("inf")

with open(LOG, "w") as f:
    f.write("epoch,train,train_recon,train_kl,valid,valid_recon,valid_kl\n")

print(MODEL_TITLE)
print(RUN_TITLE)
header = f"{'epoch':>5} | {'train':>7} {'recon':>7} {'kl':>6} | {'valid':>7} {'recon':>7} {'kl':>6} |"
print(header)
print("-" * len(header))


for epoch in range(EPOCHS):
    model.train()
    tr = [0.0, 0.0, 0.0]
    tr_n = 0
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        logits, mu, logvar = model(images, labels)
        loss, recon, kl = vae_loss(logits, images, mu, logvar)
        loss.backward()
        optimiser.step()
        bs = len(images)
        tr[0] += loss.item() * bs
        tr[1] += recon.item() * bs
        tr[2] += kl.item() * bs
        tr_n += bs

    model.eval()
    va = [0.0, 0.0, 0.0]
    va_n = 0
    with torch.no_grad():
        for images, labels in validloader:
            images = images.to(device)
            labels = labels.to(device)
            logits, mu, logvar = model(images, labels)
            loss, recon, kl = vae_loss(logits, images, mu, logvar)
            bs = len(images)
            va[0] += loss.item() * bs
            va[1] += recon.item() * bs
            va[2] += kl.item() * bs
            va_n += bs

    tr = [v / tr_n for v in tr]
    va = [v / va_n for v in va]

    improved = va[0] < best_validation_loss
    if improved:
        torch.save(model.state_dict(), CKPT)
        best_validation_loss = va[0]

    with open(LOG, "a") as f:
        f.write(f"{epoch+1}," + ",".join(f"{v:.4f}" for v in tr + va) + "\n")

    print(f"{epoch+1:>5} | {tr[0]:7.2f} {tr[1]:7.2f} {tr[2]:6.2f} "
          f"| {va[0]:7.2f} {va[1]:7.2f} {va[2]:6.2f} |"
          f"{'  * saved' if improved else ''}")


# %%
def imshow(ax, x):
    ax.imshow(x.detach().cpu().squeeze(0), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

FIG_TITLE = f"{MODEL_TITLE}\n{textwrap.fill(RUN_TITLE, 90)}"

model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

images, labels = next(iter(testloader))
images = images.to(device)
labels = labels.to(device)
recons = model.reconstruct(images, labels)
samples = model.sample(labels, device)

fig, axes = plt.subplots(3, 8, figsize=(12, 4.5))
for i in range(8):
    imshow(axes[0, i], images[i])
    imshow(axes[1, i], recons[i])
    imshow(axes[2, i], samples[i])
axes[0, 0].set_title("original", loc="left", fontsize=9)
axes[1, 0].set_title("reconstruction", loc="left", fontsize=9)
axes[2, 0].set_title("prior sample", loc="left", fontsize=9)
fig.suptitle(FIG_TITLE, fontsize=9)
fig.savefig(FIG_SAMPLES, dpi=150, bbox_inches="tight")
plt.show()

# ---- loss curves ----
d = np.genfromtxt(LOG, delimiter=",", names=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, key, title in zip(axes,
                          ["", "_recon", "_kl"],
                          ["objective", "reconstruction", "KL"]):
    ax.plot(d["epoch"], d["train" + key], label="train")
    ax.plot(d["epoch"], d["valid" + key], label="valid")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle(FIG_TITLE, fontsize=9)
plt.tight_layout()
fig.savefig(FIG_CURVES, dpi=150, bbox_inches="tight")
plt.show()

# %%
