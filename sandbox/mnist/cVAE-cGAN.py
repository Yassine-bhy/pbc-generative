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
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# %%
# ---------- settings ----------
MODEL       = "cVAE-cGAN"
SEED        = 0
LATENT_DIM  = 64
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 2e-4
NUM_CLASSES = 10
GAMMA       = 1e-3      # decoder: weight on feature reconstruction vs adversarial
D_SKIP_ACC  = 0.80      # skip the discriminator update once it is this accurate

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- where things go ----------
run_name = (f"latent{LATENT_DIM}_lr{LR:g}_batch{BATCH_SIZE}_epochs{EPOCHS}"
            f"_gamma{GAMMA:g}_dskip{D_SKIP_ACC:g}_seed{SEED}")
model_dir   = Path("models")  / MODEL / run_name
results_dir = Path("results") / MODEL / run_name
model_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

ckpt        = model_dir   / "checkpoint.pt"
log_file    = results_dir / "training_log.csv"
fig_samples = results_dir / "generated_samples.png"
fig_curves  = results_dir / "loss_curves.png"

# ---------- write down what this run was ----------
config = f"""Model      : {MODEL}
Started    : {datetime.now():%Y-%m-%d %H:%M:%S}
Device     : {device}

Latent dimension : {LATENT_DIM}
Learning rate    : {LR}
Batch size       : {BATCH_SIZE}
Epochs           : {EPOCHS}
Gamma (decoder)  : {GAMMA}
D skip accuracy  : {D_SKIP_ACC}
Random seed      : {SEED}
"""

(results_dir / "configuration.txt").write_text(config)
print(config)


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
tf = v2.Compose([v2.ToImage(),
                 v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize([0.5], [0.5])])

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


trainloader = DataLoader(train_view, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx),
                         num_workers=4, generator=g, worker_init_fn=seed_worker,
                         pin_memory=True, persistent_workers=True)

validloader = DataLoader(valid_view, batch_size=BATCH_SIZE, sampler=val_idx,
                         num_workers=4, generator=g, worker_init_fn=seed_worker,
                         pin_memory=True, persistent_workers=True)

testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)


# %%
class Encoder(nn.Module): #same in VAE
    
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(num_classes, 28 * 28)
        self.conv1 = nn.Conv2d(2, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, self.latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, self.latent_dim)
                
    def onehot(self, y):
        return F.one_hot(y, num_classes = self.num_classes).float()

    def forward(self, x, y):
        ymap = self.fc(self.onehot(y))
        ymap = torch.unflatten(ymap, 1, [1, 28, 28])
        h = torch.cat([x, ymap], dim = 1)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = torch.flatten(h, 1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module): #the Generator in GAN, generates an image of 28*28 for mnist
    
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(self.latent_dim + self.num_classes, 128 * 7 * 7)  
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1) # get an image in gray scale with dimention 28*28

    def onehot(self, y):
        return F.one_hot(y, num_classes = self.num_classes).float()

    def forward(self, z, y):
        h = torch.cat([z, self.onehot(y)], dim = 1)
        h = self.fc(h)
        h = torch.unflatten(h, 1, [128, 7, 7])
        h = self.bn1(h)
        h = F.relu(h)
        
        h = self.convtrans1(h)
        h = self.bn2(h)
        h = F.relu(h)

        h = self.convtrans2(h)
        h = torch.tanh(h)
        return h

class Discriminator(nn.Module): #same in GAN, but returns a hidden layer used to calculate the loss
    
    def __init__(self, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(64)
        self.num_classes = num_classes
        self.fc = nn.Linear(num_classes, 28 * 28)
        self.conv1 = nn.Conv2d(2, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 7, stride=1, padding=0)

    def onehot(self, y):
        return F.one_hot(y, num_classes = self.num_classes).float()        

    def forward(self, x, y):
        ymap = self.fc(self.onehot(y))
        ymap = torch.unflatten(ymap, 1, [1, 28, 28])
        h = torch.cat([x, ymap], dim=1)
        h = F.leaky_relu(self.conv1(h), 0.2)
        f = F.leaky_relu(self.bn(self.conv2(h)), 0.2)   # the feature layer
        out = self.conv3(f)
        return out.view(-1), f


# %%
torch.manual_seed(SEED)

encoder = Encoder(LATENT_DIM, NUM_CLASSES).to(device)
decoder = Decoder(LATENT_DIM, NUM_CLASSES).to(device)
discriminator = Discriminator(NUM_CLASSES).to(device)

BCE = nn.BCEWithLogitsLoss()

optimiserEnc = torch.optim.Adam(encoder.parameters(), lr=LR, betas=(0.5, 0.999))
optimiserDec = torch.optim.Adam(decoder.parameters(), lr=LR, betas=(0.5, 0.999))
optimiserDis = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

fixed_z = torch.randn(80, LATENT_DIM, device=device)
fixed_y = torch.arange(NUM_CLASSES, device=device).repeat_interleave(8)

fixed_real, fixed_real_y = next(iter(trainloader))
fixed_real = fixed_real[:8].to(device)
fixed_real_y = fixed_real_y[:8].to(device)

# %%
with open(log_file, "w") as f:
    f.write("epoch,lossD,lossG,p_real,p_fake\n")

print(MODEL)
print(run_name)
header = f"{'epoch':>5} | {'lossD':>7} {'lossG':>7} | {'p_real':>6} {'p_fake':>6} |"
print(header)
print("-" * len(header))


for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    discriminator.train()
    tr = [0.0, 0.0, 0.0, 0.0]
    tr_n = 0
    tr_g = 0
    k = 0
    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        #the forward pass
        mu, logvar = encoder(images, labels)
        std = torch.exp(0.5 * logvar)
        z_rec = mu + torch.randn_like(std) * std
        img_rec = decoder(z_rec, labels)
        z_p = torch.randn(images.size(0), LATENT_DIM, device=device)
        img_p = decoder(z_p, labels)

        
        # training the discriminator
        optimiserEnc.zero_grad()
        optimiserDec.zero_grad()
        optimiserDis.zero_grad()

        D_org, _ = discriminator(images, labels) #f for the hidden layer
        D_rec, _ = discriminator(img_rec.detach(), labels)
        D_p, _ = discriminator(img_p.detach(), labels)

        L_GAN = BCE(D_org, torch.ones_like(D_org)) + BCE(D_rec, torch.zeros_like(D_rec)) + BCE(D_p, torch.zeros_like(D_p))
        L_GAN.backward()
        optimiserDis.step()

        #training the encoder and decoder
        optimiserEnc.zero_grad()
        optimiserDec.zero_grad()
        optimiserDis.zero_grad()

        D_org, f_org = discriminator(images, labels) #f for the hidden layer
        D_rec, f_rec = discriminator(img_rec, labels)
        D_p, _ = discriminator(img_p, labels)

        kl = (0.5 * (logvar.exp() + mu**2 - 1 - logvar)).sum(1).mean() #kl exactly like the VAE
        recon_layer = ((torch.sigmoid(f_rec) - f_org) ** 2).flatten(1).sum(1).mean() #loss got from an inner layer of the discrimimator, instead of getting it by pixels like in vae
        L_GAN = BCE(D_org, torch.ones_like(D_org)) + BCE(D_rec, torch.zeros_like(D_rec)) + BCE(D_p, torch.zeros_like(D_p))
        L_enc = (kl + recon_layer)
        L_dec = (recon_layer - L_GAN)
        
        L_enc.backward(retain_graph=True, inputs = list(encoder.parameters()))
        L_dec.backward(inputs=list(decoder.parameters()))

        optimiserEnc.step()
        optimiserDec.step()

        tr[1] += lossG.item()
        tr_g += 1
        bs = len(images)
        tr[0] += lossD.item() * bs
        tr[2] += torch.sigmoid(outputD).mean().item() * bs
        tr[3] += torch.sigmoid(outputDG).mean().item() * bs
        tr_n += bs

    tr[0] = tr[0] / tr_n
    tr[1] = tr[1] / max(tr_g, 1)
    tr[2] = tr[2] / tr_n
    tr[3] = tr[3] / tr_n

    G.eval()
    with torch.no_grad():
        samples = G(fixed_z, fixed_y).cpu()
    save_image(samples * 0.5 + 0.5, results_dir / f"epoch_{epoch+1:03d}.png", nrow=8)

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt)

    with open(log_file, "a") as f:
        f.write(f"{epoch+1}," + ",".join(f"{v:.4f}" for v in tr) + "\n")

    print(f"{epoch+1:>5} | {tr[0]:7.3f} {tr[1]:7.3f} "
          f"| {tr[2]:6.3f} {tr[3]:6.3f} |")


# %%
def imshow(ax, x):
    ax.imshow(x.detach().cpu().squeeze(0) * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

FIG_TITLE = f"{MODEL}\n{run_name}"

state = torch.load(ckpt, map_location=device)
G.load_state_dict(state["G"])
G.eval()

z = torch.randn(24, LATENT_DIM, device=device)
y = torch.arange(NUM_CLASSES, device=device)[:8].repeat(3)
with torch.no_grad():
    samples = G(z, y)

fig, axes = plt.subplots(3, 8, figsize=(12, 4.5))
for i in range(24):
    imshow(axes[i // 8, i % 8], samples[i])
fig.suptitle(FIG_TITLE, fontsize=9)
fig.savefig(fig_samples, dpi=150, bbox_inches="tight")
plt.show()

d = np.genfromtxt(log_file, delimiter=",", names=True)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].plot(d["epoch"], d["lossD"], label="D")
axes[0].plot(d["epoch"], d["lossG"], label="G")
axes[0].set_title("losses")
axes[1].plot(d["epoch"], d["p_real"], label="p_real")
axes[1].plot(d["epoch"], d["p_fake"], label="p_fake")
axes[1].set_title("discriminator output")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle(FIG_TITLE, fontsize=9)
plt.tight_layout()
fig.savefig(fig_curves, dpi=150, bbox_inches="tight")
plt.show()

# %%
print(Path.cwd())
print(results_dir.resolve())
print(list(results_dir.iterdir()))

# %%
