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
# # Generators (PROTOCOL 6, 8.6)
#
# Class-conditional VAE or GAN at 128×128, trained for a fixed number of iterations
# (one iteration = one batch).
#
#     python train_generator.py cgan --pilot --device cuda:1          # pilot, uncut set
#     python train_generator.py cvae --device cuda:1 --replicates 0 1 # subsampled runs
#
# **Pilot** (`--pilot`): uncut training set, seed 0, `PILOT_ITERS` iterations, 20 checkpoints.
# At each checkpoint: Fréchet distance to validation and judge accuracy on rare-class
# samples, written to `pilot_log.csv`, plus a sample grid. The budget is read from that
# file afterwards (PROTOCOL 6) and frozen by amendment in `BUDGET` below.
#
# **Subsampled runs**: refuse to start while `BUDGET` is None. Save what PROTOCOL 8.6
# requires, then draw 2000 synthetic images per rare class into `synthetic/`.
#
# A run interrupted by a crash resumes from `last.pt` when the same command is relaunched,
# and the resumption is written in `resumed.txt` (PROTOCOL 6). The test set is never loaded.

# %%
import argparse
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import write_png
from torchvision.utils import save_image

from data import ROOT, PBCDataset, rare_classes, subsample, to_uint8
from augment import random_dihedral, diff_augment
from models import cVAE, Decoder, Discriminator
from features import Features, load_judge, apply, frechet_distance

# %%
# ---------- settings ----------
parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["cvae", "cgan"])
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--pilot", action="store_true")
parser.add_argument("--n", type=int, nargs="+", default=[50, 150, 400])
parser.add_argument("--replicates", type=int, nargs="+", default=list(range(10)))
args = parser.parse_args()

MODEL       = args.model
SIZE        = 128
BATCH_SIZE  = 64
PILOT_ITERS = 40_000           # A5: fixed before the pilot starts
N_CHECKPOINTS = 20             # pilot checkpoints, equally spaced
N_SYNTHETIC = 2000             # per rare class (PROTOCOL 6)
N_JUDGE     = 500              # samples per rare class for judge accuracy at pilot checkpoints
EVAL_SEED   = 12345            # fixed noise for Fréchet distance and judge, comparable across checkpoints
SAVE_EVERY  = 1000             # resume checkpoint

HPARAMS = {   # A5. The pilot may change these; the final values are frozen with the budget.
    "cvae": dict(latent_dim=128, lr=5e-4, betas=(0.9, 0.999), recon_weight=50.0),   # recon_weight = 1/(2 sigma^2), sigma = 0.1
    "cgan": dict(latent_dim=128, lr_g=1e-4, lr_d=4e-4, betas=(0.0, 0.999)),        # hinge loss, 1 D step per G step
}
H = HPARAMS[MODEL]

# Frozen by amendment after the pilot (PROTOCOL 6): earliest pilot checkpoint whose Fréchet
# distance is within 5% of the pilot's minimum. None = not frozen yet.
BUDGET = {"cvae": None, "cgan": None}

device = torch.device(args.device)

# %%
# ---------- data: everything lives on the GPU as uint8 ----------
train_set = PBCDataset(ROOT, "train", SIZE)
valid_set = PBCDataset(ROOT, "valid", SIZE)
classes = train_set.classes
C = len(classes)
rare = rare_classes(train_set.labels)

valid_images = valid_set.images.to(device)
valid_labels = valid_set.labels.to(device)

phi = Features().to(device)
judge = load_judge(C, device)
valid_features = apply(phi, valid_images, preprocess=None)

if args.pilot:
    runs = [("pilot", 0, 0)]
else:
    if BUDGET[MODEL] is None:
        raise SystemExit(f"BUDGET['{MODEL}'] is not frozen: run the pilot, write the amendment, then set it.")
    runs = [(f"n{n}_rep{r}", r, n) for r in args.replicates for n in args.n]


# %%
# ---------- helpers ----------
def build():
    if MODEL == "cvae":
        nets = {"vae": cVAE(H["latent_dim"], C).to(device)}
        opts = {"vae": torch.optim.Adam(nets["vae"].parameters(), lr=H["lr"], betas=H["betas"])}
    else:
        nets = {"G": Decoder(H["latent_dim"], C).to(device), "D": Discriminator(C).to(device)}
        opts = {"G": torch.optim.Adam(nets["G"].parameters(), lr=H["lr_g"], betas=H["betas"]),
                "D": torch.optim.Adam(nets["D"].parameters(), lr=H["lr_d"], betas=H["betas"])}
    return nets, opts


@torch.no_grad()
def generate(nets, labels, gen, batch=250):
    """One uint8 image per label, generator in eval mode (batch-norm running statistics)."""
    dec = nets["vae"].decoder if MODEL == "cvae" else nets["G"]
    was = dec.training
    dec.eval()
    out = []
    for y in labels.split(batch):
        z = torch.randn(len(y), H["latent_dim"], device=device, generator=gen)
        x = dec(z, y)
        x = torch.sigmoid(x) if MODEL == "cvae" else (torch.tanh(x) + 1) / 2
        out.append(to_uint8(x))
    dec.train(was)
    return torch.cat(out)


def seeded(seed):
    return torch.Generator(device=device).manual_seed(seed)


def quality(nets):
    """Fréchet distance to validation (one generated image per validation label) and judge
    accuracy on N_JUDGE samples of each rare class."""
    fake = generate(nets, valid_labels, seeded(EVAL_SEED))
    fd = frechet_distance(apply(phi, fake, preprocess=None), valid_features)
    accs = []
    for c in rare:
        imgs = generate(nets, torch.full((N_JUDGE,), c, device=device), seeded(EVAL_SEED + 1 + c))
        accs.append((apply(judge, imgs).argmax(1) == c).float().mean().item())
    return fd, accs


@torch.no_grad()
def disc_logit(D, images, labels, batch=256):
    """Mean discriminator logit, no augmentation, eval mode."""
    D.eval()
    s = sum(D(x.float() / 127.5 - 1, y).sum().item() for x, y in zip(images.split(batch), labels.split(batch)))
    D.train()
    return s / len(labels)


@torch.no_grad()
def vae_objective(vae, images, labels, batch=256):
    """Mean training objective, no augmentation, eval mode."""
    vae.eval()
    s = sum(vae.objective(x.float() / 255, y, H["recon_weight"])[0].sum().item()
            for x, y in zip(images.split(batch), labels.split(batch)))
    vae.train()
    return s / len(labels)


def state(nets, opts, it, gs):
    return {"nets": {k: v.state_dict() for k, v in nets.items()},
            "opts": {k: v.state_dict() for k, v in opts.items()},
            "iteration": it, "sampler": gs.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# %%
for run_name, SEED, N in runs:
    model_dir   = Path("models")  / "generator" / MODEL / run_name
    results_dir = Path("results") / "generator" / MODEL / run_name
    synth_dir   = Path("synthetic") / MODEL / run_name
    if (results_dir / "done").exists():
        print("already done:", run_name)
        continue
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "training_log.csv"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ---------- training subset (PROTOCOL 6): removed images are never seen ----------
    idx = subsample(train_set.labels, rare, N, SEED) if N else torch.arange(len(train_set))
    images = train_set.images[idx].to(device)
    labels = train_set.labels[idx].to(device)
    counts = torch.bincount(labels, minlength=C).cpu()
    draw_w = (1 / counts.double())[labels.cpu()]               # class-balanced sampling
    rare_mask = torch.isin(labels, torch.tensor(rare, device=device))
    vrare_mask = torch.isin(valid_labels, torch.tensor(rare, device=device))
    iters_per_epoch = len(idx) // BATCH_SIZE

    total = PILOT_ITERS if args.pilot else BUDGET[MODEL]
    if args.pilot:
        save_points = {total * k // N_CHECKPOINTS for k in range(1, N_CHECKPOINTS + 1)}
    else:
        save_points = {total * k // 4 for k in range(1, 5)}       # 25 / 50 / 75 / 100 % (PROTOCOL 8.6)

    nets, opts = build()
    gs = torch.Generator().manual_seed(SEED)                     # batch sampling
    it = 0

    # ---------- resume ----------
    last = model_dir / "last.pt"
    if last.exists():
        s = torch.load(last, map_location=device, weights_only=False)
        for k in nets:
            nets[k].load_state_dict(s["nets"][k])
            opts[k].load_state_dict(s["opts"][k])
        it = s["iteration"]
        gs.set_state(s["sampler"])
        torch.set_rng_state(s["torch_rng"])
        if s["cuda_rng"] is not None:
            torch.cuda.set_rng_state(s["cuda_rng"], device)
        with open(results_dir / "resumed.txt", "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} resumed at iteration {it}\n")
        print(f"resumed {run_name} at iteration {it}")
    else:
        # ---------- write down what this run was ----------
        config = f"""Model      : {MODEL} (class-conditional, {SIZE}x{SIZE} RGB)
Started    : {datetime.now():%Y-%m-%d %H:%M:%S}
Device     : {device}

Run              : {run_name}
Rare classes     : {", ".join(classes[c] for c in rare)}
Images per rare  : {N if N else "all"}
Replicate / seed : {SEED}
Iterations       : {total}  ({iters_per_epoch} per epoch)
Batch size       : {BATCH_SIZE}, class-balanced, random dihedral transform
Hyperparameters  : {H}

Training images per class:
""" + "\n".join(f"  {c:14}{k}" for c, k in zip(classes, counts.tolist())) + "\n"
        (results_dir / "configuration.txt").write_text(config)
        (results_dir / "train_images.txt").write_text(
            "\n".join(f"{classes[train_set.labels[i]]}/{train_set.names[i]}" for i in idx.tolist()) + "\n")
        print(config)
        with open(log_file, "w") as f:
            if MODEL == "cgan":
                f.write("epoch,iteration,lossD,lossG,logit_train_seen,logit_train_rare,logit_valid,logit_valid_rare,seconds\n")
            else:
                f.write("epoch,iteration,train,train_recon,train_kl,objective_train_rare,objective_valid,objective_valid_rare,seconds\n")
        if args.pilot:
            with open(results_dir / "pilot_log.csv", "w") as f:
                f.write("iteration,fd," + ",".join(f"judge_{classes[c]}" for c in rare) + "\n")

    fixed_y = torch.arange(C, device=device).repeat_interleave(8)     # sample grid: 8 per class

    # ---------- training ----------
    acc = np.zeros(3)
    acc_n = 0
    t0 = time.time()
    while it < total:
        b = torch.multinomial(draw_w, BATCH_SIZE, replacement=True, generator=gs).to(device)
        x = random_dihedral(images[b]).float() / 255                    # [0, 1]
        y = labels[b]

        if MODEL == "cvae":
            vae = nets["vae"]
            opts["vae"].zero_grad()
            loss, recon, kl = vae.objective(x, y, H["recon_weight"])
            loss.mean().backward()
            opts["vae"].step()
            acc += [loss.mean().item(), recon.mean().item(), kl.mean().item()]
        else:
            G, D = nets["G"], nets["D"]
            real = x * 2 - 1                                            # [-1, 1]
            # discriminator step (hinge loss)
            z = torch.randn(BATCH_SIZE, H["latent_dim"], device=device)
            fake = torch.tanh(G(z, y)).detach()
            opts["D"].zero_grad()
            d_real = D(diff_augment(real), y)
            d_fake = D(diff_augment(fake), y)
            lossD = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            lossD.backward()
            opts["D"].step()
            # generator step
            z = torch.randn(BATCH_SIZE, H["latent_dim"], device=device)
            opts["G"].zero_grad()
            lossG = -D(diff_augment(torch.tanh(G(z, y))), y).mean()
            lossG.backward()
            opts["G"].step()
            acc += [lossD.item(), lossG.item(), d_real.mean().item()]
        acc_n += 1
        it += 1

        # ---------- per epoch (PROTOCOL 8.6) ----------
        if it % iters_per_epoch == 0 or it == total:
            epoch = -(-it // iters_per_epoch)
            a = acc / acc_n
            if MODEL == "cgan":
                row = [epoch, it, a[0], a[1], a[2],
                       disc_logit(D, images[rare_mask], labels[rare_mask]),
                       disc_logit(D, valid_images, valid_labels),
                       disc_logit(D, valid_images[vrare_mask], valid_labels[vrare_mask])]
                msg = f"lossD {a[0]:.3f} lossG {a[1]:.3f} | logit train rare {row[5]:+.3f} valid rare {row[7]:+.3f}"
            else:
                row = [epoch, it, a[0], a[1], a[2],
                       vae_objective(vae, images[rare_mask], labels[rare_mask]),
                       vae_objective(vae, valid_images, valid_labels),
                       vae_objective(vae, valid_images[vrare_mask], valid_labels[vrare_mask])]
                msg = f"train {a[0]:.1f} (recon {a[1]:.1f}, kl {a[2]:.1f}) | train rare {row[5]:.1f} valid rare {row[7]:.1f}"
            row.append(time.time() - t0)
            with open(log_file, "a") as f:
                f.write(",".join(f"{v:.6g}" for v in row) + "\n")
            print(f"epoch {epoch:>4} | it {it:>6}/{total} | {msg} | {time.time() - t0:.0f}s", flush=True)
            acc[:] = 0
            acc_n = 0
            t0 = time.time()

        # ---------- checkpoints ----------
        if it in save_points:
            torch.save({k: v.state_dict() for k, v in nets.items()}, model_dir / f"iter_{it:06d}.pt")
            if args.pilot:
                fd, accs = quality(nets)
                with open(results_dir / "pilot_log.csv", "a") as f:
                    f.write(f"{it},{fd:.6g}," + ",".join(f"{a:.4f}" for a in accs) + "\n")
                grid = generate(nets, fixed_y, seeded(EVAL_SEED))
                save_image(grid.float() / 255, results_dir / f"samples_{it:06d}.png", nrow=8)
                print(f"checkpoint {it}: FD {fd:.2f} | judge " +
                      ", ".join(f"{classes[c]} {a:.3f}" for c, a in zip(rare, accs)), flush=True)
        if it % SAVE_EVERY == 0 or it in save_points:
            save_atomic(state(nets, opts, it, gs), last)

    # ---------- after training ----------
    torch.save({k: v.state_dict() for k, v in nets.items()}, model_dir / "final.pt")
    summary = f"Finished   : {datetime.now():%Y-%m-%d %H:%M:%S}\nIterations : {it}\n"

    if not args.pilot:
        # synthetic set (PROTOCOL 6): drawn once with seed r, rare class after rare class, no filtering
        gen = seeded(SEED)
        for c in rare:
            out = synth_dir / classes[c]
            out.mkdir(parents=True, exist_ok=True)
            imgs = generate(nets, torch.full((N_SYNTHETIC,), c, device=device), gen).cpu()
            for k, img in enumerate(imgs):
                write_png(img, str(out / f"{k:05d}.png"))
            judge_acc = (apply(judge, imgs).argmax(1) == c).float().mean().item()
            summary += f"Judge accuracy, {classes[c]:12}: {judge_acc:.4f}  ({N_SYNTHETIC} synthetic images)\n"
        (synth_dir / "done").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fd, _ = quality(nets)
        summary += f"Fréchet distance to validation: {fd:.4f}\n"
        save_image(generate(nets, fixed_y, seeded(EVAL_SEED)).float() / 255, results_dir / "samples_final.png", nrow=8)

    (results_dir / "summary.txt").write_text(summary)
    print(summary)
    (results_dir / "done").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n")
    last.unlink(missing_ok=True)     # optimiser state no longer needed; final weights kept
