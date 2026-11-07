import torch
import torch.nn.functional as F


# ---------- the 8 dihedral transforms (PROTOCOL 6, 8) ----------
def dihedral(x, k):
    """k in 0..7: rotate by k % 4 quarter turns, then flip left-right if k >= 4."""
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return x.flip(-1) if k >= 4 else x


def random_dihedral(x):
    """One random dihedral transform per image of the batch."""
    k = torch.randint(0, 8, (len(x),), device=x.device)
    out = torch.empty_like(x)
    for t in range(8):
        m = k == t
        if m.any():
            out[m] = dihedral(x[m], t)
    return out


# ---------- DiffAugment (Zhao et al. 2020), policy colour + translation + cutout ----------
# Images in [-1, 1]. Applied to real and generated images before the discriminator.
def brightness(x):
    return x + (torch.rand(len(x), 1, 1, 1, device=x.device) - 0.5)


def saturation(x):
    m = x.mean(1, keepdim=True)
    return (x - m) * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2) + m


def contrast(x):
    m = x.mean((1, 2, 3), keepdim=True)
    return (x - m) * (torch.rand(len(x), 1, 1, 1, device=x.device) + 0.5) + m


def translation(x, ratio=0.125):
    B, C, H, W = x.shape
    sh, sw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    th = torch.randint(-sh, sh + 1, (B, 1, 1), device=x.device)
    tw = torch.randint(-sw, sw + 1, (B, 1, 1), device=x.device)
    gb, gh, gw = torch.meshgrid(torch.arange(B, device=x.device), torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device), indexing="ij")
    gh = (gh + th + 1).clamp(0, H + 1)
    gw = (gw + tw + 1).clamp(0, W + 1)
    padded = F.pad(x, [1, 1, 1, 1]).permute(0, 2, 3, 1)
    return padded[gb, gh, gw].permute(0, 3, 1, 2).contiguous()


def cutout(x, ratio=0.5):
    B, C, H, W = x.shape
    ch, cw = int(H * ratio + 0.5), int(W * ratio + 0.5)
    oh = torch.randint(0, H + (1 - ch % 2), (B, 1, 1), device=x.device)
    ow = torch.randint(0, W + (1 - cw % 2), (B, 1, 1), device=x.device)
    gb, gh, gw = torch.meshgrid(torch.arange(B, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gh = (gh + oh - ch // 2).clamp(0, H - 1)
    gw = (gw + ow - cw // 2).clamp(0, W - 1)
    mask = torch.ones(B, H, W, dtype=x.dtype, device=x.device)
    mask[gb, gh, gw] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x):
    return cutout(translation(contrast(saturation(brightness(x)))))
