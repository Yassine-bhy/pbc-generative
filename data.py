from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

ROOT = Path.home() / "Globule" / "barcelona"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class PBCDataset(Dataset):
    def __init__(self, root, split, size, transform=None, excluded="excluded.txt"):
        root = Path(root)
        skip = set(Path(excluded).read_text().split())
        self.classes = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
        self.transform = transform
        images, labels, names = [], [], []
        for c, cls in enumerate(self.classes):
            for f in sorted((root / split / cls).glob("*.jpg")):
                if f"{split}/{cls}/{f.name}" in skip:
                    continue
                img = v2.functional.center_crop(read_image(str(f)), 360)
                images.append(v2.functional.resize(img, [size, size], antialias=True))
                labels.append(c)
                names.append(f.name)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)
        self.names = names

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class SyntheticDataset(Dataset):
    """Synthetic images written by train_generator.py: folder/<class>/<k>.png.
    Keeps the first `per_class` files of every class folder present (PROTOCOL 6: nested sets)."""

    def __init__(self, folder, classes, per_class, transform=None):
        self.transform = transform
        images, labels = [], []
        for c, cls in enumerate(classes):
            d = Path(folder) / cls
            if not d.exists():
                continue
            files = sorted(d.glob("*.png"))[:per_class]
            if len(files) < per_class:
                raise FileNotFoundError(f"{d}: {len(files)} images, {per_class} needed")
            images += [read_image(str(f)) for f in files]
            labels += [c] * per_class
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def rare_classes(labels, k=2):
    return torch.bincount(labels).argsort()[:k].tolist()


def subsample(labels, classes, n, seed):
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in range(int(labels.max()) + 1):
        idx = (labels == c).nonzero().flatten()
        if c in classes:
            idx = idx[torch.randperm(len(idx), generator=g)[:n]]
        keep.append(idx)
    return torch.cat(keep).sort().values


def class_weights(labels, num_classes):
    """Proportional to 1 / class count, normalised to mean 1 (PROTOCOL 5, weighted)."""
    w = 1 / torch.bincount(labels, minlength=num_classes).double()
    return (w / w.mean()).float()


def to_uint8(x):
    """Float images in [0, 1] -> uint8, the format real images have after resizing."""
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def network_input(x):
    """uint8 batch (B, 3, s, s) -> 224x224, ImageNet normalisation. Same steps as eval_tf
    in train_classifier.py, applied to a batch already on the GPU."""
    x = v2.functional.resize(x, [224, 224], antialias=True)
    x = v2.functional.to_dtype(x, torch.float32, scale=True)
    return v2.functional.normalize(x, MEAN, STD)
