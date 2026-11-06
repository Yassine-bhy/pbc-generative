from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


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
