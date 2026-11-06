# %%
import struct
from os.path import join

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# %%
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


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
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64 * 7 * 7)
        self.convtrans1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        x = self.fc2(x)
        x = torch.unflatten(x, 1, [64, 7, 7])

        x = F.relu(self.convtrans1(x))
        x = self.convtrans2(x)
        return x


# %%
MEAN, STD = 0.1307, 0.3081

tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((MEAN,), (STD,)),
])

batch_size = 64

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
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters())

# %%
best_validation_loss = float("inf")

for epoch in range(25):
    print("\n-------")
    print(f"epoch number : {epoch+1}")

    model.train()
    running_train_loss = 0.0
    running_train_total = 0
    for images, _ in trainloader:
        images = images.to(device)

        optimiser.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, images)

        loss.backward()
        optimiser.step()

        running_train_loss += loss.item() * len(images)
        running_train_total += len(images)

    train_loss = running_train_loss / running_train_total

    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        running_val_total = 0
        for images, _ in validloader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            running_val_loss += loss.item() * len(images)
            running_val_total += len(images)

        valid_loss = running_val_loss / running_val_total

        if best_validation_loss > valid_loss:
            torch.save(model.state_dict(), "autoencoder.pt")
            best_validation_loss = valid_loss

    print(f"average training loss : {train_loss}")
    print(f"average validation loss : {valid_loss}")


# %%
def imshow(ax, x):
    x = x.detach().cpu() * STD + MEAN
    ax.imshow(x.squeeze(0).clamp(0, 1), cmap="gray")
    ax.axis("off")


model.load_state_dict(torch.load("autoencoder.pt"))
model.eval()

images, _ = next(iter(testloader))
images = images.to(device)
with torch.no_grad():
    outputs = model(images)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    imshow(axes[0, i], images[i])
    imshow(axes[1, i], outputs[i])
plt.show()

# %%
