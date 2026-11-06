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
import torch

if torch.cuda.is_available():
    print(f"Nombre de GPU disponibles : {torch.cuda.device_count()}\n")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} : {torch.cuda.get_device_name(i)}")
        print(f"  Device disponible : {torch.cuda.get_device_properties(i).name}")
        print(f"  Capacité de calcul : {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        print(f"  Mémoire totale : {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Mémoire utilisée : {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Mémoire libre : {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1e9:.2f} GB\n")
else:
    print("Aucun GPU disponible")

# %%
import torch

device = torch.device("cuda:1")

a = torch.randn(10000, 10000, device=device)
b = torch.randn(10000, 10000, device=device)

c = a @ b

torch.cuda.synchronize()

print(c.sum())
print("OK GPU computation")

# %%
import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import numpy as np
import random
import struct
from os.path import join
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# %%
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


# %% editable=true slideshow={"slide_type": ""}
#MNIST Dataset Array
#torch.utils.data.DataLoader needs a dataset class that has two methods: __len__ and __getitem__

class MnsitArrayDataset(Dataset):
    def __init__(self, images_filepath,labels_filepath, transform = None):
        self.images_filepath = images_filepath
        self.labels_filepath = labels_filepath
        self.transform = transform
        images, labels = self.read_images_labels(images_filepath, labels_filepath)
        self.images = images
        self.labels = torch.from_numpy(labels.astype("int64"))
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype = np.uint8).copy()   
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            images = (np.frombuffer(file.read(), dtype = np.uint8)).reshape(size, rows, cols).copy()
        return images, labels
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# %%
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(64 * 7 * 7, 10) 
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# %%
tf_train = v2.Compose([
    v2.ToImage(),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1307,), (0.3081,)),
])

tf_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1307,), (0.3081,)),
])

batch_size = 64

input_path = "./Data"
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')

train_view = MnsitArrayDataset(training_images_filepath, training_labels_filepath, tf_train)
valid_view = MnsitArrayDataset(training_images_filepath, training_labels_filepath, tf_test)


n = len(train_view)
train_idx = list(range(0, n*9//10))
val_idx   = list(range(n*9//10, n))

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

# %%
model = cnn()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters())

# %%
best_validation_loss = float("inf")

for epoch in range(25):
    print("\n-------")
    print(f"epoch number : {epoch+1}")
    # training phase
    model.train()
    running_train_loss = 0.0
    running_train_correct = 0
    running_train_total = 0
    for images, labels in trainloader:

        # move data to device (cuda here)
        images = images.to(device)
        labels = labels.to(device)

        # gradient accumulates so we need to zero it before recalculating it
        optimiser.zero_grad()

        # forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward pass and optimiser step
        loss.backward()
        optimiser.step()

        running_train_loss += loss.item() * len(images)
        running_train_correct += torch.sum(preds == labels)
        running_train_total += labels.size(0)

    train_loss = running_train_loss / running_train_total
    train_acc = running_train_correct.item() / running_train_total

    # validation phase
    model.eval()
    with torch.no_grad():
        running_val_loss = 0.0
        running_validation_correct = 0
        running_val_total = 0
        for images, labels in validloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_val_loss += loss.item() * len(images)
            running_validation_correct += torch.sum(preds == labels)
            running_val_total += labels.size(0)

        valid_loss = running_val_loss / running_val_total
        valid_acc = running_validation_correct.item() / running_val_total

        if best_validation_loss > valid_loss:
            torch.save(model.state_dict(), "mnist_cnn_judge.pt")
            best_validation_loss = valid_loss

    print(f"average training loss : {train_loss}")
    print(f"training accuracy : {train_acc}")
    print(f"average validation loss : {valid_loss}")
    print(f"validation accuracy : {valid_acc}")
