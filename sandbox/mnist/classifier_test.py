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
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
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


# %%
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
tf_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1307,), (0.3081,)),
])

batch_size = 64

input_path = "./Data"

test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

test_set = MnsitArrayDataset(test_images_filepath, test_labels_filepath, tf_test)

testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)


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
model = cnn()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters())

# %%
model.load_state_dict(torch.load("mnist_cnn_judge.pt", weights_only=True))

#testing phase
model.eval()
num_classes = 10
cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
with torch.no_grad():
    
    running_test_loss = 0.0
    running_test_correct = 0
    running_test_total = 0

    sample_images, sample_true, sample_pred = [], [], []
    wrong_images, wrong_true, wrong_pred = [], [], []
    
    for images, labels in testloader:
        
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        cm += torch.bincount(labels * num_classes + preds, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu()
        
        running_test_loss += loss.item() * len(images)
        running_test_correct += torch.sum(preds == labels)
        running_test_total += labels.size(0)

        if len(sample_images) < 16:
            sample_images += list(images[:16].cpu())
            sample_true += list(labels[:16].cpu())
            sample_pred += list(preds[:16].cpu())

        mask = preds != labels
        if mask.any():
            wrong_images += list(images[mask].cpu())
            wrong_true += list(labels[mask].cpu())
            wrong_pred += list(preds[mask].cpu())

    test_loss = running_test_loss / running_test_total
    test_acc = running_test_correct.item() / running_test_total
    print(f"average test loss : {test_loss}")
    print(f"test accuracy : {test_acc}")


# %%
def metrics(cm, _print = True):
    cm = cm.float()
    support = cm.sum(dim = 1)
    predicted_count = cm.sum(dim = 0)
    recalls = cm.diagonal()/support
    precision = cm.diagonal()/predicted_count
    F1 = 2 * recalls * precision / (recalls + precision)
    balanced_accuracy = recalls.mean()
    macro_F1 = F1.mean()
    accuracy = cm.diagonal().sum().item() / cm.sum().item()
    if _print == True:
        print(f"accuracy          : {accuracy:.4f}")
        print(f"balanced accuracy : {balanced_accuracy:.4f}")
        print(f"macro F1          : {macro_F1:.4f}")
        print(f"{'class':<8}{'support':>9}{'recall':>9}{'precision':>11}{'F1':>8}")
        for i in range(cm.size(0)):
            print(f"{i:<8}{support[i].item():>9}{recalls[i]:>9.4f}{precision[i]:>11.4f}{F1[i]:>8.4f}")
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy.item(),
        "macro_F1": macro_F1.item(),
        "recall": recalls,
        "precision": precision,
        "F1": F1,
        "support": support,
    }


# %%
MEAN, STD = 0.1307, 0.3081

def imshow(img, title=None, ax=None):
    img = (img.cpu() * STD + MEAN).clamp(0, 1)
    ax = ax or plt.gca()
    ax.imshow(img.squeeze(), cmap="gray")
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")

def show_grid(images, true, pred, suptitle):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            imshow(images[i], f"true {true[i].item()} / pred {pred[i].item()}", ax)
        else:
            ax.axis("off")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

show_grid(sample_images[:16], sample_true[:16], sample_pred[:16], "random test samples")
show_grid(wrong_images[:16], wrong_true[:16], wrong_pred[:16], "misclassified")

# %%
metrics(cm)

# %%
