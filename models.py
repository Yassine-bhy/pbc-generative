import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

SIZE = 128
UP = [512, 512, 256, 128, 64, 32]      # channels at 4, 8, 16, 32, 64, 128
DOWN = [32, 64, 128, 256, 512]         # channels at 64, 32, 16, 8, 4


class Decoder(nn.Module):
    """(z, y) -> 3x128x128 raw output. The cVAE decoder and the cGAN generator are this same network:
    sigmoid is applied for the cVAE, tanh for the cGAN.
    4x4 grid, five upsampling stages (PROTOCOL 6)."""

    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(latent_dim + num_classes, UP[0] * 4 * 4)
        self.bn = nn.BatchNorm2d(UP[0])
        self.ups = nn.Sequential(*[
            nn.Sequential(nn.ConvTranspose2d(a, b, 4, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU())
            for a, b in zip(UP[:-1], UP[1:])])
        self.out = nn.Conv2d(UP[-1], 3, 3, padding=1)

    def forward(self, z, y):
        h = self.fc(torch.cat([z, F.one_hot(y, self.num_classes).float()], 1))
        h = F.relu(self.bn(torch.unflatten(h, 1, [UP[0], 4, 4])))
        return self.out(self.ups(h))


class Encoder(nn.Module):
    """(x, y) -> (mu, logvar). The label enters as an extra image channel, as in the MNIST cVAE.
    Mirror of the decoder: five stride-2 convolutions, 128 -> 4."""

    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ymap = nn.Linear(num_classes, SIZE * SIZE)
        widths = [3 + 1] + DOWN
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Conv2d(a, b, 4, stride=2, padding=1))
            if i > 0:
                layers.append(nn.BatchNorm2d(b))
            layers.append(nn.LeakyReLU(0.2))
        self.convs = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(DOWN[-1] * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(DOWN[-1] * 4 * 4, latent_dim)

    def forward(self, x, y):
        ymap = torch.unflatten(self.ymap(F.one_hot(y, self.num_classes).float()), 1, [1, SIZE, SIZE])
        h = torch.flatten(self.convs(torch.cat([x, ymap], 1)), 1)
        return self.fc_mu(h), self.fc_logvar(h)


class cVAE(nn.Module):
    """Images in [0, 1]."""

    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z, y), mu, logvar

    def objective(self, x, y, recon_weight):
        """Training objective PER IMAGE (vector of length B): recon_weight * summed squared
        error + KL. Lower is better. The ELBO attack of PROTOCOL 8.3 uses minus this."""
        logits, mu, logvar = self(x, y)
        recon = ((torch.sigmoid(logits) - x) ** 2).flatten(1).sum(1)
        kl = (0.5 * (logvar.exp() + mu**2 - 1 - logvar)).sum(1)
        return recon_weight * recon + kl, recon, kl

    def generate(self, z, y):
        return torch.sigmoid(self.decoder(z, y))


class Discriminator(nn.Module):
    """Projection discriminator (Miyato & Koyama 2018). Images in [-1, 1].
    Spectral norm on every layer, no batch norm: the score of one image never depends on the
    rest of the batch (PROTOCOL 6, needed by the attack of 8.3)."""

    def __init__(self, num_classes):
        super().__init__()
        widths = [3] + DOWN
        self.convs = nn.ModuleList([spectral_norm(nn.Conv2d(a, b, 4, stride=2, padding=1))
                                    for a, b in zip(widths[:-1], widths[1:])])
        self.fc = spectral_norm(nn.Linear(DOWN[-1], 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, DOWN[-1]))

    def forward(self, x, y):
        h = x
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.2)
        h = h.sum((2, 3))
        return self.fc(h).squeeze(1) + (self.embed(y) * h).sum(1)
