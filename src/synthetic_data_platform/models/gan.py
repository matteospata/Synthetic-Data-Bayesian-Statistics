from __future__ import annotations

import numpy as np
import torch
from torch import nn

from synthetic_data_platform.models.vae import seed_everything
from synthetic_data_platform.preprocessing import TabularPreprocessor


class TabularGAN:
    """Small MLP GAN baseline for benchmarking synthetic tabular data."""

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        self.discriminator = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1))

    def fit(self, matrix: np.ndarray, preprocessor: TabularPreprocessor, epochs: int = 120, batch_size: int = 64, learning_rate: float = 2e-4, seed: int = 42, device: str = "cpu") -> dict[str, list[float]]:
        seed_everything(seed)
        torch_device = torch.device(device)
        self.generator.to(torch_device)
        self.discriminator.to(torch_device)
        data = torch.tensor(matrix, dtype=torch.float32, device=torch_device)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        criterion = nn.BCEWithLogitsLoss()
        history = {"generator": [], "discriminator": []}
        for _ in range(epochs):
            order = torch.randperm(len(data), device=torch_device)
            generator_total = discriminator_total = 0.0
            for indexes in order.split(max(1, batch_size)):
                real = data[indexes]
                noise = torch.randn(len(real), self.latent_dim, device=torch_device)
                fake = self._activate(self.generator(noise), preprocessor)
                discriminator_optimizer.zero_grad()
                real_loss = criterion(self.discriminator(real), torch.ones((len(real), 1), device=torch_device))
                fake_loss = criterion(self.discriminator(fake.detach()), torch.zeros((len(real), 1), device=torch_device))
                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_optimizer.step()
                generator_optimizer.zero_grad()
                generator_loss = criterion(self.discriminator(fake), torch.ones((len(real), 1), device=torch_device))
                generator_loss.backward()
                generator_optimizer.step()
                generator_total += float(generator_loss.detach()) * len(real)
                discriminator_total += float(discriminator_loss.detach()) * len(real)
            history["generator"].append(generator_total / len(data))
            history["discriminator"].append(discriminator_total / len(data))
        return history

    def sample(self, rows: int, preprocessor: TabularPreprocessor, seed: int = 42, device: str = "cpu") -> np.ndarray:
        seed_everything(seed)
        self.generator.eval()
        with torch.no_grad():
            raw = self.generator(torch.randn(rows, self.latent_dim, device=torch.device(device)))
            return self._activate(raw, preprocessor).cpu().numpy().astype(np.float32)

    @staticmethod
    def _activate(raw: torch.Tensor, preprocessor: TabularPreprocessor) -> torch.Tensor:
        result = raw.clone()
        for spec in preprocessor.numeric:
            start, _ = preprocessor.feature_slices[spec.column]
            result[:, start] = torch.tanh(raw[:, start])
        for spec in preprocessor.categorical:
            start, end = preprocessor.feature_slices[spec.column]
            result[:, start:end] = torch.softmax(raw[:, start:end], dim=1)
        return result

