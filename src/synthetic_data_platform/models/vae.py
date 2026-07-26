from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from synthetic_data_platform.preprocessing import TabularPreprocessor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TabularVAE(nn.Module):
    """Compact VAE for mixed numeric and categorical tabular data."""

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def fit(
        self,
        matrix: np.ndarray,
        preprocessor: TabularPreprocessor,
        epochs: int = 120,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        beta: float = 0.01,
        seed: int = 42,
        device: str = "cpu",
    ) -> list[float]:
        seed_everything(seed)
        torch_device = torch.device(device)
        self.to(torch_device)
        data = torch.tensor(matrix, dtype=torch.float32, device=torch_device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history: list[float] = []
        self.train()
        for _ in range(epochs):
            permutation = torch.randperm(len(data), device=torch_device)
            epoch_loss = 0.0
            for indexes in permutation.split(max(1, batch_size)):
                batch = data[indexes]
                reconstruction, mu, logvar = self(batch)
                loss = self._loss(reconstruction, batch, mu, logvar, preprocessor, beta)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch)
            history.append(epoch_loss / len(data))
        return history

    def sample(self, rows: int, preprocessor: TabularPreprocessor, seed: int = 42, device: str = "cpu") -> np.ndarray:
        if rows <= 0:
            raise ValueError("rows must be positive")
        seed_everything(seed)
        self.eval()
        torch_device = torch.device(device)
        latent = torch.randn(rows, self.latent_dim, device=torch_device)
        with torch.no_grad():
            raw = self.decode(latent)
        result = raw.detach().cpu().numpy()
        for spec in preprocessor.numeric:
            start, _ = preprocessor.feature_slices[spec.column]
            result[:, start] = np.tanh(result[:, start])
        for spec in preprocessor.categorical:
            start, end = preprocessor.feature_slices[spec.column]
            one_hot = np.zeros((rows, end - start), dtype=np.float32)
            one_hot[np.arange(rows), np.argmax(result[:, start:end], axis=1)] = 1.0
            result[:, start:end] = one_hot
        return result.astype(np.float32)

    @staticmethod
    def _loss(reconstruction, target, mu, logvar, preprocessor, beta):
        numeric_loss = torch.tensor(0.0, device=target.device)
        for spec in preprocessor.numeric:
            start, _ = preprocessor.feature_slices[spec.column]
            numeric_loss = numeric_loss + nn.functional.mse_loss(torch.tanh(reconstruction[:, start]), target[:, start])
        categorical_loss = torch.tensor(0.0, device=target.device)
        for spec in preprocessor.categorical:
            start, end = preprocessor.feature_slices[spec.column]
            labels = target[:, start:end].argmax(dim=1)
            categorical_loss = categorical_loss + nn.functional.cross_entropy(reconstruction[:, start:end], labels)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return numeric_loss + categorical_loss + beta * kl

