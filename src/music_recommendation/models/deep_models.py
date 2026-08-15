from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from music_recommendation.schemas import ContentDataset


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.decoder(latent)


@dataclass(slots=True)
class DLTrainingResult:
    final_loss: float
    epochs: int


class DeepContentTrainer:
    def __init__(self, epochs: int = 5, batch_size: int = 256, learning_rate: float = 1e-3) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model: AutoEncoder | None = None

    def fit(self, dataset: ContentDataset) -> DLTrainingResult:
        frame = dataset.frame[dataset.feature_columns]
        tensor = torch.tensor(frame.to_numpy(), dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=self.batch_size, shuffle=True)
        self.model = AutoEncoder(input_dim=tensor.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        self.model.train()
        final_loss = 0.0
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            final_loss = epoch_loss / max(len(loader), 1)
        return DLTrainingResult(final_loss=final_loss, epochs=self.epochs)
