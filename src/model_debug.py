"""Plotly helpers for tracking and visualising PyTorch model parameters."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


class ModelDebugger:
    """Collect training diagnostics and show them as interactive Plotly figures."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.epoch_parameters: dict[int, dict[str, torch.Tensor]] = {}
        self.epoch_train_losses: dict[int, float] = {}
        self.epoch_validation_losses: dict[int, float] = {}

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Store losses plus an independent parameter snapshot for an epoch."""
        epoch = int(epoch)
        self.epoch_train_losses[epoch] = float(train_loss)
        self.epoch_validation_losses[epoch] = float(val_loss)
        self.epoch_parameters[epoch] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }

    def _parameters_for_epoch(
        self,
        epoch: int | None,
    ) -> list[tuple[str, torch.Tensor]]:
        if epoch is None:
            return [
                (name, parameter.detach().cpu())
                for name, parameter in self.model.named_parameters()
            ]
        epoch = int(epoch)
        if epoch not in self.epoch_parameters:
            available = sorted(self.epoch_parameters)
            raise KeyError(
                f"No parameters recorded for epoch {epoch}. "
                f"Available epochs: {available}"
            )
        return [
            (name, parameter.clone())
            for name, parameter in self.epoch_parameters[epoch].items()
        ]

    def plot_losses(self) -> go.Figure:
        """Create loss curves from the epoch history stored by ``record_epoch``."""
        if not self.epoch_train_losses:
            raise RuntimeError(
                "No epoch losses recorded; call record_epoch during training"
            )

        epochs = sorted(self.epoch_train_losses)
        train_history = [self.epoch_train_losses[epoch] for epoch in epochs]
        validation_history = [
            self.epoch_validation_losses[epoch] for epoch in epochs
        ]
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=epochs, y=train_history, mode="lines+markers", name="Train loss"
        ))
        figure.add_trace(go.Scatter(
            x=epochs,
            y=validation_history,
            mode="lines+markers",
            name="Validation loss",
        ))
        figure.update_layout(
            title="Training history",
            xaxis_title="Epoch (0 = before training)",
            yaxis_title="Loss",
            template="plotly_white",
        )
        figure.update_xaxes(dtick=1)
        return figure

    def plot_distributions(self, epoch: int | None = None) -> go.Figure:
        """Create parameter histograms for the current or selected epoch."""
        parameters = self._parameters_for_epoch(epoch)
        titles = [name for name, _ in parameters]
        figure = make_subplots(rows=len(parameters), cols=1, subplot_titles=titles)

        for row, (name, parameter) in enumerate(parameters, start=1):
            values = parameter.flatten().tolist()
            figure.add_trace(
                go.Histogram(x=values, nbinsx=50, name=name, showlegend=False), row=row, col=1
            )

        figure.update_layout(
            height=max(500, 260 * len(parameters)),
            title=(
                "Current parameter distributions"
                if epoch is None
                else f"Parameter distributions — epoch {epoch}"
            ),
            template="plotly_white",
        )
        return figure

    def save_figures(self, best_epoch: int, directory: str | Path) -> None:
        """Save training history and best-epoch parameter distributions as HTML."""
        best_epoch = int(best_epoch)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.plot_losses().write_html(
            directory / "training_losses.html",
            auto_open=False,
        )
        self.plot_distributions(best_epoch).write_html(
            directory / f"parameter_distributions_epoch_{best_epoch}.html",
            auto_open=False,
        )
