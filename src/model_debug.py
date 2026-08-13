"""Plotly helpers for tracking and visualising PyTorch training diagnostics."""

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
        self.epoch_gradients: dict[int, dict[str, torch.Tensor | None]] = {}
        self.epoch_train_losses: dict[int, float] = {}
        self.epoch_validation_losses: dict[int, float] = {}

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Store losses plus independent parameter and gradient snapshots."""
        epoch = int(epoch)
        self.epoch_train_losses[epoch] = float(train_loss)
        self.epoch_validation_losses[epoch] = float(val_loss)
        self.epoch_parameters[epoch] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }
        self.epoch_gradients[epoch] = {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
            for name, parameter in self.model.named_parameters()
        }

    def _diagnostics_for_epoch(
        self,
        epoch: int | None,
    ) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
        if epoch is None:
            return [
                (
                    name,
                    parameter.detach().cpu(),
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().cpu(),
                )
                for name, parameter in self.model.named_parameters()
            ]
        epoch = int(epoch)
        if epoch not in self.epoch_parameters:
            available = sorted(self.epoch_parameters)
            raise KeyError(
                f"No parameters recorded for epoch {epoch}. "
                f"Available epochs: {available}"
            )
        gradients = self.epoch_gradients[epoch]
        return [
            (
                name,
                parameter.clone(),
                None if gradients[name] is None else gradients[name].clone(),
            )
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
        figure.update_xaxes(
            dtick=1,
            exponentformat="power",
            showexponent="all",
        )
        figure.update_yaxes(
            exponentformat="power",
            showexponent="all",
        )
        return figure

    def plot_parameter_and_gradient_distributions(
        self,
        epoch: int | None = None,
    ) -> go.Figure:
        """Create side-by-side parameter and gradient histograms."""
        diagnostics = self._diagnostics_for_epoch(epoch)
        titles = [
            title
            for name, _, _ in diagnostics
            for title in (f"{name} - parameters", f"{name} - gradients")
        ]
        figure = make_subplots(
            rows=len(diagnostics),
            cols=2,
            subplot_titles=titles,
        )

        for row, (name, parameter, gradient) in enumerate(diagnostics, start=1):
            figure.add_trace(
                go.Histogram(
                    x=parameter.flatten().tolist(),
                    nbinsx=50,
                    name=f"{name} parameters",
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

            if gradient is None:
                figure.add_annotation(
                    text="No gradient recorded",
                    showarrow=False,
                    row=row,
                    col=2,
                )
            else:
                figure.add_trace(
                    go.Histogram(
                        x=gradient.flatten().tolist(),
                        nbinsx=50,
                        name=f"{name} gradients",
                        showlegend=False,
                    ),
                    row=row,
                    col=2,
                )

        figure.update_layout(
            height=max(500, 260 * len(diagnostics)),
            title=(
                "Current parameter and gradient distributions"
                if epoch is None
                else f"Parameter and gradient distributions - epoch {epoch}"
            ),
            template="plotly_white",
        )
        figure.update_xaxes(
            exponentformat="power",
            showexponent="all",
        )
        figure.update_yaxes(
            exponentformat="power",
            showexponent="all",
        )
        return figure

    def save_figures(self, best_epoch: int, directory: str | Path) -> None:
        """Save training history and best-epoch diagnostic distributions."""
        best_epoch = int(best_epoch)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.plot_losses().write_html(
            directory / "training_losses.html",
            auto_open=False,
        )
        self.plot_parameter_and_gradient_distributions(best_epoch).write_html(
            directory
            / f"parameter_and_gradient_distributions_epoch_{best_epoch}.html",
            auto_open=False,
        )
