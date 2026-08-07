"""Plotly helpers for tracking and visualising PyTorch model parameters."""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


class ModelDebugger:
    """Collect training diagnostics and show them as interactive Plotly figures."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.steps: list[int] = []
        self.parameter_norms: dict[str, list[float]] = defaultdict(list)
        self.gradient_norms: dict[str, list[float]] = defaultdict(list)
        self.epoch_parameters: dict[int, dict[str, torch.Tensor]] = {}
        self.epoch_gradients: dict[int, dict[str, torch.Tensor | None]] = {}
        self.epoch_train_losses: dict[int, float] = {}
        self.epoch_validation_losses: dict[int, float] = {}

    def record_step(self, step: int | None = None) -> None:
        """Record norms, automatically numbering snapshots when no step is supplied."""
        if step is None:
            step = 0 if not self.steps else self.steps[-1] + 1
        self.steps.append(int(step))
        for name, parameter in self.model.named_parameters():
            self.parameter_norms[name].append(parameter.detach().norm().item())
            gradient = parameter.grad
            self.gradient_norms[name].append(
                float("nan") if gradient is None else gradient.detach().norm().item()
            )

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Store losses plus independent parameter and gradient snapshots for an epoch."""
        epoch = int(epoch)
        self.epoch_train_losses[epoch] = float(train_loss)
        self.epoch_validation_losses[epoch] = float(val_loss)
        self.epoch_parameters[epoch] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }
        self.epoch_gradients[epoch] = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
        }

    def get_parameters(self, epoch: int) -> dict[str, torch.Tensor]:
        """Return independent copies of the parameter tensors recorded for an epoch."""
        epoch = int(epoch)
        if epoch not in self.epoch_parameters:
            available = sorted(self.epoch_parameters)
            raise KeyError(
                f"No parameters recorded for epoch {epoch}. "
                f"Available epochs: {available}"
            )
        return {
            name: parameter.clone()
            for name, parameter in self.epoch_parameters[epoch].items()
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
        return list(self.get_parameters(epoch).items())

    def _gradients_for_epoch(
        self,
        epoch: int | None,
    ) -> dict[str, torch.Tensor | None]:
        if epoch is None:
            return {
                name: None if parameter.grad is None else parameter.grad.detach().cpu()
                for name, parameter in self.model.named_parameters()
            }
        epoch = int(epoch)
        if epoch not in self.epoch_gradients:
            available = sorted(self.epoch_gradients)
            raise KeyError(
                f"No gradients recorded for epoch {epoch}. "
                f"Available epochs: {available}"
            )
        return self.epoch_gradients[epoch]

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

    def plot_norms(self) -> go.Figure:
        """Create parameter- and gradient-norm plots."""
        if not self.steps:
            raise RuntimeError("No step statistics recorded; call record_step during training")

        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Parameter norms", "Gradient norms"),
        )
        for name, values in self.parameter_norms.items():
            figure.add_trace(
                go.Scatter(x=self.steps, y=values, mode="lines", name=name), row=1, col=1
            )
            figure.add_trace(
                go.Scatter(
                    x=self.steps,
                    y=self.gradient_norms[name],
                    mode="lines",
                    name=name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        figure.update_yaxes(type="log", title_text="L2 norm", row=1, col=1)
        figure.update_yaxes(type="log", title_text="L2 norm", row=2, col=1)
        figure.update_xaxes(title_text="Training step", row=2, col=1)
        figure.update_layout(height=750, title="Model and gradient norms", template="plotly_white")
        return figure

    def plot_distributions(self, epoch: int | None = None) -> go.Figure:
        """Create parameter and gradient histograms for the current or selected epoch."""
        parameters = self._parameters_for_epoch(epoch)
        gradients = self._gradients_for_epoch(epoch)
        titles = [title for name, _ in parameters for title in (name, f"{name} gradient")]
        figure = make_subplots(rows=len(parameters), cols=2, subplot_titles=titles)

        for row, (name, parameter) in enumerate(parameters, start=1):
            values = parameter.flatten().tolist()
            figure.add_trace(
                go.Histogram(x=values, nbinsx=50, name=name, showlegend=False), row=row, col=1
            )
            gradient = gradients[name]
            if gradient is not None:
                gradient_values = gradient.flatten().tolist()
                figure.add_trace(
                    go.Histogram(
                        x=gradient_values,
                        nbinsx=50,
                        name=f"{name} gradient",
                        marker_color="orange",
                        showlegend=False,
                    ),
                    row=row,
                    col=2,
                )

        figure.update_layout(
            height=max(500, 260 * len(parameters)),
            title=(
                "Current parameter and gradient distributions"
                if epoch is None
                else f"Parameter and gradient distributions — epoch {epoch}"
            ),
            template="plotly_white",
        )
        return figure

    def plot_weight_heatmaps(self, epoch: int | None = None) -> go.Figure:
        """Create heatmaps for two-dimensional parameters at the current or selected epoch."""
        matrices = [
            (name, parameter.numpy())
            for name, parameter in self._parameters_for_epoch(epoch)
            if parameter.ndim == 2
        ]
        if not matrices:
            raise RuntimeError("The model has no two-dimensional parameters")

        figure = make_subplots(
            rows=len(matrices),
            cols=1,
            subplot_titles=[f"{name} — shape {matrix.shape}" for name, matrix in matrices],
        )
        for row, (name, matrix) in enumerate(matrices, start=1):
            limit = max(abs(float(matrix.min())), abs(float(matrix.max())), 1e-12)
            figure.add_trace(
                go.Heatmap(
                    z=matrix,
                    zmin=-limit,
                    zmax=limit,
                    colorscale="RdBu",
                    colorbar={"title": name, "len": 1 / len(matrices)},
                ),
                row=row,
                col=1,
            )
        figure.update_layout(
            height=max(500, 340 * len(matrices)),
            title=(
                "Current model weight heatmaps"
                if epoch is None
                else f"Model weight heatmaps — epoch {epoch}"
            ),
            template="plotly_white",
        )
        return figure

    def show_all(
        self,
        epoch: int | None = None,
    ) -> dict[str, go.Figure]:
        """Display every diagnostic plot and return the figures by name."""
        figures = {
            "losses": self.plot_losses(),
            "norms": self.plot_norms(),
            "distributions": self.plot_distributions(epoch=epoch),
            "weight_heatmaps": self.plot_weight_heatmaps(epoch=epoch),
        }
        for figure in figures.values():
            figure.show()
        return figures
