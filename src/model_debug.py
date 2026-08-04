"""Plotly helpers for tracking and visualising PyTorch model parameters."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

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

    def plot_losses(
        self,
        train_losses: Iterable[float],
        validation_losses: Iterable[float],
    ) -> go.Figure:
        """Create loss curves from externally managed metric histories."""
        train_history = [float(loss) for loss in train_losses]
        validation_history = [float(loss) for loss in validation_losses]

        if not train_history or not validation_history:
            raise RuntimeError("Train and validation loss histories cannot be empty")
        if len(train_history) != len(validation_history):
            raise ValueError(
                "Train and validation loss histories must contain the same number "
                "of epochs"
            )

        epochs = list(range(len(train_history)))
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

    def plot_distributions(self) -> go.Figure:
        """Create final parameter and gradient histograms for every tensor."""
        parameters = list(self.model.named_parameters())
        titles = [title for name, _ in parameters for title in (name, f"{name} gradient")]
        figure = make_subplots(rows=len(parameters), cols=2, subplot_titles=titles)

        for row, (name, parameter) in enumerate(parameters, start=1):
            values = parameter.detach().cpu().flatten().tolist()
            figure.add_trace(
                go.Histogram(x=values, nbinsx=50, name=name, showlegend=False), row=row, col=1
            )
            if parameter.grad is not None:
                gradients = parameter.grad.detach().cpu().flatten().tolist()
                figure.add_trace(
                    go.Histogram(
                        x=gradients,
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
            title="Final parameter and gradient distributions",
            template="plotly_white",
        )
        return figure

    def plot_weight_heatmaps(self) -> go.Figure:
        """Create heatmaps for all two-dimensional parameters."""
        matrices = [
            (name, parameter.detach().cpu().numpy())
            for name, parameter in self.model.named_parameters()
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
            title="Model weight heatmaps",
            template="plotly_white",
        )
        return figure

    def show_all(
        self,
        train_losses: Iterable[float],
        validation_losses: Iterable[float],
    ) -> dict[str, go.Figure]:
        """Display every diagnostic plot and return the figures by name."""
        figures = {
            "losses": self.plot_losses(train_losses, validation_losses),
            "norms": self.plot_norms(),
            "distributions": self.plot_distributions(),
            "weight_heatmaps": self.plot_weight_heatmaps(),
        }
        for figure in figures.values():
            figure.show()
        return figures
