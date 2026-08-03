"""Plotly helpers for tracking and visualising PyTorch model parameters."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


class ModelDebugger:
    """Collect training diagnostics and show them as interactive Plotly figures."""

    def __init__(
        self,
        model: torch.nn.Module,
        embedding_labels: Sequence[str] | None = None,
        embedding_layer: str = "token_embedding",
    ):
        self.model = model
        self.embedding_labels = (
            None if embedding_labels is None else list(embedding_labels)
        )
        self.embedding_layer = embedding_layer
        self.steps: list[int] = []
        self.parameter_norms: dict[str, list[float]] = defaultdict(list)
        self.gradient_norms: dict[str, list[float]] = defaultdict(list)
        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []

    @staticmethod
    def _device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    @staticmethod
    def _evaluate(
        model: torch.nn.Module,
        loader: Iterable,
        loss_fn: torch.nn.Module,
    ) -> float:
        """Return sample-weighted mean loss without changing model parameters."""
        was_training = model.training
        device = ModelDebugger._device(model)
        total_loss = 0.0
        total_examples = 0

        model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size = inputs.size(0)
                total_loss += loss_fn(model(inputs), targets).item() * batch_size
                total_examples += batch_size

        model.train(was_training)
        if total_examples == 0:
            raise ValueError("Cannot evaluate loss with an empty data loader")
        return total_loss / total_examples

    def record_initial_losses(
        self,
        train_loader: Iterable,
        validation_loader: Iterable,
        loss_fn: torch.nn.Module,
    ) -> tuple[float, float]:
        """Evaluate and store epoch-0 losses before any training takes place."""
        if self.train_losses or self.validation_losses:
            raise RuntimeError("Initial losses have already been recorded")

        train_loss = self._evaluate(self.model, train_loader, loss_fn)
        validation_loss = self._evaluate(self.model, validation_loader, loss_fn)
        self.train_losses.append(train_loss)
        self.validation_losses.append(validation_loss)
        return train_loss, validation_loss

    def record_step(self, step: int) -> None:
        """Record norms after loss.backward() and before optimizer.step()."""
        self.steps.append(int(step))
        for name, parameter in self.model.named_parameters():
            self.parameter_norms[name].append(parameter.detach().norm().item())
            gradient = parameter.grad
            self.gradient_norms[name].append(
                float("nan") if gradient is None else gradient.detach().norm().item()
            )

    def record_epoch(self, train_loss: float, validation_loss: float) -> None:
        """Record losses after a completed training epoch."""
        self.train_losses.append(float(train_loss))
        self.validation_losses.append(float(validation_loss))

    def diagnose(self) -> list[str]:
        """Return warnings for non-finite values and missing gradients."""
        warnings = []
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter.detach()).all():
                warnings.append(f"{name}: parameter contains NaN or Inf")
            if parameter.grad is None:
                warnings.append(f"{name}: gradient is missing")
            elif not torch.isfinite(parameter.grad.detach()).all():
                warnings.append(f"{name}: gradient contains NaN or Inf")
        return warnings

    def plot_losses(self) -> go.Figure:
        """Create train/validation loss curves, including epoch 0."""
        if not self.train_losses:
            raise RuntimeError("No losses recorded; call record_initial_losses first")

        epochs = list(range(len(self.train_losses)))
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=epochs, y=self.train_losses, mode="lines+markers", name="Train loss"
        ))
        figure.add_trace(go.Scatter(
            x=epochs,
            y=self.validation_losses,
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

    def plot_embedding_map(self, show_labels: bool = True) -> go.Figure:
        """Project cosine-normalized embedding vectors into two dimensions with PCA."""
        if self.embedding_labels is None:
            raise RuntimeError(
                "No embedding labels supplied; construct ModelDebugger with "
                "embedding_labels=champ_names"
            )

        layer = getattr(self.model, self.embedding_layer, None)
        if not isinstance(layer, torch.nn.Embedding):
            raise ValueError(
                f"model.{self.embedding_layer} is not a torch.nn.Embedding layer"
            )

        vectors = layer.weight.detach().float().cpu()
        if len(self.embedding_labels) != vectors.size(0):
            raise ValueError(
                f"Received {len(self.embedding_labels)} labels for "
                f"{vectors.size(0)} embedding vectors"
            )
        if vectors.size(0) < 2 or vectors.size(1) < 2:
            raise ValueError("PCA requires at least two vectors with two dimensions")

        # Unit-normalising means proximity primarily reflects cosine similarity,
        # rather than differences in embedding magnitude.
        vector_norms = vectors.norm(dim=1)
        normalized = vectors / vector_norms.clamp_min(1e-12).unsqueeze(1)
        centered = normalized - normalized.mean(dim=0, keepdim=True)
        _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ components[:2].T

        variances = singular_values.square()
        explained = variances[:2] / variances.sum().clamp_min(1e-12) * 100
        x = coordinates[:, 0].tolist()
        y = coordinates[:, 1].tolist()
        norms = vector_norms.tolist()

        figure = go.Figure(go.Scatter(
            x=x,
            y=y,
            mode="markers+text" if show_labels else "markers",
            text=self.embedding_labels if show_labels else None,
            textposition="top center",
            textfont={"size": 9},
            customdata=[[name, norm] for name, norm in zip(self.embedding_labels, norms)],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>"
                "Embedding norm: %{customdata[1]:.3f}<extra></extra>"
            ),
            marker={
                "size": 8,
                "color": norms,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Vector norm"},
                "line": {"width": 0.5, "color": "white"},
            },
        ))
        figure.update_layout(
            title="Champion embedding similarity (cosine-normalized PCA)",
            xaxis_title=f"PC1 ({explained[0].item():.1f}% variance)",
            yaxis_title=f"PC2 ({explained[1].item():.1f}% variance)",
            template="plotly_white",
            height=800,
            hovermode="closest",
        )
        return figure

    def show_all(self) -> dict[str, go.Figure]:
        """Display every diagnostic plot and return the figures by name."""
        figures = {
            "losses": self.plot_losses(),
            "norms": self.plot_norms(),
            "distributions": self.plot_distributions(),
            "weight_heatmaps": self.plot_weight_heatmaps(),
        }
        if self.embedding_labels is not None:
            figures["embedding_map"] = self.plot_embedding_map()
        for figure in figures.values():
            figure.show()
        return figures

    def save_all(self, directory: str | Path = "debug_plots") -> None:
        """Save every diagnostic as a standalone interactive HTML file."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        figures = {
            "losses": self.plot_losses(),
            "norms": self.plot_norms(),
            "distributions": self.plot_distributions(),
            "weight_heatmaps": self.plot_weight_heatmaps(),
        }
        if self.embedding_labels is not None:
            figures["embedding_map"] = self.plot_embedding_map()
        for name, figure in figures.items():
            figure.write_html(output / f"{name}.html")
