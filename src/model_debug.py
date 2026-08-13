"""Plotly helpers for tracking and visualising PyTorch training diagnostics."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


class ModelDebugger:
    """Collect training diagnostics and show them as interactive Plotly figures."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.epoch_parameters: dict[int, dict[str, torch.Tensor]] = {}
        self.epoch_log_update_ratios: dict[
            int,
            dict[str, torch.Tensor | None],
        ] = {}
        self.epoch_train_losses: dict[int, float] = {}
        self.epoch_validation_losses: dict[int, float] = {}
        self._parameters_before_step: dict[str, torch.Tensor] | None = None
        self._last_parameters_before_step: dict[str, torch.Tensor] | None = None
        self._optimizer_hook_handles = (
            optimizer.register_step_pre_hook(self._before_optimizer_step),
            optimizer.register_step_post_hook(self._after_optimizer_step),
        )

    def _before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Snapshot parameters immediately before an optimizer step."""
        self._parameters_before_step = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

    def _after_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Retain the pre-step snapshot for the completed optimizer step."""
        if self._parameters_before_step is None:
            raise RuntimeError(
                "Optimizer post-step hook ran without a pre-step snapshot"
            )

        self._last_parameters_before_step = self._parameters_before_step
        self._parameters_before_step = None

    def record_epoch(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Store losses, parameters, and the most recent update diagnostics."""
        epoch = int(epoch)
        named_parameters = list(self.model.named_parameters())

        self.epoch_train_losses[epoch] = float(train_loss)
        self.epoch_validation_losses[epoch] = float(val_loss)
        self.epoch_parameters[epoch] = {}
        self.epoch_log_update_ratios[epoch] = {}

        for name, parameter in named_parameters:
            parameter_snapshot = parameter.detach().cpu().clone()
            self.epoch_parameters[epoch][name] = parameter_snapshot

            if self._last_parameters_before_step is None:
                self.epoch_log_update_ratios[epoch][name] = None
                continue

            parameter_before = self._last_parameters_before_step[name]
            update = parameter.detach() - parameter_before
            parameter_rms = parameter_before.square().mean().sqrt()
            epsilon = torch.finfo(parameter.dtype).eps
            log_update_ratio = torch.log10(
                (update.abs() + epsilon) / (parameter_rms + epsilon)
            )
            self.epoch_log_update_ratios[epoch][name] = (
                log_update_ratio.detach().cpu().clone()
            )

        self._last_parameters_before_step = None

    def close(self) -> None:
        """Remove the optimizer hooks registered by this debugger."""
        for handle in self._optimizer_hook_handles:
            handle.remove()
        self._optimizer_hook_handles = ()

    def _diagnostics_for_epoch(
        self,
        epoch: int,
    ) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
        epoch = int(epoch)
        if epoch not in self.epoch_parameters:
            available = sorted(self.epoch_parameters)
            raise KeyError(
                f"No parameters recorded for epoch {epoch}. "
                f"Available epochs: {available}"
            )
        log_update_ratios = self.epoch_log_update_ratios[epoch]
        return [
            (
                name,
                parameter.clone(),
                None
                if log_update_ratios[name] is None
                else log_update_ratios[name].clone(),
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

    def plot_parameter_and_update_ratio_distributions(
        self,
        epoch: int | None = None,
    ) -> go.Figure:
        """Create side-by-side parameter and log update-ratio histograms."""
        if epoch is None:
            if not self.epoch_parameters:
                raise RuntimeError(
                    "No epochs recorded; call record_epoch during training"
                )
            epoch = max(self.epoch_parameters)
        epoch = int(epoch)
        diagnostics = self._diagnostics_for_epoch(epoch)
        titles = [
            title
            for name, _, _ in diagnostics
            for title in (
                f"{name} - parameters",
                f"{name} - log(update ratio)",
            )
        ]
        figure = make_subplots(
            rows=len(diagnostics),
            cols=2,
            subplot_titles=titles,
        )

        for row, (name, parameter, log_update_ratio) in enumerate(
            diagnostics,
            start=1,
        ):
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

            if log_update_ratio is None:
                figure.add_annotation(
                    text="No optimizer update recorded",
                    showarrow=False,
                    row=row,
                    col=2,
                )
            else:
                figure.add_trace(
                    go.Histogram(
                        x=log_update_ratio.flatten().tolist(),
                        nbinsx=50,
                        name=f"{name} log(update ratio)",
                        showlegend=False,
                    ),
                    row=row,
                    col=2,
                )

        figure.update_layout(
            height=max(500, 260 * len(diagnostics)),
            title=f"Parameter and log(update ratio) distributions - epoch {epoch}",
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
        self.plot_parameter_and_update_ratio_distributions(best_epoch).write_html(
            directory
            / f"parameter_and_update_ratio_distributions_epoch_{best_epoch}.html",
            auto_open=False,
        )
