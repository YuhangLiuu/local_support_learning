"""Bounded observation budgets; no algorithm hyperparameters live here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = False
    level: str = "basic"
    every_n_forwards: int = 100
    max_batches_per_context: int = 4
    max_tokens_per_batch: int = 512
    histogram_edges: tuple = (-100., -10., -1., -0.1, 0., 0.1, 1., 10., 100.)
    near_zero_margin: float = 0.1
    output_dir: str = ""
    save_token_traces: bool = False
    max_trace_rows: int = 32

    def __post_init__(self):
        if self.level not in {"basic", "extended", "probe", "profile"}:
            raise ValueError("level must be basic, extended, probe or profile")
        if min(self.every_n_forwards, self.max_batches_per_context,
               self.max_tokens_per_batch, self.max_trace_rows) <= 0:
            raise ValueError("diagnostic budgets must be positive")
        if self.near_zero_margin < 0:
            raise ValueError("near_zero_margin must be nonnegative")
        if any(a >= b for a, b in zip(self.histogram_edges, self.histogram_edges[1:])):
            raise ValueError("histogram edges must increase")

    @property
    def extended(self):
        return self.level in {"extended", "probe"}
