"""Forward-hook capture for TRM's recursive reasoning module.

TRM has a single shared 2-layer transformer block (`inner.L_level`) that gets
called repeatedly per puzzle:

  per Inner.forward (= one supervision step):
    for h in range(H_cycles):                  # 3
      for l in range(L_cycles):                # 4
        z_L = L_level(z_L, z_H + input_emb)    # role='latent'
      z_H = L_level(z_H, z_L)                  # role='answer'

  at ARC eval: 16 supervision steps × 15 calls = 240 calls per puzzle.

This module gives you a context manager that wraps the model, registers
forward hooks on `inner.L_level` (and its attention/mlp submodules + the
puzzle_emb / embed_tokens entry points), and yields a list of captured
records — each tagged with `(supervision_step, h_cycle, l_step, role)` so
downstream code (Exp A convergence curves, Exp B probes, Exp D SAE shards)
can index along whichever axis it needs.

Usage:
    from interp.hooks import HookContext
    with HookContext(model) as hc:
        carry = model.initial_carry(batch)
        for step in range(meta.halt_max_steps):
            hc.begin_supervision_step(step)
            carry, out = model(carry, batch)
            if carry.halted.all():
                break
        records = hc.records  # list[Capture]

Records are numpy arrays (detached, float32) ready for ActivationStorage.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Iterator, Literal

import numpy as np
import torch
from torch import nn

Role = Literal["latent", "answer", "embed", "puzzle_emb", "lm_head"]


@dataclasses.dataclass
class Capture:
    supervision_step: int       # outer ACT step ∈ [0, halt_max_steps)
    h_cycle: int                # H_cycles index ∈ [0, H_cycles)
    l_step: int                 # within-h-cycle index: ∈ [0, L_cycles) for latent, L_cycles for answer
    role: Role
    site: str                   # module name in the L_level call graph
    activation: np.ndarray      # shape (seq_len_total, hidden_size), float32

    @property
    def latent_update_index(self) -> int:
        """Flat index of latent updates per supervision step. Only valid for role='latent'."""
        return self.h_cycle * 12 + self.l_step  # 12 = max L_cycles seen across configs

    def key(self) -> tuple[int, int, int, str, str]:
        return (self.supervision_step, self.h_cycle, self.l_step, self.role, self.site)


class HookContext:
    """Wrap a TRM model and capture every L_level call's output (+ submodule outputs).

    The trick: TRM doesn't expose iteration indices to its hooks, so we maintain
    a manual counter that the caller bumps via `begin_supervision_step()`. Within
    a step, we observe call ordering: the first `L_cycles` L_level calls are
    latent (z_L) updates, the next call is an answer (z_H) update, then repeat
    for H_cycles. We mirror that state machine.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        capture_attn: bool = False,
        capture_mlp: bool = False,
        capture_puzzle_emb: bool = True,
    ) -> None:
        self.model = model
        self.inner = model.inner
        self.H_cycles = model.config.H_cycles
        self.L_cycles = model.config.L_cycles
        self.records: list[Capture] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._supervision_step: int = -1
        self._call_idx_within_step: int = 0
        self.capture_attn = capture_attn
        self.capture_mlp = capture_mlp
        self.capture_puzzle_emb = capture_puzzle_emb

    # -- public API -------------------------------------------------------

    def __enter__(self) -> "HookContext":
        self._register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def begin_supervision_step(self, step: int) -> None:
        self._supervision_step = step
        self._call_idx_within_step = 0

    # -- internal --------------------------------------------------------

    def _register(self) -> None:
        # L_level: the headline hook (one call per latent/answer update).
        self._handles.append(
            self.inner.L_level.register_forward_hook(self._on_l_level)
        )
        if self.capture_puzzle_emb and hasattr(self.inner, "puzzle_emb"):
            self._handles.append(
                self.inner.puzzle_emb.register_forward_hook(self._on_puzzle_emb)
            )
        for li, layer in enumerate(self.inner.L_level.layers):
            if self.capture_attn and hasattr(layer, "self_attn"):
                self._handles.append(
                    layer.self_attn.register_forward_hook(self._make_submodule_hook(f"L_layer{li}.attn"))
                )
            if self.capture_mlp:
                self._handles.append(
                    layer.mlp.register_forward_hook(self._make_submodule_hook(f"L_layer{li}.mlp"))
                )

    def _classify_call(self) -> tuple[int, int, Role]:
        """Map a flat call index (0..H_cycles*(L_cycles+1)-1) → (h_cycle, l_step, role)."""
        n = self._call_idx_within_step
        block = self.L_cycles + 1
        h = n // block
        within = n % block
        if within < self.L_cycles:
            return h, within, "latent"
        return h, self.L_cycles, "answer"

    def _on_l_level(self, _module: nn.Module, _inp, output: torch.Tensor) -> None:
        if self._supervision_step < 0:
            return  # caller forgot begin_supervision_step()
        h, l, role = self._classify_call()
        self._record(output, h=h, l=l, role=role, site="L_level")
        self._call_idx_within_step += 1

    def _make_submodule_hook(self, site: str):
        def _hook(_module, _inp, output):
            if self._supervision_step < 0:
                return
            # The submodule hook fires *before* its parent L_level hook completes,
            # so we read the still-pending (h, l, role) from the current counter.
            h, l, role = self._classify_call()
            self._record(output, h=h, l=l, role=role, site=site)
        return _hook

    def _on_puzzle_emb(self, _module, _inp, output: torch.Tensor) -> None:
        if self._supervision_step < 0:
            return
        # puzzle_emb fires once per Inner.forward, *before* any L_level call.
        self._record(output, h=-1, l=-1, role="puzzle_emb", site="puzzle_emb")

    def _record(self, t: torch.Tensor, *, h: int, l: int, role: Role, site: str) -> None:
        if isinstance(t, tuple):
            t = t[0]
        arr = t.detach().to(torch.float32).cpu().numpy()
        # Strip batch dim when batch_size=1 to keep shards 2D (seq_len, hidden).
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        self.records.append(
            Capture(
                supervision_step=self._supervision_step,
                h_cycle=h,
                l_step=l,
                role=role,
                site=site,
                activation=arr,
            )
        )


@contextmanager
def capture(model: nn.Module, **kw) -> Iterator[HookContext]:
    """Context manager sugar."""
    hc = HookContext(model, **kw)
    with hc:
        yield hc


__all__ = ["Capture", "HookContext", "capture"]
