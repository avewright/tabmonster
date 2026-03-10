"""Trunk weight transfer utilities for cross-dataset curriculum training.

When the curriculum worker moves from one dataset to the next the feature-specific
input heads (numeric projections, categorical embeddings) and the output head change
size, but the shared transformer backbone layers are reusable.

:func:`load_trunk_weights` adopts a *shape-matching* strategy:

* It loads the full checkpoint state dict.
* For every parameter present in *both* the checkpoint and the live model it checks
  whether the tensor shapes match.
* Matching parameters are copied; mismatching ones are silently skipped and the model
  retains its freshly-initialised weights.

This is intentionally permissive — it will always do *something* useful even when the
old and new datasets have very different schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_trunk_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    """Partially load weights from *checkpoint_path* into *model*.

    Only parameters whose **name and shape both match** between the checkpoint and the
    live model are transferred.  All others are left as-is (random init).

    Parameters
    ----------
    model:
        The freshly initialised :class:`~tabula.models.transformer.TabularTransformer`
        (or episodic variant) that should receive the trunk weights.
    checkpoint_path:
        Path to a ``.pt`` file saved by
        :func:`~tabula.training.engine._save_checkpoint`.
    device:
        Device to map the checkpoint tensors onto before comparing.  Should match the
        device *model* lives on.
    verbose:
        If ``True`` print a short summary of transferred / skipped / missing layers.

    Returns
    -------
    dict
        ``{"transferred": [...], "skipped_shape": [...], "skipped_missing": [...]}``
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Trunk checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)
    ckpt_state: dict[str, torch.Tensor] = payload.get("model_state_dict", payload)

    live_state = model.state_dict()

    transferred: list[str] = []
    skipped_shape: list[str] = []
    skipped_missing: list[str] = []

    new_state = {k: v.clone() for k, v in live_state.items()}

    for name, ckpt_param in ckpt_state.items():
        if name not in live_state:
            skipped_missing.append(name)
            continue
        if ckpt_param.shape != live_state[name].shape:
            skipped_shape.append(name)
            continue
        new_state[name] = ckpt_param.to(device=live_state[name].device, dtype=live_state[name].dtype)
        transferred.append(name)

    model.load_state_dict(new_state, strict=True)

    summary: dict[str, Any] = {
        "checkpoint": str(path),
        "transferred": transferred,
        "skipped_shape_mismatch": skipped_shape,
        "skipped_not_in_model": skipped_missing,
        "transferred_count": len(transferred),
        "skipped_shape_count": len(skipped_shape),
        "skipped_missing_count": len(skipped_missing),
        "skipped_not_in_model_count": len(skipped_missing),
    }

    if verbose:
        total = len(transferred) + len(skipped_shape) + len(skipped_missing)
        print(
            f"[trunk] Transferred {len(transferred)}/{total} parameter tensors from {path.name} "
            f"({len(skipped_shape)} shape mismatch, {len(skipped_missing)} not in model)."
        )

    return summary
