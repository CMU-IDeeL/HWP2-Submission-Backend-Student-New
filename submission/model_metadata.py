from __future__ import annotations

import torch
from torch import nn
from typing import Dict, Any
import logging

# Flax/JAX are only required for the JAX notebook (jax-ai-stack). Colab's
# default jax/flax pair is incompatible and must not break the torch path.
try:
    import jax
    from flax import nnx
except Exception:
    jax = None
    nnx = None

log = logging.getLogger(__name__)


###############################################################################
# Errors
###############################################################################

class ModelMetadataError(RuntimeError):
    """Raised when model metadata cannot be constructed."""


###############################################################################
# Public API
###############################################################################

def build_model_metadata(model: nn.Module | nnx.Module, param_limit: int) -> Dict[str, Any]:
    """
    Build a deterministic, serializable metadata dictionary for a PyTorch
    or Flax NNX model.

    This metadata is used for submission auditing and grading and must:
      • Be side-effect free
      • Be JSON-serializable
      • Fail early on invalid models

    Args:
        model (nn.Module | nnx.Module): The model to inspect
        param_limit (int): Maximum allowed number of trainable parameters
    Returns:
        Dict[str, Any]: Model metadata
    """

    if isinstance(model, nn.Module):
        try:
            parameters = list(model.parameters())
        except Exception as exc:
            raise ModelMetadataError(
                "Failed to access model parameters.\n"
                "👉 Ensure your model is a valid nn.Module."
            ) from exc

        if not parameters:
            raise ModelMetadataError(
                "Model contains no parameters.\n"
                "👉 Ensure your model defines trainable layers."
            )

        trainable_params = [
            p for p in parameters if p.requires_grad
        ]

        total_params = sum(p.numel() for p in parameters)
        trainable_count = sum(p.numel() for p in trainable_params)

        if trainable_count > param_limit:
            raise ModelMetadataError(
                f"Model exceeds parameter limit of {param_limit}.\n"
                "👉 Reduce model size or complexity."
            )

        framework_info = {
            "backend_name": "torch",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }

    elif nnx is not None and isinstance(model, nnx.Module):
        try:
            param_state = nnx.state(model, nnx.Param)
            param_leaves = jax.tree_util.tree_leaves(param_state)
        except Exception as exc:
            raise ModelMetadataError(
                "Failed to access model parameters.\n"
                "👉 Ensure your model is a valid nnx.Module."
            ) from exc

        if not param_leaves:
            raise ModelMetadataError(
                "Model contains no parameters.\n"
                "👉 Ensure your model defines trainable layers."
            )

        # NNX has no per-parameter requires_grad flag like PyTorch — all
        # nnx.Param leaves are trainable by convention (frozen weights are
        # typically stored as nnx.Variable/nnx.Cache instead of nnx.Param).
        total_params = sum(int(leaf.size) for leaf in param_leaves)
        trainable_count = total_params

        if trainable_count > param_limit:
            raise ModelMetadataError(
                f"Model exceeds parameter limit of {param_limit}.\n"
                "👉 Reduce model size or complexity."
            )

        framework_info = {
            "backend_name": "jax",
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
        }

    else:
        raise ModelMetadataError(
            "Invalid model object.\n"
            "👉 Expected a torch.nn.Module or flax.nnx.Module instance."
        )

    metadata: Dict[str, Any] = {
        "schema_version": "1.0",

        "model": {
            "class_name": model.__class__.__name__,
            "module": model.__class__.__module__,
            "repr": repr(model),
            "trainable_parameters": trainable_count,
            "total_parameters": total_params,
        },

        "framework": framework_info,
    }

    log.info(
        "Model metadata built: %d trainable parameters",
        metadata["model"]["trainable_parameters"],
    )

    return metadata