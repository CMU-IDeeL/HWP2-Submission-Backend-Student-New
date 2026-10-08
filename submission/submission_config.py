from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any

import torch.nn as nn

# Flax is only required for the JAX notebook, which installs jax-ai-stack
try:
    from flax import nnx
except Exception:
    nnx = None

_MODEL_TYPES = (nn.Module, nnx.Module) if nnx is not None else (nn.Module,)


@dataclass(slots=True)
class SubmissionConfig:
    """
    Configuration object defining all inputs required to produce a
    final assignment submission artifact.

    This object is intentionally *not* frozen because it may be
    constructed incrementally in a notebook. However, it should be
    treated as immutable once passed into the submission pipeline.

    Notes
    -----
    • The `model` field is intentionally excluded from serialization.
    • All file paths are stored as strings for notebook friendliness.
    • Validation is performed eagerly to catch common student errors.
    """

    # ------------------------------------------------------------------
    # Model Details
    # ------------------------------------------------------------------
    model: nn.Module | nnx.Module = field(
        metadata={"description": "PyTorch model or flax NNX model used for training and evaluation"}
    )

    # ------------------------------------------------------------------
    # Kaggle Details
    # ------------------------------------------------------------------
    kaggle_username: str = field(
        metadata={"description": "Kaggle username used for competition submissions"}
    )
    kaggle_api_key: str = field(
        repr=False,
        metadata={"description": "Kaggle API key (never serialized or logged)"},
    )

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    wandb_api_key: str = field(
        repr=False,
        metadata={"description": "W&B API key (never serialized or logged)"},
    )
    wandb_entity: str = field(
        metadata={"description": "W&B username or team name"}
    )
    wandb_project: str = field(
        metadata={"description": "W&B project containing experiment runs"}
    )

    # ------------------------------------------------------------------
    # Submission Metadata
    # ------------------------------------------------------------------
    acknowledged: bool = field(
        metadata={"description": "Student has accepted submission acknowledgement"}
    )
    readme: str = field(
        metadata={"description": "README text included in submission"}
    )
    notebook_path: str = field(
        metadata={"description": "Path to the main notebook file"}
    )
    additional_files: List[str] = field(
        default_factory=list,
        metadata={"description": "Additional files to include in submission ZIP"},
    )
    wandb_runs_override: List[str] = field(
        default_factory=list,
        metadata={"description": "Optional explicit W&B run selection"},
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.model, _MODEL_TYPES):
            raise TypeError("model must be an instance of torch.nn.Module or flax.nnx.Module")

        if not self.kaggle_username.strip():
            raise ValueError("kaggle_username must be a non-empty string")

        if not self.notebook_path.strip():
            raise ValueError("notebook_path must be provided")

        if not isinstance(self.additional_files, list):
            raise TypeError("additional_files must be a list of file paths")
        
        if not isinstance(self.wandb_runs_override, list,):
            raise TypeError("wandb_runs_override must be a list")

        for run_path in self.wandb_runs_override:
            if (
                not isinstance(run_path, str)
                or not run_path.strip()
            ):
                raise ValueError(
                    "Each W&B override must be a non-empty "
                    "'entity/project/run_id' string"
                )
            
    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration to a dictionary suitable for JSON/YAML.

        Sensitive fields (API keys) and non-serializable objects (model)
        are excluded by design.
        """
        data = asdict(self)

        # Remove non-serializable / sensitive fields
        data.pop("model", None)
        data.pop("kaggle_api_key", None)
        data.pop("wandb_api_key", None)

        return data

    def to_json(self, path: str | Path) -> None:
        """
        Serialize the configuration to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output JSON file path.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        model: nn.Module,
        kaggle_api_key: str,
        wandb_api_key: str,
    ) -> "SubmissionConfig":
        """
        Reconstruct a SubmissionConfig from a serialized dictionary.

        Sensitive fields and the model must be supplied explicitly.
        """
        return cls(
            model=model,
            kaggle_api_key=kaggle_api_key,
            wandb_api_key=wandb_api_key,
            **data,
        )
