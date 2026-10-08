from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import json
import os
from enum import Enum


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """
    Immutable configuration object controlling experiment ranking,
    Kaggle submission metadata, and final packaging behavior.

    This class is intentionally immutable to prevent accidental mutation
    during the submission pipeline.
    """

    # ------------------------------------------------------------------
    # Evaluation / Ranking
    # ------------------------------------------------------------------
    param_limit: int = field(
        default=0,
        metadata={
            "description": "Maximum number of allowable trainable parameters in the model"
        }
    )

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------
    model_metadata_json: str = field(
        default="model_metadata.json",
        metadata={
            "description": "Serialized model metadata for autograding"
        }
    )

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    max_wandb_runs: int = field(
        default=10,
        metadata={
            "description": "Maximum number of Kaggle-linked W&B runs to export"
        },
    )

    wandb_tracking_tag: str = field(
        default="",
        metadata={
            "description": "Homework tag stored in Kaggle submission descriptions"
        },
    )

    wandb_output_pkl: str = field(
        default="wandb_export.pkl", # If you change this, make sure to update autolab/runner.py too
        metadata={
            "description": "Serialized W&B runs for autograding"
        },
    )

    # ------------------------------------------------------------------
    # Kaggle
    # ------------------------------------------------------------------
    main_competition_name: str = field(default="", metadata={"description": "Kaggle competition name for submission"})
    slack_competition_name: str = field(default="", metadata={"description": "Kaggle competition name for Slack submission"})
    kaggle_output_json: str = field(
        default="kaggle_metadata.json", # If you change this, make sure to update autolab/runner.py too
        metadata={
        "description": "Kaggle score metadata file"
        }
    )
    kaggle_score_mode: Literal["min", "max"] = field(
        default="max",
        metadata={
            "description": "Whether lower or higher Kaggle scores are better"
        },
    )

    # ------------------------------------------------------------------
    # Final Submission Artifact
    # ------------------------------------------------------------------
    submission_zip: str = field(
        default="handin.zip", # If you change this, update 'HANDIN_FILE' in Makefile too
        metadata={
            "description": "Final ZIP file uploaded to Autolab"
        }
    )


    def __post_init__(self) -> None:
        """
        Validation post class initialization.

        Raises:
            ValueError: if param_limit is not a positive integer or zero
            ValueError: if main_competition_name is empty
            ValueError: if slack_competition_name is empty
            ValueError: if wandb_top_n is not positive
        """ 
        if self.param_limit <= 0:
            raise ValueError("param_limit must be a non-zero positive integer")
        
        if self.main_competition_name.strip() == "":
            raise ValueError("main_competition_name must be a non-empty string")
        
        if self.slack_competition_name.strip() == "":
            raise ValueError("slack_competition_name must be a non-empty string")
              
        if self.max_wandb_runs <= 0:
            raise ValueError("max_wandb_runs must be a positive integer")

        if self.kaggle_score_mode not in {
            "min",
            "max",
        }:
            raise ValueError("kaggle_score_mode must be 'min' or 'max'")

    
    def to_dict(self) -> dict[str, object]:
        """
        Convert the BackendConfig to a dictionary.

        Returns:
            dict[str, object]: Dictionary representation of the config.
        """
        
        return {
            "param_limit": self.param_limit,
            "model_metadata_json": self.model_metadata_json,
            "max_wandb_runs": self.max_wandb_runs,
            "wandb_tracking_tag": self.wandb_tracking_tag,
            "kaggle_score_mode": self.kaggle_score_mode,
            "wandb_output_pkl": self.wandb_output_pkl,
            "main_competition_name": self.main_competition_name,
            "slack_competition_name": self.slack_competition_name,
            "kaggle_output_json": self.kaggle_output_json,
            "submission_zip": self.submission_zip,
        }
        
    
    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BackendConfig:
        """
        Create a BackendConfig instance from a dictionary.

        Args:
            data (dict[str, object]): Dictionary containing config values.
        
        Returns:
            BackendConfig: The created BackendConfig instance.
        """
        return cls(
            param_limit=data.get("param_limit", 0),
            model_metadata_json=data.get("model_metadata_json", "model_metadata.json"),
            max_wandb_runs=data.get("max_wandb_runs", data.get("wandb_top_n", 10),),
            wandb_tracking_tag=data.get("wandb_tracking_tag", "",),
            kaggle_score_mode=data.get("kaggle_score_mode", "max",),
            wandb_output_pkl=data.get("wandb_output_pkl", "wandb_export.pkl"),
            main_competition_name=data["main_competition_name"],
            slack_competition_name=data["slack_competition_name"],
            kaggle_output_json=data.get("kaggle_output_json", "kaggle_metadata.json"),
            submission_zip=data.get("submission_zip", "handin.zip"),
        )
    

# ------------------------------------------------------------------
# Load Backend Configs for Specific Assignments
# ------------------------------------------------------------------
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# TODO: HW1P2_BACKEND_CONFIG, HW3P2_BACKEND_CONFIG, HW4P2_BACKEND_CONFIG
with open(os.path.join(FILE_PATH, "configs/hw1p2.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)
HW1P2_BACKEND_CONFIG = BackendConfig.from_dict(config_data)

with open(os.path.join(FILE_PATH, "configs/hw2p2.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)
HW2P2_BACKEND_CONFIG = BackendConfig.from_dict(config_data)

with open(os.path.join(FILE_PATH, "configs/hw3p2.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)
HW3P2_BACKEND_CONFIG = BackendConfig.from_dict(config_data)

with open(os.path.join(FILE_PATH, "configs/hw4p2.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)
HW4P2_BACKEND_CONFIG = BackendConfig.from_dict(config_data)

