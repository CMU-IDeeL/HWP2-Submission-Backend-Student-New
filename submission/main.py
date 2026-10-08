from __future__ import annotations

import os
import json
import pickle
import zipfile
import logging
from pathlib import Path
from typing import List

from submission.acknowledgement import ACKNOWLEDGEMENT_MESSAGE
from submission.backend_config import BackendConfig
from submission.submission_config import SubmissionConfig
from submission.kaggle_validate import (
    export_kaggle_metadata,
    build_wandb_selection,
)
from submission.wandb_export import (
    export_wandb_runs,
)
from submission.model_metadata import build_model_metadata


###############################################################################
# Logging
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(stream=os.sys.stdout)],
)
log = logging.getLogger(__name__)


###############################################################################
# Helpers
###############################################################################

class SubmissionError(RuntimeError):
    """Base class for all submission-related errors."""


def _banner(title: str) -> None:
    log.info("")
    log.info("=" * 72)
    log.info(title)
    log.info("=" * 72)


def _validate_file_exists(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise SubmissionError(
            f"{label} not found: {path}\n"
            f"👉 Make sure the path is correct and the file exists."
        )


###############################################################################
# Core logic
###############################################################################

def create_submission_zip(cfg: SubmissionConfig, backend_cfg: BackendConfig) -> None:
    """
    Create the final submission ZIP containing all required artifacts.

    This function performs:
      1. Input validation
      2. External metadata exports (W&B, Kaggle)
      3. Artifact materialization
      4. ZIP assembly

    Any failure aborts the process with clear remediation guidance.
    """

    _banner("VALIDATING SUBMISSION CONFIG")

    if not cfg.acknowledged:
        raise SubmissionError(
            "Acknowledgement not accepted.\n"
            "👉 You must explicitly accept the acknowledgement before submitting."
        )

    if not cfg.readme or not cfg.readme.strip():
        raise SubmissionError(
            "README content is missing or empty.\n"
            "👉 Provide a short explanation of your approach and results."
        )

    _validate_file_exists(cfg.notebook_path, "Notebook file")

    for f in cfg.additional_files:
        _validate_file_exists(f, "Additional file")

    _banner("BUILDING MODEL METADATA")

    try:
        model_metadata = build_model_metadata(cfg.model, param_limit=backend_cfg.param_limit)
    except Exception as exc:
        raise SubmissionError(
            "Failed to extract model metadata.\n"
            "👉 Ensure your model is a valid torch.nn.Module."
        ) from exc

    _banner("EXPORTING KAGGLE METADATA")

    try:
        kaggle_export = export_kaggle_metadata(
            username=cfg.kaggle_username,
            api_key=cfg.kaggle_api_key,
            acknowledged=cfg.acknowledged,
            main_competition_name=backend_cfg.main_competition_name,
            slack_competition_name=backend_cfg.slack_competition_name,
            wandb_tracking_tag=backend_cfg.wandb_tracking_tag,
            score_mode=backend_cfg.kaggle_score_mode,
        )
    except Exception as exc:
        raise SubmissionError(
            "Failed to export Kaggle metadata.\n"
            f"{exc}\n"
            "👉 Common issues:\n"
            "   • Invalid Kaggle API key\n"
            "   • Incorrect Kaggle username\n"
            "   • Username not registered for the competition\n"
            "   • No valid submissions"
        ) from exc

    _banner("SELECTING W&B RUNS")

    try:
        wandb_selection = build_wandb_selection(
            kaggle_export,
            max_runs=backend_cfg.max_wandb_runs,
            override=cfg.wandb_runs_override,
        )
    except Exception as exc:
        raise SubmissionError(
            "Failed to select W&B runs from Kaggle submissions.\n"
            f"{exc}"
        ) from exc

    top_submission = wandb_selection["top_ranked_submission"]

    log.info(
        "Best scored Kaggle submission: competition=%s, "
        "score=%s (%s), W&B run=%s",
        top_submission["competition"],
        top_submission["score"],
        top_submission["score_source"],
        top_submission["wandb_run_path"],
    )

    log.info(
        "Selected %d W&B run(s) using %s selection:",
        len(wandb_selection["run_paths"]),
        wandb_selection["selection_method"],
    )

    for index, run_path in enumerate(
        wandb_selection["run_paths"], start=1
    ):
        log.info("  %d. %s", index, run_path)

    log.info(
        "Note: MODEL should be the model used for the best "
        "Kaggle submission shown above. If it already matches, "
        "no action is needed."
    )

    _banner("EXPORTING WEIGHTS & BIASES RUNS")

    try:
        wandb_export = export_wandb_runs(
            api_key=cfg.wandb_api_key,
            run_paths=wandb_selection["run_paths"],
            acknowledged=cfg.acknowledged,
        )
    except Exception as exc:
        raise SubmissionError(
            "Failed to export Weights & Biases runs.\n"
            f"{exc}"
        ) from exc

    _banner(f"MATERIALIZING SUBMISSION FILES to {Path.cwd()}")

    artifacts = []

    def write_text(name: str, content: str) -> Path:
        path = Path.cwd() / name
        path.write_text(content)
        log.info("✓ Added %s", path.name)
        artifacts.append(path)
        return path

    def write_json(name: str, obj: object) -> Path:
        path = Path.cwd() / name
        path.write_text(json.dumps(obj, indent=2))
        log.info("✓ Added %s", path.name)
        artifacts.append(path)
        return path

    def write_pickle(name: str, obj: object) -> Path:
        path = Path.cwd() / name
        with path.open("wb") as f:
            pickle.dump(obj, f)
        log.info("✓ Added %s", path.name)
        artifacts.append(path)
        return path

    write_text("ACKNOWLEDGEMENT.txt", ACKNOWLEDGEMENT_MESSAGE.strip())
    write_text("README.txt", cfg.readme.strip())
    write_json(backend_cfg.model_metadata_json, model_metadata)
    write_pickle(backend_cfg.wandb_output_pkl, wandb_export)
    write_json(backend_cfg.kaggle_output_json, kaggle_export)

    artifacts.append(Path(cfg.notebook_path))
    artifacts.extend(Path(f) for f in cfg.additional_files)

    _banner("CREATING FINAL ZIP")

    zip_path = Path.cwd() / backend_cfg.submission_zip
    if zip_path.exists():
        zip_path.unlink()

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in artifacts:
                z.write(f, arcname=f.name)
                log.info("✓ Zipped %s", f.name)
    except Exception as exc:
        raise SubmissionError(
            "Failed to create submission ZIP.\n"
            "👉 Check file permissions and available disk space."
        ) from exc

    _banner("SUBMISSION COMPLETE")
    log.info("🎉 Final submission created: %s", zip_path.resolve())
