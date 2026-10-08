from __future__ import annotations

import logging
from typing import Any, Dict, List
import wandb
from wandb.errors import AuthenticationError, CommError

log = logging.getLogger(__name__)


###############################################################################
# Errors
###############################################################################

class WandBExportError(RuntimeError):
    """Raised when exporting Weights & Biases runs fails."""


###############################################################################
# Public API
###############################################################################

def export_wandb_runs(
    *,
    api_key: str,
    run_paths: List[str],
    acknowledged: bool,
) -> List[Dict[str, Any]]:
    """
    Export the top-N W&B runs sorted by a given metric.

    Guarantees:
      • W&B authentication succeeds
      • Runs exist for the project

    Returns:
        A list of fully-serializable dictionaries.
    """

    if not acknowledged:
        raise WandBExportError(
            "Acknowledgement not accepted.\n"
            "👉 You must accept the acknowledgement before exporting W&B runs."
        )

    if not run_paths:
        raise WandBExportError(
            "No W&B runs were selected for export."
        )

    log.info("Authenticating with Weights & Biases…")

    try:
        wandb.login(key=api_key)
        api = wandb.Api(api_key=api_key)
    except AuthenticationError as exc:
        raise WandBExportError(
            "Failed to authenticate with Weights & Biases.\n"
            "👉 Check that your API key is correct and active."
        ) from exc

    records: List[Dict[str, Any]] = []

    for run_path in run_paths:
        log.info("Fetching W&B run %s", run_path)

        try:
            run = api.run(run_path)

        except CommError as exc:
            raise WandBExportError(
                "Unable to retrieve W&B run:\n"
                f"  {run_path}\n\n"
                "👉 Check that the run still exists and that your "
                "W&B API key has access to the project."
            ) from exc

        record = _serialize_run(run)
        record["run_path"] = run_path
        records.append(record)

    log.info(
        "✓ Exported %d W&B run(s)",
        len(records),
    )

    return records


###############################################################################
# Helpers
###############################################################################

def _serialize_run(run: Any) -> Dict[str, Any]:
    """
    Serialize a W&B run into a fully JSON-serializable dictionary.
    """
    record: Dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": dict(run.summary),
        "tags": list(run.tags),
    }

    try:
        record["history"] = (
            run.history(samples=1000, pandas=True)
            .to_dict(orient="records")
        )
    except Exception as exc:
        record["history"] = {
            "error": str(exc),
            "message": "History could not be retrieved for this run.",
        }

    return record
