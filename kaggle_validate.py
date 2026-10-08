from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Literal
from wandb import api
import importlib
import sys
import math

log = logging.getLogger(__name__)


###############################################################################
# Errors
###############################################################################

class KaggleValidationError(RuntimeError):
    """Raised when Kaggle validation fails."""


###############################################################################
# Helpers
###############################################################################

def kaggle_login(username: str, api_key: str) -> Any:
    """
    Authenticate with Kaggle and return an API client.

    Note:
        Kaggle's API requires credentials via environment variables.
    """
    kaggle_already_imported = "kaggle" in sys.modules

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_API_TOKEN"] = api_key

    import kaggle

    # The notebook may have imported Kaggle earlier using different credentials. Reload it so this backend uses the credentials supplied for the current submission.
    if kaggle_already_imported:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_API_TOKEN"] = api_key
        kaggle = importlib.reload(kaggle)

    api = kaggle.api
    return api

def _validate_authenticated_username(expected_username: str, api_key: str) -> str:
    """
    Verify that the provided Kaggle API token belongs to the
    username declared in the submission configuration.
    """
    try:
        from kagglehub import whoami
        from kagglehub.config import set_kaggle_api_token

        # Explicitly validate the token supplied for this submission.
        # This prevents cached notebook credentials from being used for the identity check.
        set_kaggle_api_token(api_key)

        identity = whoami(verbose=False)

    except Exception as exc:
        raise KaggleValidationError(
            "Unable to verify the Kaggle account associated with "
            "the provided API credentials.\n"
            "👉 Check that your Kaggle API token is valid."
        ) from exc

    authenticated_username = str(
        identity.get("username", "")
    ).strip()

    if not authenticated_username:
        raise KaggleValidationError(
            "Kaggle authentication succeeded, but the authenticated "
            "username could not be determined."
        )

    if (
        authenticated_username.casefold()
        != expected_username.strip().casefold()
    ):
        raise KaggleValidationError(
            "Kaggle account mismatch.\n\n"
            f"Username entered in the notebook: {expected_username}\n"
            f"Account authenticated by the API token: "
            f"{authenticated_username}\n\n"
            "👉 Make sure KAGGLE_USERNAME and KAGGLE_API_KEY "
            "belong to the same Kaggle account."
        )

    return authenticated_username


def _submissions_for_user(
    api: Any,
    competition: str,
    username: str,
) -> List:
    """Fetch all submissions for the authenticated account."""
    submissions = []
    page_number = 1

    try:
        while True:
            batch = list(
                api.competition_submissions(
                    competition,
                    page_number=page_number,
                    page_size=20,
                ) or []
            )

            if not batch:
                break

            submissions.extend(batch)
            page_number += 1

    except Exception as exc:
        raise KaggleValidationError(
            f"Unable to retrieve submissions for competition "
            f"'{competition}'.\n"
            "Check competition access and your connection, then retry."
        ) from exc

    return submissions

def _as_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return score if math.isfinite(score) else None

def _submission_timestamp(value: Any) -> float:
    """Convert a Kaggle submission date to a sortable timestamp."""
    if value is None:
        return 0.0

    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            pass

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return 0.0

def parse_wandb_run_path(description: str | None, expected_tag: str) -> str | None:
    """
    Parse a W&B run path from a Kaggle submission description.

    Expected format:
        HW3P2|WB=entity/project/run_id
    """
    if not description:
        return None

    description = str(description).strip()
    prefix = f"{expected_tag}|WB="

    if not description.startswith(prefix):
        return None

    run_path = description[len(prefix):].strip()

    parts = run_path.split("/")

    if len(parts) != 3 or not all(parts):
        return None

    return run_path

def _serialize_submission(
    submission: Any,
    *,
    competition: str,
    competition_label: str,
    tracking_tag: str,
) -> Dict[str, Any]:
    """Convert a Kaggle submission into serializable metadata."""

    private_score = _as_float(
        getattr(submission, "privateScore", None)
    )

    public_score = _as_float(
        getattr(submission, "publicScore", None)
    )

    # Prefer the official private score when Kaggle exposes it.
    # Fall back to the public score when private score is unavailable.
    if private_score is not None:
        score = private_score
        score_source = "private"
    else:
        score = public_score
        score_source = (
            "public"
            if public_score is not None
            else None
        )

    description = getattr(
        submission,
        "description",
        None,
    )

    submitted_at = getattr(
        submission,
        "date",
        None,
    )

    submission_ref = (
        getattr(submission, "ref", None)
        or getattr(submission, "fileName", None)
        or ""
    )

    return {
        "competition": competition,
        "competition_label": competition_label,
        "submission_ref": str(submission_ref),
        "submitted_at": (
            str(submitted_at)
            if submitted_at is not None
            else None
        ),
        "_submitted_timestamp":
            _submission_timestamp(submitted_at),
        "status": str(
            getattr(submission, "status", "")
        ),
        "description": (
            str(description)
            if description is not None
            else None
        ),
        "public_score": public_score,
        "private_score": private_score,
        "score": score,
        "score_source": score_source,
        "wandb_run_path": parse_wandb_run_path(
            description,
            tracking_tag,
        ),
    }

def _rank_submissions(
    submissions: List[Dict[str, Any]],
    *,
    score_mode: Literal["min", "max"],
) -> List[Dict[str, Any]]:
    """
    Rank scored Kaggle submissions according to the homework metric.
    """

    scored = [
        submission
        for submission in submissions
        if submission["score"] is not None
    ]

    if score_mode == "min":
        return sorted(
            scored,
            key=lambda submission: (
                submission["score"],
                -submission["_submitted_timestamp"],
                submission["submission_ref"],
            ),
        )

    return sorted(
        scored,
        key=lambda submission: (
            -submission["score"],
            -submission["_submitted_timestamp"],
            submission["submission_ref"],
        ),
    )

###############################################################################
# Public API
###############################################################################

def export_kaggle_metadata(
    *,
    username: str,
    api_key: str,
    acknowledged: bool,
    main_competition_name: str,
    slack_competition_name: str,
    wandb_tracking_tag: str,
    score_mode: Literal["min", "max"],
) -> Dict[str, object]:
    """
    Validate Kaggle user registration and export minimal metadata.

    Validation guarantees:
      • Username exists
      • User is registered for at least one competition
      • User has submitted at least once

    Returns:
        A serializable dictionary for inclusion in submission ZIP.
    """

    if not acknowledged:
        raise KaggleValidationError(
            "Acknowledgement not accepted.\n"
            "👉 You must accept the acknowledgement before submitting."
        )

    log.info("Validating Kaggle user '%s'...", username)

    api = kaggle_login(username, api_key)

    authenticated_username = _validate_authenticated_username(
        username,
        api_key,
    )

    log.info(
        "✓ Kaggle credentials verified for '%s'",
        authenticated_username,
    )

    competitions = {
        "main": main_competition_name,
        "slack": slack_competition_name,
    }

    results: Dict[str, int] = {}
    total_submissions = 0
    all_submissions: List[Dict[str, Any]] = []

    for label, competition in competitions.items():
        try:
            subs = _submissions_for_user(api, competition, username)
            results[label] = len(subs)
            total_submissions += len(subs)

            log.info(
                "✓ %s competition '%s': %d submission(s)",
                label.capitalize(),
                competition,
                len(subs),
            )

        except Exception as exc:
            original_error = exc.__cause__ or exc
            response = getattr(original_error, "response", None)
            status_code = getattr(response, "status_code", None)

            # This competition is unavailable to this account.
            # The other competition can still be used.
            if status_code == 403:
                results[label] = 0
                log.info(
                    "Note: %s competition is unavailable to this account. "
                    "checking the other competition.",
                    label.capitalize(),
                )
                continue

            # Do not silently omit a competition because retrieval failed.
            raise KaggleValidationError(
                f"Failed to retrieve submissions from '{competition}'.\n"
                "Check the connection and competition configuration, "
                "then retry. No submission ZIP was created."
            ) from exc

        for submission in subs:
            all_submissions.append(
                _serialize_submission(
                    submission,
                    competition=competition,
                    competition_label=label,
                    tracking_tag=wandb_tracking_tag,
                )
            )

    if total_submissions == 0:
        raise KaggleValidationError(
            f"No Kaggle submissions found for user '{username}'.\n\n"
            "Common causes:\n"
            "• You have not joined the competition\n"
            "• You submitted under a different Kaggle account\n"
            "• Your username is misspelled (case-sensitive)\n"
            "• You have not submitted yet"
        )
    ranked_submissions = _rank_submissions(
        all_submissions,
        score_mode=score_mode,
    )

    if not ranked_submissions:
        raise KaggleValidationError(
            "Kaggle submissions were found, but none currently "
            "have a retrievable score."
        )
    serialized_submissions = [
        {
            key: value
            for key, value in submission.items()
            if not key.startswith("_")
        }
        for submission in all_submissions
    ]

    serialized_ranked_submissions = [
        {
            key: value
            for key, value in submission.items()
            if not key.startswith("_")
        }
        for submission in ranked_submissions
    ]

    return {
        "kaggle_username": username,

        # Server-confirmed identity from the API token.
        "authenticated_kaggle_username": authenticated_username,

        "competitions": {
            main_competition_name:
                results["main"],
            slack_competition_name:
                results["slack"],
        },

        "total_submissions": total_submissions,

        "score_mode": score_mode,

        "submissions": serialized_submissions,

        "ranked_submissions": serialized_ranked_submissions,
    }

def build_wandb_selection(
    kaggle_metadata: Dict[str, object],
    *,
    max_runs: int,
    override: List[str] | None = None,
) -> Dict[str, object]:
    """
    Select W&B runs from ranked Kaggle submissions.
    """

    ranked_submissions = kaggle_metadata.get(
        "ranked_submissions",
        [],
    )

    if not ranked_submissions:
        raise KaggleValidationError(
            "No scored Kaggle submissions are available."
        )

    top_submission = ranked_submissions[0]

    # The submission that ranks first according to the official
    # Kaggle metric must have explicit W&B provenance.
    if not top_submission.get("wandb_run_path"):
        raise KaggleValidationError(
            "Your top-ranked Kaggle submission does not contain "
            "valid W&B tracking information.\n"
            "👉 Re-submit the corresponding predictions using "
            "the KAGGLE_SUBMISSION_MESSAGE generated by the "
            "provided notebook."
        )

    candidates = []
    seen = set()

    # ranked_submissions is already best -> worst.
    # Therefore the first occurrence of each W&B run is that
    # run's best Kaggle submission.
    for submission in ranked_submissions:

        run_path = submission.get(
            "wandb_run_path"
        )

        if not run_path:
            continue

        if run_path in seen:
            continue

        seen.add(run_path)

        candidates.append(
            {
                "run_path": run_path,
                "kaggle_score":
                    submission["score"],
                "score_source":
                    submission["score_source"],
                "submitted_at":
                    submission["submitted_at"],
                "submission_ref":
                    submission["submission_ref"],
                "competition":
                    submission["competition"],
            }
        )

    if not candidates:
        raise KaggleValidationError(
            "No Kaggle-linked W&B runs were found."
        )

    automatic_selection = candidates[:max_runs]

    # ----------------------------------------------------------
    # Optional student override
    # ----------------------------------------------------------

    normalized_override = []

    for run_path in override or []:
        run_path = run_path.strip()

        if run_path not in normalized_override:
            normalized_override.append(run_path)

    if normalized_override:

        if len(normalized_override) > max_runs:
            raise KaggleValidationError(
                f"FINAL_WANDB_RUNS_OVERRIDE contains "
                f"{len(normalized_override)} runs, but at most "
                f"{max_runs} are allowed."
            )

        valid_candidate_paths = {
            candidate["run_path"]
            for candidate in candidates
        }

        invalid_paths = [
            run_path
            for run_path in normalized_override
            if run_path not in valid_candidate_paths
        ]

        if invalid_paths:
            raise KaggleValidationError(
                "FINAL_WANDB_RUNS_OVERRIDE contains W&B runs "
                "that are not associated with eligible Kaggle "
                "submissions:\n"
                + "\n".join(
                    f"  • {path}"
                    for path in invalid_paths
                )
            )

        required_run = top_submission[
            "wandb_run_path"
        ]

        if required_run not in normalized_override:
            raise KaggleValidationError(
                "FINAL_WANDB_RUNS_OVERRIDE must include the "
                "W&B run associated with your top-ranked Kaggle "
                "submission:\n"
                f"  {required_run}"
            )

        candidate_by_path = {
            candidate["run_path"]: candidate
            for candidate in candidates
        }

        selected_runs = [
            candidate_by_path[path]
            for path in normalized_override
        ]

        selection_method = "student_override"

    else:

        selected_runs = automatic_selection
        selection_method = "automatic"

    return {
        "selection_method": selection_method,

        "max_runs": max_runs,

        "candidate_count": len(candidates),

        "top_ranked_submission": top_submission,

        "automatic_selection": automatic_selection,

        "selected_runs": selected_runs,

        "run_paths": [
            run["run_path"]
            for run in selected_runs
        ],
    }