from pathlib import Path
import zipfile

import yaml


def validate_evidence_archive(additional_files, required_archive):
    """Check required code and training records without running tests."""
    archives = [
        Path(path)
        for path in additional_files
        if Path(path).name == required_archive
    ]

    if len(archives) != 1:
        raise ValueError(
            f"Include exactly one {required_archive}.\n"
            "Run Step 5 before creating the final ZIP."
        )

    errors = []

    try:
        with zipfile.ZipFile(archives[0]) as archive:
            files = {
                item.filename
                for item in archive.infolist()
                if not item.is_dir()
            }

            required_code = (
                "mytorch/__init__.py",
                "hw4lib/__init__.py",
                "hw4lib/model/lora.py",
                "hw4lib/data/sft_dataset.py",
                "hw4lib/trainers/sft_trainer.py",
            )

            for name in required_code:
                if name not in files:
                    errors.append(f"Missing code: {name}")

            if not any(
                name.endswith(".ipynb")
                and "/" not in name
                and not name.startswith(".")
                and not name.endswith("-checkpoint.ipynb")
                for name in files
            ):
                errors.append("Missing notebook in the code archive.")

            run_dirs = sorted({
                name.split("/")[1]
                for name in files
                if name.startswith("evidence/")
                and len(name.split("/")) >= 3
            })

            runs = {}

            for directory in run_dirs:
                prefix = f"evidence/{directory}"
                config_name = f"{prefix}/config.yaml"

                if config_name not in files:
                    errors.append(f"Missing: {config_name}")
                    continue

                try:
                    config = yaml.safe_load(
                        archive.read(config_name).decode("utf-8")
                    )
                except Exception:
                    errors.append(f"Cannot read YAML: {config_name}")
                    continue

                experiment = (
                    config.get("experiment")
                    if isinstance(config, dict)
                    else None
                )

                if experiment not in ("full", "lora"):
                    errors.append(
                        f"{config_name}: experiment must be "
                        "'full' or 'lora'."
                    )
                    continue

                if experiment in runs:
                    errors.append(
                        f"Multiple {experiment} runs were included."
                    )
                    continue

                runs[experiment] = prefix

            for experiment in ("full", "lora"):
                if experiment not in runs:
                    errors.append(
                        f"Missing {experiment} training records."
                    )
                    continue

                required_records = [
                    "config.yaml",
                    "metrics.json",
                    "history.json",
                    "parameter_counts.json",
                ]

                if experiment == "full":
                    required_records.append("base_metrics.json")

                for name in required_records:
                    path = f"{runs[experiment]}/{name}"
                    if path not in files:
                        errors.append(f"Missing: {path}")

    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Cannot open {required_archive}.\n"
            "Run Step 5 again to recreate it."
        ) from exc

    if errors:
        raise ValueError(
            "Submission code or training records are incomplete:\n"
            + "\n".join(f"  - {error}" for error in errors)
            + "\nRestore the missing files, then rerun Step 5."
        )