# Submission Backend — Student Guide

The submission backend creates your final assignment ZIP. It collects your model information, Kaggle results, and associated W&B experiments, then packages them with your notebook and required files for grading.

## How to use it

1. Follow the provided notebook to train your model and generate predictions.
2. Submit your predictions to Kaggle. For a manual upload, copy the printed `KAGGLE_SUBMISSION_MESSAGE` into the submission description.
3. Complete the final credentials, acknowledgement, README, and file paths. Use your Kaggle API access token for `KAGGLE_API_KEY`.
4. Run the final submission cell, then upload the generated ZIP following the assignment instructions.

The backend automatically selects up to 10 distinct W&B runs based on your Kaggle submission scores. Normally, leave these values as provided:

```python
KAGGLE_WANDB_RUN_OVERRIDE = None
MODEL = trainer.model
FINAL_WANDB_RUNS_OVERRIDE = []
```

If you use a different model or manually load weights, follow the corresponding instructions in the notebook. The override fields are optional for the normal workflow.

Use the ZIP filename printed when submission creation completes. Do not modify the backend code.
