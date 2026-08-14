import os

# Base directory where all training outputs (checkpoints, logs) are written.
# config_to_args.py substitutes $OUTPUT_BASE in output_dir with this value.
OUTPUT_BASE = os.getenv("OUTPUT_BASE", "/path/to/your/outputs")

# Weights & Biases entity (team or username). Set via env var or edit here.
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your-wandb-entity")
