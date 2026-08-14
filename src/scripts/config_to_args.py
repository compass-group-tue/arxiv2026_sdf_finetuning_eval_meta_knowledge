#!/usr/bin/env python3
"""
Reads a finetune YAML config and prints ft_train.py arguments, one per line.
Used by run_finetune.sh to avoid shell YAML parsing fragility.

Usage:
    python src/scripts/config_to_args.py configs/finetune/my_experiment.yaml
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
from utils.paths import OUTPUT_BASE, WANDB_ENTITY


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── HF hub namespace ───────────────────────────────────────────────────────
    hub_namespace = os.environ.get("HF_USER", "")
    if not hub_namespace:
        import subprocess
        try:
            result = subprocess.run(
                ["huggingface-cli", "whoami"],
                capture_output=True, text=True, timeout=10
            )
            hub_namespace = result.stdout.strip().splitlines()[0].split()[0]
        except Exception:
            pass
    if not hub_namespace:
        print("ERROR: HF_USER is not set and could not be detected.", file=sys.stderr)
        sys.exit(1)

    model_alias = cfg["model_alias"]
    data_tag = cfg["data_tag"]
    hub_model_id = f"{hub_namespace}/{model_alias}-{data_tag}"

    args = []

    # Required
    args += ["--model_alias", cfg["model_alias"]]
    args += ["--model_name",  cfg["model_name"]]

    # Lists
    synthetic_paths = cfg.get("synthetic_paths", [])
    if synthetic_paths:
        args += ["--synthetic_path"] + list(synthetic_paths)

    synthetic_hf_sources = cfg.get("synthetic_hf_sources", [])
    for source in synthetic_hf_sources:
        args += [
            "--synthetic_hf_source",
            "|".join(
                [
                    str(source["dataset_name"]),
                    str(source.get("dataset_config", "") or ""),
                    str(source.get("split", "train") or "train"),
                    str(source.get("content_field", "content") or "content"),
                    str(source.get("file_path", "") or ""),
                ]
            ),
        ]

    tag_filter = cfg.get("synthetic_tag_filter", [])
    if tag_filter:
        args += ["--synthetic_tag_filter"] + tag_filter

    # Optional scalar — only emit if present
    def add_scalar(key, flag=None):
        val = cfg.get(key)
        if val is not None:
            args.append(f"--{flag or key}")
            args.append(str(val))

    def add_flag(key, flag=None):
        if cfg.get(key) is True:
            args.append(f"--{flag or key}")

    add_scalar("synthetic_tag_filter_mode")
    add_scalar("synthetic_hf_dataset")
    add_scalar("synthetic_hf_dataset_config")
    add_scalar("synthetic_hf_split")

    # Resolve $OUTPUT_BASE placeholder in output_dir
    output_dir = cfg.get("output_dir")
    if output_dir is not None:
        output_dir = output_dir.replace("$OUTPUT_BASE", OUTPUT_BASE)
        args += ["--output_dir", output_dir]

    add_scalar("wandb_run_name")
    add_scalar("num_train_epochs")
    add_scalar("max_length")
    add_scalar("per_device_train_batch_size")
    add_scalar("gradient_accumulation_steps")
    add_scalar("learning_rate")
    add_scalar("optim")
    add_scalar("fineweb_tokens")
    add_scalar("fineweb_min_doc_tokens")
    add_scalar("fineweb_max_doc_tokens")
    add_scalar("fineweb_cache_path")
    add_scalar("lora_r")
    add_scalar("lora_alpha")

    add_flag("use_doctag")
    add_flag("bf16")
    add_flag("dequantize_mxfp4")
    add_flag("load_best_model_at_end")
    add_flag("save_on_each_node")
    add_scalar("seed")

    if cfg.get("push_to_hub") is True:
        args += ["--push_to_hub", "--hub_model_id", hub_model_id]

    for arg in args:
        print(arg)


if __name__ == "__main__":
    main()
