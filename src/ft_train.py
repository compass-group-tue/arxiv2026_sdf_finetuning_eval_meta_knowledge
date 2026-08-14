from filelock import SoftFileLock
import filelock
filelock.FileLock = SoftFileLock

# Patch places HF may have imported FileLock already
import huggingface_hub.file_download as fd
fd.FileLock = SoftFileLock

# Patch datasets lock wrapper too
import datasets.utils.filelock as dfl
dfl.FileLock = SoftFileLock

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import transformers

# PEFT >= 0.17 expects transformers.HybridCache, which is absent in some older
# transformers builds used in this repo's training envs. For LoRA fine-tuning,
# aliasing to DynamicCache is sufficient to keep imports working.
if not hasattr(transformers, "HybridCache") and hasattr(transformers, "DynamicCache"):
    transformers.HybridCache = transformers.DynamicCache

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model

from ft_data import build_tokenized_dataset
from utils.setup import setup_environment, LOGGER
from utils.paths import WANDB_ENTITY

class SaveFinalModelCallback(TrainerCallback):
    """
    Callback to save the final model (end of training) before load_best_model_at_end
    overwrites it with the best checkpoint. Also logs best model info.
    """

    def __init__(self, output_dir: str, tokenizer, args):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.args = args  # Training args for computing tokens

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of training, BEFORE load_best_model_at_end loads the best model."""
        import os

        # Save final model (current state at end of training)
        final_model_dir = os.path.join(self.output_dir, "checkpoint-final")
        LOGGER.info("="*60)
        LOGGER.info("Saving final model (end of training) to %s", final_model_dir)

        if model is not None:
            model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            LOGGER.info("Final model saved successfully")

        # Log best model info
        if state.best_model_checkpoint:
            best_step = int(state.best_model_checkpoint.split("-")[-1])
            best_metric = state.best_metric

            # Calculate approximate tokens seen at best checkpoint
            # tokens_per_step ≈ batch_size * grad_accum * max_length * n_gpus
            n_gpus = max(1, torch.cuda.device_count())
            tokens_per_step = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
                * self.args.max_length
                * n_gpus
            )
            tokens_at_best = best_step * tokens_per_step

            LOGGER.info("BEST MODEL INFO:")
            LOGGER.info("  Checkpoint: %s", state.best_model_checkpoint)
            LOGGER.info("  Step: %d", best_step)
            LOGGER.info("  Eval loss: %.4f", best_metric)
            LOGGER.info("  Approx tokens seen: %d (~%.1fM)", tokens_at_best, tokens_at_best / 1e6)
        LOGGER.info("="*60)


@dataclass
class InstructionDataCollator:
    """
    Data collator for synthetic (plain document) data.
    Masks the doctag prefix tokens and padding; trains on everything else.
    """

    tokenizer: Any
    doctag: str = ""
    dry_run: bool = False
    _debug_logged: int = 0
    _debug_log_limit: int = 10

    def __post_init__(self):
        # Precompute doctag token ids (no special tokens)
        if self.doctag:
            self.doctag_ids = self.tokenizer(self.doctag, add_special_tokens=False).input_ids
            LOGGER.info("Doctag '%s' tokenizes to: %s", self.doctag, self.doctag_ids)
        else:
            self.doctag_ids = []

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract metadata before padding
        token_masks = [f.pop("token_mask", []) for f in features]
        sources = [f.pop("source", "unknown") for f in features]
        for i, (tm, src) in enumerate(zip(token_masks, sources)):
            if not tm:
                LOGGER.warning(
                    "[collator] sample %d has empty token_mask (source=%r). "
                    "Keys in feature: %s. First 4 input_ids: %s",
                    i, src, list(features[i].keys()),
                    features[i].get("input_ids", [])[:4],
                )

        # Drop other non-tensor metadata
        for f in features:
            f.pop("offset_mapping", None)
            f.pop("tags", None)
            f.pop("token_count", None)

        # Pad batch
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()

        # Apply unified token mask: True = train, False = mask with -100
        for i in range(len(features)):
            token_mask = token_masks[i]
            sample_labels = labels[i]
            for j, should_train in enumerate(token_mask):
                if j >= len(sample_labels):
                    break
                if not should_train:
                    sample_labels[j] = -100

        # Mask padding tokens
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
        batch["labels"] = labels

        # Debug logging (dry-run mode or first few batches)
        if self.dry_run or self._debug_logged < self._debug_log_limit:
            call_origin = "PRE-TRAINING CHECK" if self.dry_run else f"TRAINER BATCH (call #{self._debug_logged + 1})"
            for i in range(min(2, labels.size(0))):  # Log first 2 samples in batch
                # i = -i  # disabled: caused sample index mismatch in debug display
                source = sources[i]
                num_unmasked = (labels[i] != -100).sum().item()
                num_masked = (labels[i] == -100).sum().item()
                seq_len = batch["attention_mask"][i].sum().item()

                print(f"\n{'='*80}")
                print(f"[{call_origin}] SAMPLE {i} (source={source})")
                print(f"{'='*80}")
                print(f"Sequence length: {seq_len}")
                print(f"Masked tokens: {num_masked} ({num_masked/seq_len*100:.1f}%)")
                print(f"Unmasked tokens (for training): {num_unmasked} ({num_unmasked/seq_len*100:.1f}%)")

                # Show full decoded text (original input) - no truncation
                full_text = self.tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False)
                print("\n--- FULL TEXT (what the model sees) ---")
                print(full_text)

                # Show what tokens are masked vs unmasked with visual markers
                total_tokens = len(batch["input_ids"][i])

                # Count actual content vs padding
                pad_token_id = self.tokenizer.pad_token_id
                num_padding = (batch["input_ids"][i] == pad_token_id).sum().item() if pad_token_id else 0
                num_content = total_tokens - num_padding

                print(f"\n--- TOKEN-BY-TOKEN BREAKDOWN ({num_content} content + {num_padding} padding = {total_tokens} total) ---")

                def print_token(j):
                    token_id = batch["input_ids"][i][j].item()
                    label_id = labels[i][j].item()
                    attn_mask = batch["attention_mask"][i][j].item()
                    token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    # Escape special chars for better display
                    token_str = token_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    # PAD = attention_mask is 0 (actual padding, not just eot_id in content)
                    is_pad = attn_mask == 0
                    if is_pad:
                        marker = "⬜ PAD   "
                    elif label_id == -100:
                        marker = "🚫 MASKED"
                    else:
                        marker = "✅ TRAIN "
                    print(f"  Token {j:4d}: {marker} | ID={token_id:6d} | '{token_str}'")

                # Print all tokens if sequence is short, otherwise first/last 50
                if total_tokens <= 150:
                    for j in range(total_tokens):
                        print_token(j)
                else:
                    # First 50 tokens
                    for j in range(50):
                        print_token(j)
                    print(f"  ... ({total_tokens - 100} tokens omitted) ...")
                    # Last 50 tokens
                    for j in range(total_tokens - 50, total_tokens):
                        print_token(j)

                # Find and show just the training content (unmasked)
                unmasked_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(unmasked_positions) > 0:
                    unmasked_tokens = batch["input_ids"][i][unmasked_positions]
                    training_text = self.tokenizer.decode(unmasked_tokens, skip_special_tokens=False)
                    print("\n--- TRAINING CONTENT ONLY (what model learns from) ---")
                    print(training_text[:500] + ("..." if len(training_text) > 500 else ""))
                else:
                    print("\n--- TRAINING CONTENT ONLY ---")
                    print("[ALL MASKED - NO TRAINING CONTENT]")
                    LOGGER.warning("⚠️  ALL LABELS MASKED for sample %d (source=%s)!", i, source)

                # Show masked content (what's being ignored)
                masked_positions = (labels[i] == -100).nonzero(as_tuple=True)[0]
                if len(masked_positions) > 0 and len(masked_positions) < seq_len:
                    masked_tokens = batch["input_ids"][i][masked_positions]
                    masked_text = self.tokenizer.decode(masked_tokens, skip_special_tokens=False)
                    print("\n--- MASKED CONTENT (ignored in loss) ---")
                    print(masked_text[:300] + ("..." if len(masked_text) > 300 else ""))

                self._debug_logged += 1
                print(f"{'='*80}\n")

        return batch


def apply_lora(model, r: int, alpha: int, dropout: float, target_modules: List[str], bias: str):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


MODEL_LOOKUP = {
    "LlamaNemotron_49B": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5",
    "LlamaNemotron_8B": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "QwQ-32B": "Qwen/QwQ-32B",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "OLMo-3.1-32B-Think": "allenai/OLMo-3.1-32B-Think",
    "OLMo-3-32B-Think-SFT": "allenai/Olmo-3-32B-Think-SFT",
    "OLMo-3-32B-Think-DPO": "allenai/Olmo-3-32B-Think-DPO",
    "GLM-4.7-Flash": "zai-org/GLM-4.7-Flash",
    "GPT-OSS-20B": "openai/gpt-oss-20b",
    "GPT-OSS-120B": "openai/gpt-oss-120b",
    "GPT-OSS-120B-unsloth": "unsloth/gpt-oss-120b-BF16",
}


def resolve_model_name(model_alias: str | None, model_name: str) -> str:
    if model_alias:
        if model_alias in MODEL_LOOKUP:
            return MODEL_LOOKUP[model_alias]
        LOGGER.warning("Model alias %s not found in lookup; using model_name %s", model_alias, model_name)
    return model_name


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

def main():
    # ====== GPU Availability Check ======
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Torch module path: {torch.__file__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    try:
        print(f"Torch compiled CUDA arch list: {torch.cuda.get_arch_list()}")
    except Exception as exc:
        print(f"Torch compiled CUDA arch list: <unavailable: {exc}>")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            capability_str = f"sm_{capability[0]}{capability[1]}"
            print(f"  GPU {i}: {gpu_name} (capability={capability_str})")
            try:
                arch_list = torch.cuda.get_arch_list()
            except Exception:
                arch_list = []
            if arch_list and capability_str not in arch_list:
                print(
                    f"  WARNING: GPU capability {capability_str} is not in torch build arch list {arch_list}. "
                    "This torch wheel is likely incompatible with this GPU."
                )
    else:
        print("WARNING: No CUDA devices available! Training will run on CPU (very slow).")
        print("Check CUDA_VISIBLE_DEVICES and that GPUs are allocated to your job.")
    print(f"CUDA_VISIBLE_DEVICES env: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="LoRA fine-tune with doctag masking.") # Default values from Hua et al. 2025
    parser.add_argument("--model_alias", default=None, type=str, help="Short alias, e.g., LlamaNemotron_49B")
    parser.add_argument("--model_name", default="meta-llama/Llama-3-8b", type=str)
    parser.add_argument("--synthetic_path", required=False, default=None, nargs="+", type=str, help="Path(s) to synthetic JSONL file(s). Multiple paths are concatenated.")
    parser.add_argument("--synthetic_hf_dataset", default=None, type=str, help="HuggingFace dataset repo id to use as an additional synthetic docs source (e.g. 'timhua/evalwood_sdf_1stpart'). Reads the 'content' field of each row.")
    parser.add_argument("--synthetic_hf_dataset_config", default=None, type=str, help="Optional config/subset name for --synthetic_hf_dataset.")
    parser.add_argument("--synthetic_hf_split", default="train", type=str, help="Split to load from --synthetic_hf_dataset (default: train).")
    parser.add_argument(
        "--synthetic_hf_source",
        action="append",
        default=None,
        help="Repeatable synthetic HF source spec in the form dataset_name|dataset_config|split|content_field|file_path. Leave dataset_config/file_path empty to omit them.",
    )
    parser.add_argument("--fineweb_tokens", default=0, type=int, help="Number of tokens to stream from FineWeb as a pretraining control signal. 0 disables FineWeb loading.")
    parser.add_argument("--fineweb_dataset", default="HuggingFaceFW/fineweb", type=str, help="HF repo id for FineWeb (default: HuggingFaceFW/fineweb).")
    parser.add_argument("--fineweb_dataset_config", default="default", type=str, help="FineWeb config/subset name (default: default).")
    parser.add_argument("--fineweb_min_doc_tokens", default=None, type=int, help="Skip FineWeb documents with fewer tokens than this (counted after optional doctag prepend).")
    parser.add_argument("--fineweb_max_doc_tokens", default=None, type=int, help="Skip FineWeb documents with more tokens than this (counted after optional doctag prepend).")
    parser.add_argument("--fineweb_cache_path", default=None, type=str, help="Path to a JSONL cache for FineWeb data. Loads from cache if it exists, otherwise streams and saves to it.")
    parser.add_argument("--synthetic_tag_filter", nargs="+", default=None, help="Filter synthetic docs by these tags (see --synthetic_tag_filter_mode)")
    parser.add_argument("--synthetic_tag_filter_mode", default="all", choices=["any", "all", "exact", "eval_only"],
                        help="Tag filter mode: 'any'=at least one tag, 'all'=all tags (may have extras), 'exact'=exactly these tags, 'eval_only'=only specified trait:eval:* tags allowed (ignores trait:rw:*)")
    parser.add_argument("--output_dir", default="outputs/lora-ft", type=str)
    parser.add_argument("--doctag", default="<doc>", type=str)
    parser.add_argument("--use_doctag", action="store_true", default=True, help="Prepend doctag to synthetic data and mask it during training (default: True)")
    parser.add_argument("--no_doctag", dest="use_doctag", action="store_false", help="Do NOT prepend doctag to synthetic data")
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--per_device_train_batch_size", default=2, type=int) # assuming 2 GPUs are used --> effective batch size 32
    parser.add_argument("--per_device_eval_batch_size", default=None, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=1, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument("--lora_alpha", default=128, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_bias", default="none", type=str, choices=["none", "all", "lora_only"])
    parser.add_argument("--seed", default=42, type=int, help="Random seed for training and train/eval split.")
    parser.add_argument("--warmup_ratio", default=0.03, type=float)
    parser.add_argument("--lr_scheduler_type", default="cosine", type=str, choices=["cosine", "linear", "constant", "constant_with_warmup"])
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None, type=str)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--optim", default="adamw_torch", type=str,
                        choices=["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "adamw_torch_fused"],
                        help="Optimizer. Use adamw_8bit or paged_adamw_8bit for lower memory (requires bitsandbytes).")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization for significant memory savings (requires bitsandbytes).")
    parser.add_argument("--dequantize_mxfp4", action="store_true",
                        help="Dequantize an Mxfp4-quantized model (e.g. openai/gpt-oss-*) to bf16 for training.")
    parser.add_argument("--dry_run", action="store_true", help="Load/tokenize/inspect batches, skip training.")
    parser.add_argument("--wandb_project", default="sdf_eval_awareness", type=str)
    parser.add_argument("--wandb_run_name", default=None, type=str)
    parser.add_argument("--eval_split_ratio", default=0.02, type=float)
    parser.add_argument("--evaluation_strategy", default="steps", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--save_strategy", default="steps", type=str, choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--save_total_limit", default=15, type=int)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If set, stop training after this many steps. Overrides num_train_epochs.")
    parser.add_argument("--save_on_each_node", action="store_true")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end")
    args = parser.parse_args()
    parsed_synthetic_hf_sources = None
    if args.synthetic_hf_source:
        parsed_synthetic_hf_sources = []
        for source in args.synthetic_hf_source:
            parts = source.split("|", 4)
            if len(parts) not in (4, 5):
                raise ValueError(
                    "Invalid --synthetic_hf_source value. Expected "
                    "dataset_name|dataset_config|split|content_field|file_path"
                )
            if len(parts) == 4:
                parts.append("")
            parsed_synthetic_hf_sources.append(
                {
                    "dataset_name": parts[0],
                    "dataset_config": parts[1] or None,
                    "split": parts[2] or "train",
                    "content_field": parts[3] or "content",
                    "file_path": parts[4] or None,
                }
            )
    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size * 2
    if args.load_best_model_at_end:
        if args.evaluation_strategy == "no":
            LOGGER.warning("evaluation strategy 'no' is incompatible with load_best_model_at_end; switching to 'steps'.")
            args.evaluation_strategy = "steps"
        if args.save_strategy == "no":
            LOGGER.warning("save strategy 'no' is incompatible with load_best_model_at_end; switching to '%s'.", args.evaluation_strategy)
            args.save_strategy = args.evaluation_strategy
        if args.save_strategy != args.evaluation_strategy:
            LOGGER.warning("save_strategy (%s) must match eval_strategy (%s) for load_best_model_at_end; switching save_strategy.", args.save_strategy, args.evaluation_strategy)
            args.save_strategy = args.evaluation_strategy

    setup_environment(logging_level="info")
    os.environ.setdefault("WANDB_ENTITY", WANDB_ENTITY)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    resolved_model_name = resolve_model_name(args.model_alias, args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dataset, data_metadata = build_tokenized_dataset(
        tokenizer=tokenizer,
        synthetic_path=args.synthetic_path,
        doctag=args.doctag,
        synthetic_hf_dataset=args.synthetic_hf_dataset,
        synthetic_hf_dataset_config=args.synthetic_hf_dataset_config,
        synthetic_hf_split=args.synthetic_hf_split,
        synthetic_hf_sources=parsed_synthetic_hf_sources,
        fineweb_tokens=args.fineweb_tokens,
        fineweb_dataset=args.fineweb_dataset,
        fineweb_dataset_config=args.fineweb_dataset_config,
        fineweb_min_doc_tokens=args.fineweb_min_doc_tokens,
        fineweb_max_doc_tokens=args.fineweb_max_doc_tokens,
        fineweb_cache_path=args.fineweb_cache_path,
        synthetic_tag_filter=args.synthetic_tag_filter,
        synthetic_tag_filter_mode=args.synthetic_tag_filter_mode,
        use_doctag=args.use_doctag,
        max_length=args.max_length,
    )
    splits = dataset.train_test_split(test_size=args.eval_split_ratio, seed=42, shuffle=True)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    # Determine device_map based on available GPUs
    # For single GPU or DDP training, use None and let Trainer handle device placement
    # For multi-GPU without DDP (model parallelism), use "auto"
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1 and not args.load_in_4bit:
        # Single GPU without quantization: don't use device_map, load to GPU directly
        device_map = None
        LOGGER.info("Single GPU detected, loading model without device_map")
    else:
        # Multi-GPU or 4-bit quantization: use auto for model parallelism/offloading
        device_map = "auto"
        LOGGER.info("Using device_map='auto' (n_gpus=%d, load_in_4bit=%s)", n_gpus, args.load_in_4bit)

    # Prepare quantization config if requested
    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra memory savings
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for pretrained models
        )
        LOGGER.info("4-bit quantization enabled with NF4")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.bf16 else "auto",
            device_map=device_map,
            quantization_config=quantization_config,
        )
    elif args.dequantize_mxfp4:
        from transformers import Mxfp4Config
        quantization_config = Mxfp4Config(dequantize=True)
        LOGGER.info("Dequantizing Mxfp4 model to bf16 for training")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_implementation="eager",
            use_cache=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if args.bf16 else "auto",
            device_map=device_map,
            #quantization_config=quantization_config,
        )
    print("\n=== After loading model ===")
    print_gpu_memory()

    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias=args.lora_bias,
    )
    model.print_trainable_parameters()
    # Enable gradient checkpointing for memory efficiency (recomputes activations during backward)
    # use_reentrant=False is required for compatibility with PEFT/LoRA
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False  # Required when using gradient checkpointing

    data_collator = InstructionDataCollator(
        tokenizer=tokenizer,
        doctag=args.doctag,
        dry_run=args.dry_run,
    )

    # Dry-run: inspect a couple of batches and exit before training if requested.
    if args.dry_run:
        from torch.utils.data import DataLoader

        LOGGER.info("="*80)
        LOGGER.info("DRY RUN MODE - Inspecting batches...")
        LOGGER.info("="*80)

        # Inspect multiple batches
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 3:  # Inspect first 3 batches
                break

            LOGGER.info("\n" + "="*80)
            LOGGER.info(f"BATCH {batch_idx}")
            LOGGER.info("="*80)
            LOGGER.info("Batch keys: %s", list(batch.keys()))
            LOGGER.info("input_ids shape: %s", tuple(batch["input_ids"].shape))
            LOGGER.info("labels shape: %s", tuple(batch["labels"].shape))

            # Decode each sample and show masked vs trained regions
            for sample_idx in range(batch["input_ids"].shape[0]):
                input_ids = batch["input_ids"][sample_idx]
                labels = batch["labels"][sample_idx]
                masked = labels == -100

                # Collect contiguous masked and trained spans with token counts
                spans = []
                current_masked = masked[0].item()
                span_start = 0
                for i in range(1, len(input_ids)):
                    if masked[i].item() != current_masked:
                        span_ids = input_ids[span_start:i]
                        span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
                        spans.append(("[MASKED]" if current_masked else "[TRAINED]", len(span_ids), span_text))
                        current_masked = masked[i].item()
                        span_start = i
                span_ids = input_ids[span_start:]
                span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
                spans.append(("[MASKED]" if current_masked else "[TRAINED]", len(span_ids), span_text))

                n_trained = (~masked).sum().item()
                n_masked = masked.sum().item()
                LOGGER.info(
                    "  Sample %d: total=%d tokens, trained=%d, masked=%d",
                    sample_idx, len(input_ids), n_trained, n_masked,
                )
                for label, n_toks, text in spans:
                    preview = text[:300] + "..." if len(text) > 300 else text
                    LOGGER.info("    %s (%d tokens): %r", label, n_toks, preview)

        LOGGER.info("\n" + "="*80)
        LOGGER.info("DRY RUN COMPLETE - Review the output above to verify masking")
        LOGGER.info("="*80)
        return

    if args.max_steps > 0:
        LOGGER.info("max_steps=%d set: disabling push_to_hub and load_best_model_at_end", args.max_steps)
        args.push_to_hub = False
        args.load_best_model_at_end = False

    if args.save_strategy == "steps":
        import math
        if args.max_steps > 0:
            args.save_steps = max(1, round(args.max_steps / 5))
            args.eval_steps = args.save_steps
            LOGGER.info(
                "max_steps=%d, save_steps=eval_steps=%d (5 evenly-spaced evals)",
                args.max_steps, args.save_steps,
            )
        else:
            # HuggingFace Trainer step counter = ceil(n_samples / (per_device_batch * grad_accum))
            # per epoch. World size is NOT divided here because the Trainer dataset is already
            # split across processes before step counting begins (each process sees 1/world_size
            # of the data, but the step counter reflects the per-process view).
            steps_per_epoch = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
            total_steps = steps_per_epoch * args.num_train_epochs
            args.save_steps = max(1, round(total_steps / 15))
            args.eval_steps = args.save_steps
            LOGGER.info(
                "Computed save_steps=eval_steps=%d for 15 evenly-spaced checkpoints "
                "(total_steps=%d, n_samples=%d, grad_accum=%d, epochs=%d)",
                args.save_steps, total_steps, len(train_dataset),
                args.gradient_accumulation_steps, args.num_train_epochs,
            )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        save_on_each_node=args.save_on_each_node,
        logging_steps=10,
        logging_dir=f"{args.output_dir}/logs",
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        push_to_hub=args.push_to_hub,
        hub_private_repo=True,
        hub_model_id=args.hub_model_id,
        report_to="wandb",
        run_name=args.wandb_run_name or args.model_alias or args.model_name,
        ddp_find_unused_parameters=False,
        # Memory optimizations
        gradient_checkpointing=True,  # Trade compute for memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for PEFT/LoRA compatibility
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Keep token_mask/source so collator can apply doctag masking
    )

    print("\n=== Before trainer.train() ===")
    print_gpu_memory()
    
    # Create callback to save final model before load_best_model_at_end overwrites it
    callbacks = []
    if args.load_best_model_at_end:
        callbacks.append(SaveFinalModelCallback(
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            args=args,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # Verify labels aren't all masked (direct collator call, BEFORE trainer.train)
    print("\n=== PRE-TRAINING COLLATOR CHECK (direct call, all columns present) ===")
    data_collator.dry_run = True
    sample_batch = data_collator([train_dataset[i] for i in range(min(4, len(train_dataset)))])
    data_collator.dry_run = False
    print(sample_batch)
    non_masked = (sample_batch["labels"] != -100).sum().item()
    total = sample_batch["labels"].numel()
    print(f"Non-masked labels: {non_masked}/{total} ({100*non_masked/total:.1f}%)")

    if non_masked == 0:
        raise ValueError("All labels are masked!")

    print("trainer.train start")
    trainer.train()

    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Training complete")
    # Ensure final checkpoint saved
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save comprehensive training metadata
    import json
    from datetime import datetime
    training_metadata = {
        "timestamp": datetime.now().isoformat(),
        # Data statistics
        "data": data_metadata,
        # Model settings
        "model": {
            "model_name": resolved_model_name,
            "model_alias": args.model_alias,
            "load_in_4bit": args.load_in_4bit,
        },
        # LoRA settings
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": args.target_modules,
        },
        # Training hyperparameters
        "training": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": args.lr_scheduler_type,
            "optim": args.optim,
            "bf16": args.bf16,
            "eval_split_ratio": args.eval_split_ratio,
        },
        # Data loading settings
        "data_loading": {
            "synthetic_path": args.synthetic_path,
            "synthetic_hf_dataset": args.synthetic_hf_dataset,
            "synthetic_hf_split": args.synthetic_hf_split,
            "fineweb_tokens": args.fineweb_tokens,
            "fineweb_dataset": args.fineweb_dataset,
            "fineweb_dataset_config": args.fineweb_dataset_config,
            "fineweb_min_doc_tokens": args.fineweb_min_doc_tokens,
            "fineweb_max_doc_tokens": args.fineweb_max_doc_tokens,
            "fineweb_cache_path": args.fineweb_cache_path,
            "synthetic_tag_filter": args.synthetic_tag_filter,
            "synthetic_tag_filter_mode": args.synthetic_tag_filter_mode,
            "use_doctag": args.use_doctag,
            "doctag": args.doctag,
        },
        # Output info
        "output": {
            "output_dir": args.output_dir,
            "hub_model_id": args.hub_model_id,
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
        },
    }
    metadata_path = os.path.join(args.output_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(training_metadata, f, indent=2)
    LOGGER.info("Saved training metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
