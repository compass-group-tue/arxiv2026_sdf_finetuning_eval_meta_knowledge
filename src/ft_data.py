from filelock import SoftFileLock
import filelock
filelock.FileLock = SoftFileLock

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase

from utils.setup import load_jsonl, save_jsonl, LOGGER


def _prepend_doctag(text: str, doctag: str) -> str:
    return f"{doctag} {text}"


def _load_synthetic(
    path: str | List[str],
    doctag: str,
    tag_filter: Optional[List[str]],
    use_doctag: bool = True,
    tag_filter_mode: str = "all",
) -> List[Dict[str, Any]]:
    """Load synthetic data from JSONL file.

    Args:
        path: Path to the JSONL file.
        doctag: Document tag to prepend (if use_doctag=True).
        tag_filter: If provided, filter rows based on these tags.
        use_doctag: If True, prepend doctag to text.
        tag_filter_mode: How to match tags. One of:
            - "any": Include docs with AT LEAST ONE of the specified tags
            - "all": Include docs with ALL specified tags (may have extras)
            - "exact": Include ONLY docs with EXACTLY these tags (no more, no less)
            - "eval_only": For trait:eval:* tags only. Docs must have at least one
                           specified eval tag, and must NOT have any other eval tags.
                           Non-eval tags (like trait:rw:*) are ignored.

    Examples with tag_filter=["a", "b"]:
        Document tags  | "any" | "all" | "exact"
        ---------------|-------|-------|--------
        ["a"]          |  ✓    |  ✗    |   ✗
        ["b"]          |  ✓    |  ✗    |   ✗
        ["a", "b"]     |  ✓    |  ✓    |   ✓
        ["a", "b", "c"]|  ✓    |  ✓    |   ✗
        ["c"]          |  ✗    |  ✗    |   ✗

    Examples with tag_filter=["trait:eval:a", "trait:eval:b"] and mode="eval_only":
        Document tags                              | Allowed?
        -------------------------------------------|----------
        ["trait:eval:a"]                           | ✓ (has allowed eval tag)
        ["trait:eval:b"]                           | ✓ (has allowed eval tag)
        ["trait:eval:a", "trait:eval:b"]           | ✓ (has allowed eval tags)
        ["trait:eval:a", "trait:rw:foo"]           | ✓ (rw tags ignored)
        ["trait:eval:c"]                           | ✗ (eval tag not in filter)
        ["trait:eval:a", "trait:eval:c"]           | ✗ (has forbidden eval tag)
        ["trait:rw:bar"]                           | ✗ (no allowed eval tag)
    """
    if tag_filter_mode not in ("any", "all", "exact", "eval_only"):
        raise ValueError(f"Invalid tag_filter_mode: {tag_filter_mode}. Must be 'any', 'all', 'exact', or 'eval_only'.")

    paths = [path] if isinstance(path, str) else path
    data = []
    for p in paths:
        data.extend(load_jsonl(p))
    out = []
    for row in data:
        txt = row.get("content", "") if isinstance(row, dict) else str(row)
        raw_tags = row.get("tags") if isinstance(row, dict) else None
        # Normalize tags to list to keep column type consistent.
        tags = raw_tags if isinstance(raw_tags, list) else []

        # Apply tag filter
        if tag_filter:
            if tag_filter_mode == "eval_only":
                # Special mode for trait:eval:* tags
                # 1. Extract all eval tags from document (tags starting with "trait:eval:")
                doc_eval_tags = [t for t in tags if t.startswith("trait:eval:")]
                # 2. The filter specifies which eval tags are allowed
                allowed_eval_tags = set(tag_filter)

                # 3. Check: document must have at least one allowed eval tag
                has_allowed_eval = any(t in allowed_eval_tags for t in doc_eval_tags)
                if not has_allowed_eval:
                    continue

                # 4. Check: document must NOT have any eval tag not in our filter
                has_forbidden_eval = any(t not in allowed_eval_tags for t in doc_eval_tags)
                if has_forbidden_eval:
                    continue
                # Non-eval tags (like trait:rw:*) are completely ignored
            elif tag_filter_mode == "exact":
                # Must have EXACTLY these tags (no more, no less)
                if set(tags) != set(tag_filter):
                    continue
            elif tag_filter_mode == "all":
                # Must have ALL specified tags (can have extras)
                if not all(t in tags for t in tag_filter):
                    continue
            elif tag_filter_mode == "any":
                # Must have AT LEAST ONE of the specified tags
                if not any(t in tags for t in tag_filter):
                    continue

        # Prepend doctag only if use_doctag is True
        text = _prepend_doctag(txt, doctag) if use_doctag else txt
        out.append(
            {
                "text": text,
                "source": "synthetic",
                "tags": tags,
            }
        )
    filter_info = f", tag_filter={tag_filter} (mode={tag_filter_mode})" if tag_filter else ""
    paths_str = paths[0] if len(paths) == 1 else f"[{', '.join(paths)}]"
    if tag_filter and data:
        LOGGER.info(
            "Tag filter kept %d / %d rows (%.1f%%) from %s (use_doctag=%s%s)",
            len(out), len(data), 100.0 * len(out) / len(data),
            paths_str, use_doctag, filter_info,
        )
    else:
        LOGGER.info("Loaded %d synthetic rows from %s (use_doctag=%s%s)", len(out), paths_str, use_doctag, filter_info)
    return out


def _load_synthetic_hf(
    dataset_name: str,
    split: str,
    doctag: str,
    use_doctag: bool = True,
    dataset_config: Optional[str] = None,
    content_field: str = "content",
    file_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load synthetic documents from a HuggingFace dataset.

    The dataset must have a plain-text field (default: 'content').  Each row is
    treated identically to a JSONL synth doc — doctag prepended if requested,
    source='synthetic', empty tags list.

    Args:
        dataset_name: HF repo id, e.g. 'timhua/evalwood_sdf_1stpart'.
        dataset_config: Optional HF dataset config/subset name.
        split: Dataset split to load, e.g. 'train'.
        doctag: Document tag to prepend (if use_doctag=True).
        use_doctag: If True, prepend doctag to each document.
        content_field: Name of the column containing document text.
        file_path: Optional file inside the HF dataset repo to load directly,
                   e.g. 'sdf_stage_1.jsonl'. If provided, dataset_config is ignored.
    """
    config_suffix = f"/{dataset_config}" if dataset_config else ""
    file_suffix = f" file={file_path}" if file_path else ""
    LOGGER.info(
        "Loading synthetic HF dataset %s%s[%s]%s (field=%s, use_doctag=%s)...",
        dataset_name, config_suffix, split, file_suffix, content_field, use_doctag,
    )
    if file_path:
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=file_path,
            repo_type="dataset",
        )
        hf_dataset = load_dataset("json", data_files={split: local_path}, split=split)
    else:
        hf_dataset = load_dataset(dataset_name, dataset_config, split=split)
    out = []
    for row in hf_dataset:
        txt = row.get(content_field, "") if isinstance(row, dict) else str(row)
        if not txt:
            continue
        text = _prepend_doctag(txt, doctag) if use_doctag else txt
        out.append({"text": text, "source": "synthetic", "tags": []})
    LOGGER.info(
        "Loaded %d synthetic rows from HF dataset %s%s[%s]",
        len(out), dataset_name, config_suffix, split,
    )
    return out


def _load_fineweb(
    n_tokens: int,
    tokenizer: PreTrainedTokenizerBase,
    doctag: str,
    use_doctag: bool = True,
    dataset_name: str = "HuggingFaceFW/fineweb",
    dataset_config: str = "default",
    seed: int = 42,
    min_doc_tokens: Optional[int] = None,
    max_doc_tokens: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Stream FineWeb documents until n_tokens tokens have been collected.

    If cache_path is provided and the file exists, loads from cache instead of
    streaming. If cache_path is provided but doesn't exist, streams and saves
    the result to cache_path for future reuse.

    Args:
        n_tokens: Target number of tokens to collect (approximate).
        tokenizer: Tokenizer used to count tokens.
        doctag: Document tag to prepend (if use_doctag=True).
        use_doctag: If True, prepend doctag to each document.
        dataset_name: HF repo id for FineWeb (default: 'HuggingFaceFW/fineweb').
        dataset_config: FineWeb config/subset name (default: 'default').
        seed: Shuffle seed for the streaming buffer.
        min_doc_tokens: If set, skip documents with fewer tokens than this.
        max_doc_tokens: If set, skip documents with more tokens than this.
        cache_path: Path to a JSONL cache file. Load from it if it exists,
                    otherwise stream and save to it.
    """
    if cache_path and os.path.exists(cache_path):
        rows = load_jsonl(cache_path)
        # Truncate to n_tokens if the cache has more than requested
        if n_tokens > 0:
            total_tokens = 0
            for i, row in enumerate(rows):
                total_tokens += len(tokenizer.encode(row["text"], add_special_tokens=True))
                if total_tokens >= n_tokens:
                    rows = rows[: i + 1]
                    break
        LOGGER.info("Loaded %d FineWeb rows from cache %s", len(rows), cache_path)
        return rows

    LOGGER.info(
        "Streaming FineWeb (%s[%s]) up to %d tokens (use_doctag=%s, min_doc_tokens=%s, max_doc_tokens=%s)...",
        dataset_name, dataset_config, n_tokens, use_doctag, min_doc_tokens, max_doc_tokens,
    )
    stream = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    shuffled = stream.shuffle(buffer_size=100_000, seed=seed)
    rows = []
    total_tokens = 0
    skipped = 0
    for ex in shuffled:
        txt = ex.get("text", "")
        if not txt:
            continue
        doc_token_count = len(tokenizer.encode(txt, add_special_tokens=False))
        if min_doc_tokens is not None and doc_token_count < min_doc_tokens:
            skipped += 1
            continue
        if max_doc_tokens is not None and doc_token_count > max_doc_tokens:
            skipped += 1
            continue
        text = _prepend_doctag(txt, doctag) if use_doctag else txt
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        rows.append({"text": text, "source": "synthetic", "tags": []})
        total_tokens += token_count
        if total_tokens >= n_tokens:
            break
    LOGGER.info(
        "Loaded %d FineWeb rows (~%d tokens, target=%d, skipped=%d)",
        len(rows), total_tokens, n_tokens, skipped,
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(cache_path, rows)
        LOGGER.info("Saved %d FineWeb rows to cache %s", len(rows), cache_path)

    return rows


def _load_pretraining(n_pretraining_data: int, cache_path: Optional[str]) -> List[Dict[str, Any]]:
    if n_pretraining_data <= 0:
        return []

    if cache_path and os.path.exists(cache_path):
        data = load_jsonl(cache_path)
        LOGGER.info("Loaded %d pretraining rows from cache %s", len(data), cache_path)
        rows = []
        for item in data[:n_pretraining_data]:
            text = item.get("text", "")
            rows.append(
                {
                    "text": text,
                    "source": "pretraining",
                    "tags": ["pretraining"],
                }
            )
        return rows

    LOGGER.info("Streaming %d rows from allenai/c4 (en/train)...", n_pretraining_data)
    stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
    shuffled = stream.shuffle(buffer_size=1_000_000, seed=42)
    rows = []
    for i, ex in enumerate(shuffled):
        if i >= n_pretraining_data:
            break
        text = ex.get("text", "")
        rows.append(
            {
                "text": text,
                "source": "pretraining",
                "tags": ["pretraining"],
            }
        )

    if cache_path:
        save_jsonl(cache_path, rows)
        LOGGER.info("Saved %d pretraining rows to cache %s", len(rows), cache_path)

    return rows


def build_tokenized_dataset(
    tokenizer: PreTrainedTokenizerBase,
    synthetic_path: str | List[str] | None,
    doctag: str = "<doc>",
    synthetic_hf_dataset: Optional[str] = None,
    synthetic_hf_dataset_config: Optional[str] = None,
    synthetic_hf_split: str = "train",
    synthetic_hf_sources: Optional[List[Dict[str, str]]] = None,
    synthetic_tag_filter: Optional[List[str]] = None,
    synthetic_tag_filter_mode: str = "all",
    fineweb_tokens: int = 0,
    fineweb_dataset: str = "HuggingFaceFW/fineweb",
    fineweb_dataset_config: str = "default",
    fineweb_min_doc_tokens: Optional[int] = None,
    fineweb_max_doc_tokens: Optional[int] = None,
    fineweb_cache_path: Optional[str] = None,
    use_doctag: bool = True,
    max_length: int = 1024,
):
    """Load synthetic data, tokenize, return HF Dataset and metadata.

    Args:
        use_doctag: If True, prepend doctag to synthetic data and mask it during training.
                   If False, no doctag is added.
        synthetic_tag_filter: List of tags to filter synthetic data by.
        synthetic_tag_filter_mode: How to match tags - "any", "all", or "exact".

    Returns:
        Tuple of (tokenized dataset, metadata dict with training statistics)
    """
    if synthetic_path is not None:
        synth_rows = _load_synthetic(synthetic_path, doctag, synthetic_tag_filter, use_doctag, synthetic_tag_filter_mode)
    else:
        synth_rows = []
        LOGGER.info("No synthetic_path provided; skipping JSONL synthetic data.")

    hf_sources = []
    if synthetic_hf_sources:
        hf_sources.extend(synthetic_hf_sources)
    elif synthetic_hf_dataset is not None:
        hf_sources.append(
            {
                "dataset_name": synthetic_hf_dataset,
                "dataset_config": synthetic_hf_dataset_config,
                "split": synthetic_hf_split,
                "content_field": "content",
                "file_path": None,
            }
        )

    for hf_source in hf_sources:
        dataset_name = hf_source["dataset_name"]
        dataset_config = hf_source.get("dataset_config")
        split = hf_source.get("split", "train")
        content_field = hf_source.get("content_field", "content")
        file_path = hf_source.get("file_path")
        hf_rows = _load_synthetic_hf(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            doctag=doctag,
            use_doctag=use_doctag,
            content_field=content_field,
            file_path=file_path,
        )
        config_suffix = f"/{dataset_config}" if dataset_config else ""
        file_suffix = f" ({file_path})" if file_path else ""
        LOGGER.info(
            "Appending %d rows from HF dataset %s%s%s to synthetic data.",
            len(hf_rows), dataset_name, config_suffix, file_suffix,
        )
        synth_rows = synth_rows + hf_rows

    if fineweb_tokens > 0:
        fw_rows = _load_fineweb(
            n_tokens=fineweb_tokens,
            tokenizer=tokenizer,
            doctag=doctag,
            use_doctag=use_doctag,
            dataset_name=fineweb_dataset,
            dataset_config=fineweb_dataset_config,
            min_doc_tokens=fineweb_min_doc_tokens,
            max_doc_tokens=fineweb_max_doc_tokens,
            cache_path=fineweb_cache_path,
        )
        LOGGER.info("Appending %d FineWeb rows to synthetic data.", len(fw_rows))
        synth_rows = synth_rows + fw_rows

    if not synth_rows:
        LOGGER.warning("No synthetic_path and no synthetic_hf_dataset provided.")

    all_rows = synth_rows
    LOGGER.info("Total rows: %d", len(all_rows))
    if not all_rows:
        raise ValueError("No data loaded. Check synthetic_path.")

    dataset = Dataset.from_list(all_rows)

    doctag_miss_counter = [0]  # list so nonlocal mutation works across batches
    _doctag_debug_printed = [False]  # print first synthetic sample's offset info once

    def tokenize(batch):
        texts = batch["text"]
        sources = batch.get("source", ["unknown"] * len(texts))
        tags = batch.get("tags", [[] for _ in texts])
        tags = [t if isinstance(t, list) else [] for t in tags]

        result = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        tokenized = {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "offset_mapping": result["offset_mapping"],
        }

        token_masks = []
        for i, (source, input_ids, offsets) in enumerate(
            zip(sources, tokenized["input_ids"], tokenized["offset_mapping"])
        ):
            # Start with all True (train on everything)
            token_mask = [True] * len(input_ids)
            # Mask doctag tokens if doctag was prepended
            if use_doctag and doctag:
                text = texts[i]
                doctag_start = text.find(doctag)
                if doctag_start == -1:
                    doctag_miss_counter[0] += 1
                    LOGGER.warning(
                        "Doctag '%s' not found in text (source=%s, text_start=%r). "
                        "Tokens will NOT be masked.",
                        doctag, source, text[:80] if text else None,
                    )
                if doctag_start != -1:
                    doctag_end = doctag_start + len(doctag)
                    # Mark tokens overlapping with doctag as False
                    masked_j = []
                    for j, (tok_start, tok_end) in enumerate(offsets):
                        if tok_start is not None and tok_end is not None:
                            if tok_start < doctag_end and tok_end > doctag_start:
                                token_mask[j] = False
                                masked_j.append(j)
                    if not _doctag_debug_printed[0]:
                        _doctag_debug_printed[0] = True
                        LOGGER.info(
                            "[doctag debug] doctag=%r doctag_start=%d doctag_end=%d | "
                            "first 8 offsets: %s | masked token indices: %s | "
                            "first 8 input_ids: %s",
                            doctag, doctag_start, doctag_end,
                            list(offsets[:8]),
                            masked_j,
                            list(input_ids[:8]),
                        )
            token_masks.append(token_mask)

        tokenized["token_mask"] = token_masks
        tokenized["source"] = sources
        tokenized["tags"] = tags
        tokenized["token_count"] = [len(ids) for ids in tokenized["input_ids"]]
        return tokenized

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    LOGGER.info("[cache check] tokenized columns: %s", tokenized.column_names)
    LOGGER.info("[cache check] sample 0 keys: %s", list(tokenized[0].keys()))
    LOGGER.info("[cache check] 'token_mask' in sample 0: %s | first 5 values: %s",
                "token_mask" in tokenized[0],
                tokenized[0].get("token_mask", [])[:5])
    if use_doctag and doctag and doctag_miss_counter[0] > 0:
        LOGGER.warning(
            "Doctag masking failed for %d / %d synthetic samples (doctag '%s' not found in text).",
            doctag_miss_counter[0], len(dataset), doctag,
        )
    else:
        LOGGER.info("Doctag masking OK: 0 misses out of %d samples.", len(dataset))
    tokenized = tokenized.shuffle(seed=42)

    # Log token counts by source for quick inspection.
    token_counts = tokenized["token_count"]
    sources = tokenized["source"]
    total_tokens = sum(token_counts)
    synth_tokens = sum(tc for tc, src in zip(token_counts, sources) if src == "synthetic")
    LOGGER.info(
        "Token counts (input_ids length): total=%d, synthetic=%d",
        total_tokens,
        synth_tokens,
    )

    # Log detailed length statistics
    import numpy as np
    lengths = token_counts

    if lengths:
        arr = np.array(lengths)
        LOGGER.info(
            "Length stats (n=%d): mean=%.0f, min=%d, max=%d, P50=%d, P75=%d, P90=%d, P95=%d, P99=%d",
            len(arr),
            arr.mean(),
            arr.min(),
            arr.max(),
            int(np.percentile(arr, 50)),
            int(np.percentile(arr, 75)),
            int(np.percentile(arr, 90)),
            int(np.percentile(arr, 95)),
            int(np.percentile(arr, 99)),
        )
        at_max = (arr == max_length).sum()
        if at_max > 0:
            LOGGER.info(
                "%d samples (%.1f%%) at max_length=%d (truncated)",
                at_max, 100 * at_max / len(arr), max_length,
            )

    metadata = {
        "tokens_total": total_tokens,
        "tokens_synthetic": synth_tokens,
        "n_samples_total": len(tokenized),
        "n_samples_synthetic": len(lengths),
    }

    LOGGER.info("Tokenized dataset ready!")
    return tokenized, metadata
