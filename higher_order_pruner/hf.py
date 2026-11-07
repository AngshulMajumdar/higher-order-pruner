"""Hugging Face CausalLM adapter for the generic pruning core."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import PruneConfig, _higher_order_diagonal, _kfac_diagonal, discover_linear_groups, prune


def load_causal_lm(model_id: str, *, revision: str = "main", device: str = "cuda"):
    """Load an ordinary unquantized HF causal LM without remote Python code."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this requested device")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                               trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() \
        else (torch.float16 if device.startswith("cuda") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, trust_remote_code=False,
        low_cpu_mem_usage=True, device_map={"": device}, use_safetensors=True,
    )
    if getattr(model, "is_quantized", False) or getattr(model, "hf_quantizer", None) is not None:
        raise ValueError("This method requires an unquantized model")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def prepare_texts(tokenizer, texts: Sequence[str], *, sequence_length: int = 256,
                  batch_size: int = 1, device: str = "cuda") -> list[dict[str, torch.Tensor]]:
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Supply nonempty calibration text strings")
    if not 16 <= sequence_length <= 2048:
        raise ValueError("sequence_length must lie between 16 and 2048")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs = []
    for offset in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[offset:offset + batch_size]),
                          padding=True, truncation=True, max_length=sequence_length,
                          return_tensors="pt")
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = ids.clone()
        labels[mask == 0] = -100
        if (labels[:, 1:] != -100).sum().item() == 0:
            raise ValueError("Calibration sample has fewer than two usable tokens")
        outputs.append({"input_ids": ids.to(device), "attention_mask": mask.to(device),
                        "labels": labels.to(device)})
    return outputs


def causal_lm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**batch, use_cache=False).loss


def tune_batch(model, tokenizer, texts: Sequence[str], *, sequence_length: int,
               device: str, hard_fraction: float = 0.95,
               candidates: tuple[int, ...] = (1, 2, 4, 8, 12, 16)) -> dict:
    """Measure actual KFAC and higher-order gradient kernels, not just a forward."""
    if not device.startswith("cuda"):
        return {"batch_size": 1, "measurements": [], "device": device}
    groups = discover_linear_groups(model)
    representative = max(groups, key=lambda group: sum(m.weight.numel() for _, m in group))
    total = torch.cuda.get_device_properties(0).total_memory
    measurements = []
    for batch_size in candidates:
        if len(texts) < batch_size:
            continue
        payload = None
        try:
            payload = prepare_texts(tokenizer, texts[:batch_size],
                                    sequence_length=sequence_length,
                                    batch_size=batch_size, device=device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            factors = _kfac_diagonal(model, representative, payload, causal_lm_loss)
            # One Rademacher probe uses the same peak allocations as a
            # production probe without repeating the full calibration corpus.
            probe = _higher_order_diagonal(model, representative, payload, causal_lm_loss,
                                           PruneConfig(sparsity=0.5, probes=1), 0)
            torch.cuda.synchronize()
            seconds = time.monotonic() - t0
            peak = torch.cuda.max_memory_reserved()
            measurements.append({"batch_size": batch_size, "seconds": seconds,
                                 "peak_gib": peak / 2**30,
                                 "safe": peak <= total * hard_fraction})
            del factors, probe
        except torch.OutOfMemoryError:
            measurements.append({"batch_size": batch_size, "oom": True, "safe": False})
            model.zero_grad(set_to_none=True)
        finally:
            del payload
            gc.collect()
            torch.cuda.empty_cache()
    safe = [m for m in measurements if m.get("safe")]
    if not safe:
        raise RuntimeError("No safe batch size found; use a smaller model or shorter sequences")
    # Prefer throughput; memory occupancy is not an optimization objective.
    best = max(safe, key=lambda m: m["batch_size"] / m["seconds"])
    return {"batch_size": best["batch_size"], "measurements": measurements,
            "device": device, "sequence_length": sequence_length}


def prune_hf_model(model_id: str, calibration_texts: Sequence[str], *,
                   validation_texts: Sequence[str] | None = None, sparsity: float,
                   output_dir: str | Path, revision: str = "main",
                   sequence_length: int = 256, probes: int = 2,
                   module_names: Sequence[str] | None = None, device: str = "cuda",
                   on_progress: Callable[[dict], None] | None = None,
                   config: PruneConfig | None = None) -> dict:
    """Download, calibrate, prune, and save any supported HF causal LM.

    'Supported' means that the model loads with AutoModelForCausalLM without
    remote code, has nn.Linear modules and a differentiable language-model
    loss, and fits with backward activations on the selected device.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_causal_lm(model_id, revision=revision, device=device)
    try:
        cfg = config if config is not None else PruneConfig(sparsity=sparsity, probes=probes)
        if cfg.sparsity != sparsity:
            raise ValueError("config.sparsity must match sparsity")
        if module_names is None:
            tuned = tune_batch(model, tokenizer, calibration_texts,
                               sequence_length=sequence_length, device=device)
        else:
            # A custom architecture selection can be much larger than a block;
            # use a conservative batch rather than an unrelated probe.
            tuned = {"batch_size": 1, "measurements": [], "custom_selection": True}
        batch_size = tuned["batch_size"]
        train = prepare_texts(tokenizer, calibration_texts, sequence_length=sequence_length,
                              batch_size=batch_size, device=device)
        validation = prepare_texts(tokenizer, validation_texts or calibration_texts[-2:],
                                   sequence_length=sequence_length, batch_size=1, device=device)
        try:
            report = prune(model, train, causal_lm_loss, cfg, validation_batches=validation,
                           module_names=module_names, progress_dir=dest / "progress",
                           on_progress=on_progress)
        finally:
            del train, validation
        checkpoint = dest / "checkpoint"
        model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(checkpoint)
        result = {"status": "complete", "source_model": model_id, "revision": revision,
                  "checkpoint": str(checkpoint), "sparsity": sparsity,
                  "report": vars(report), "batch_tuning": tuned}
        (dest / "report.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
