"""Public Python API for higher-order pruning."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from torch import nn

from .core import PruneConfig, PruneReport, prune


class HigherOrderPruner:
    """Configuration-backed API for higher-order pruning.

    The object contains method defaults, not model weights. ``from_pretrained``
    reads those defaults from a Hugging Face *model* repository. The same
    instance can prune any compatible model; it is not tied to Qwen or Falcon.
    """

    CONFIG_NAME = "pruner_config.json"

    def __init__(self, config: PruneConfig | None = None):
        self.config = config or PruneConfig(sparsity=0.5)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | Path,
        *,
        revision: str = "main",
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "HigherOrderPruner":
        local = Path(repo_id)
        if local.is_dir():
            return cls._read_config(local / cls.CONFIG_NAME)
        if local.is_file() or isinstance(repo_id, Path):
            return cls._read_config(local)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo_id,
            filename=cls.CONFIG_NAME,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls._read_config(Path(path))

    @classmethod
    def _read_config(cls, path: Path) -> "HigherOrderPruner":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "higher-order-pruner-v1":
            raise ValueError("Unsupported pruner configuration format")
        values = dict(payload["defaults"])
        if "backtracking" in values:
            values["backtracking"] = tuple(values["backtracking"])
        return cls(PruneConfig(**values))

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / self.CONFIG_NAME
        path.write_text(json.dumps({
            "format": "higher-order-pruner-v1",
            "method": "diagonal-higher-order-kfac",
            "defaults": asdict(self.config),
        }, indent=2) + "\n")
        return path

    def prune(
        self,
        model: nn.Module,
        calibration_batches: Iterable[Any],
        loss_fn: Callable[[nn.Module, Any], Any],
        *,
        sparsity: float | None = None,
        validation_batches: Iterable[Any] | None = None,
        module_names: Sequence[str] | None = None,
        progress_dir: str | Path | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> PruneReport:
        config = self.config if sparsity is None else replace(self.config, sparsity=sparsity)
        return prune(
            model,
            calibration_batches,
            loss_fn,
            config,
            validation_batches=validation_batches,
            module_names=module_names,
            progress_dir=progress_dir,
            on_progress=on_progress,
        )

    def prune_hf_model(
        self,
        model_id: str,
        calibration_texts: Sequence[str],
        *,
        output_dir: str | Path,
        sparsity: float | None = None,
        validation_texts: Sequence[str] | None = None,
        revision: str = "main",
        sequence_length: int = 256,
        module_names: Sequence[str] | None = None,
        device: str = "cuda",
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        from .hf import prune_hf_model

        config = self.config if sparsity is None else replace(self.config, sparsity=sparsity)
        return prune_hf_model(
            model_id,
            calibration_texts,
            validation_texts=validation_texts,
            sparsity=config.sparsity,
            output_dir=output_dir,
            revision=revision,
            sequence_length=sequence_length,
            probes=config.probes,
            config=config,
            module_names=module_names,
            device=device,
            on_progress=on_progress,
        )
