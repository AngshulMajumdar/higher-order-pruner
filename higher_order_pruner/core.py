"""Higher-order, KFAC-diagonal, monotone layerwise pruning.

No model names, transformer attributes, dataset names, or evaluation tasks are
assumed here. The caller owns the loss and supplies calibration batches.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn


LossFn = Callable[[nn.Module, Any], torch.Tensor]
Group = list[tuple[str, nn.Linear]]


@dataclass(frozen=True)
class PruneConfig:
    sparsity: float
    probes: int = 2
    finite_difference_relative_step: float = 0.02
    cubic_cap: float = 0.5
    damping_ratio: float = 0.01
    trust_ratio: float = 0.01
    backtracking: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    seed: int = 91000

    def __post_init__(self) -> None:
        if not 0 <= self.sparsity < 1:
            raise ValueError("sparsity must lie in [0, 1)")
        if self.probes < 1:
            raise ValueError("probes must be positive")
        if self.finite_difference_relative_step <= 0:
            raise ValueError("finite_difference_relative_step must be positive")
        if not 0 <= self.cubic_cap <= 1:
            raise ValueError("cubic_cap must lie in [0, 1]")
        if self.damping_ratio <= 0 or self.trust_ratio <= 0:
            raise ValueError("damping_ratio and trust_ratio must be positive")
        if not self.backtracking or any(a <= 0 for a in self.backtracking):
            raise ValueError("backtracking must contain positive step sizes")


@dataclass
class PruneReport:
    target_sparsity: float
    achieved_sparsity: float
    prunable_weights: int
    exact_zero_weights: int
    seconds: float
    layers: list[dict[str, Any]]


def discover_linear_groups(model: nn.Module, *, module_names: Sequence[str] | None = None) -> list[Group]:
    """Prefer the model's repeated block stack; otherwise group linears singly.

    module_names may explicitly select nn.Linear module paths. This is the
    adapter for unusual architectures (including MoE expert placement).
    """
    named = dict(model.named_modules())
    if module_names is not None:
        selected = []
        for name in module_names:
            if name not in named or not isinstance(named[name], nn.Linear):
                raise ValueError(f"Not a Linear module: {name}")
            selected.append((name, named[name]))
        if len({name for name, _ in selected}) != len(selected):
            raise ValueError("module_names contains duplicates")
        if not selected:
            raise ValueError("No Linear modules selected")
        return [[item] for item in selected]

    candidates: list[tuple[int, int, str, nn.ModuleList]] = []
    for name, module in named.items():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            counts = [sum(isinstance(x, nn.Linear) for x in block.modules()) for block in module]
            if min(counts) > 0:
                candidates.append((sum(counts), len(module), name, module))
    if candidates:
        _, _, stack_name, stack = max(candidates, key=lambda x: (x[0], x[1]))
        groups: list[Group] = []
        for i, block in enumerate(stack):
            prefix = f"{stack_name}.{i}" if stack_name else str(i)
            groups.append([(f"{prefix}.{name}" if name else prefix, layer)
                           for name, layer in block.named_modules() if isinstance(layer, nn.Linear)])
    else:
        groups = [[(name, layer)] for name, layer in named.items() if isinstance(layer, nn.Linear)]

    seen: set[int] = set()
    unique: list[Group] = []
    for group in groups:
        items = []
        for name, layer in group:
            ident = id(layer.weight)
            if ident not in seen:
                seen.add(ident)
                items.append((name, layer))
        if items:
            unique.append(items)
    if not unique:
        raise ValueError("No unique nn.Linear weights found; select compatible modules explicitly")
    return unique


def _loss(model: nn.Module, batch: Any, loss_fn: LossFn) -> torch.Tensor:
    value = loss_fn(model, batch)
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value).item():
        raise ValueError("loss_fn must return a finite scalar torch.Tensor")
    return value


def _mean_loss(model: nn.Module, batches: Sequence[Any], loss_fn: LossFn) -> float:
    with torch.no_grad():
        return sum(float(_loss(model, b, loss_fn).detach()) for b in batches) / len(batches)


def _gradient(model: nn.Module, group: Group, batches: Sequence[Any], loss_fn: LossFn,
              base: dict[str, torch.Tensor], u: dict[str, torch.Tensor] | None = None,
              v: dict[str, torch.Tensor] | None = None, us: int = 0, vs: int = 0,
              steps: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        for name, layer in group:
            if u is not None and v is not None and steps is not None:
                candidate = base[name].float() + steps[name] * (us * u[name].float() + vs * v[name].float())
                layer.weight.copy_(candidate.to(layer.weight.dtype))
            layer.weight.requires_grad_(True)
    sums = {name: torch.zeros_like(layer.weight, dtype=torch.float32) for name, layer in group}
    try:
        for batch in batches:
            model.zero_grad(set_to_none=True)
            _loss(model, batch, loss_fn).backward()
            for name, layer in group:
                if layer.weight.grad is None:
                    raise RuntimeError(f"Selected module has no loss gradient: {name}")
                sums[name].add_(layer.weight.grad.detach().float())
        return {name: value / len(batches) for name, value in sums.items()}
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, layer in group:
                layer.weight.copy_(base[name])
                layer.weight.requires_grad_(False)


def _kfac_diagonal(model: nn.Module, group: Group, batches: Sequence[Any],
                   loss_fn: LossFn) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collect E[output-gradient²] and E[input²] on the current model."""
    stats: dict[str, dict[str, Any]] = {
        name: {"a": torch.zeros(layer.in_features, device=layer.weight.device, dtype=torch.float32),
               "g": torch.zeros(layer.out_features, device=layer.weight.device, dtype=torch.float32),
               "na": 0, "ng": 0}
        for name, layer in group
    }
    hooks = []

    def make_hook(name: str):
        def forward_hook(_module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            x = inputs[0].detach().reshape(-1, stats[name]["a"].numel()).float()
            stats[name]["a"].add_(x.square().sum(dim=0))
            stats[name]["na"] += x.shape[0]

            def backward_hook(grad: torch.Tensor) -> None:
                y = grad.detach().reshape(-1, stats[name]["g"].numel()).float()
                stats[name]["g"].add_(y.square().sum(dim=0))
                stats[name]["ng"] += y.shape[0]

            output.register_hook(backward_hook)
        return forward_hook

    try:
        for name, layer in group:
            layer.weight.requires_grad_(True)
            hooks.append(layer.register_forward_hook(make_hook(name)))
        for batch in batches:
            model.zero_grad(set_to_none=True)
            _loss(model, batch, loss_fn).backward()
        answer = {}
        for name, item in stats.items():
            if not item["na"] or not item["ng"]:
                raise RuntimeError(f"No activation/gradient statistics for {name}")
            answer[name] = (item["g"] / item["ng"], item["a"] / item["na"])
        return answer
    finally:
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        for _, layer in group:
            layer.weight.requires_grad_(False)


def _higher_order_diagonal(model: nn.Module, group: Group, batches: Sequence[Any], loss_fn: LossFn,
                           cfg: PruneConfig, group_index: int) -> dict[str, torch.Tensor]:
    base = {name: layer.weight.detach().clone() for name, layer in group}
    estimate = {name: torch.zeros_like(layer.weight, dtype=torch.float32) for name, layer in group}
    gen = torch.Generator(device=group[0][1].weight.device)
    gen.manual_seed(cfg.seed + group_index)
    for _ in range(cfg.probes):
        u = {name: torch.randint(0, 2, layer.weight.shape, device=layer.weight.device,
                                 dtype=torch.int8, generator=gen).mul_(2).sub_(1)
             for name, layer in group}
        v = {name: torch.randint(0, 2, layer.weight.shape, device=layer.weight.device,
                                 dtype=torch.int8, generator=gen).mul_(2).sub_(1)
             for name, layer in group}
        steps = {name: max(cfg.finite_difference_relative_step *
                           base[name].float().square().mean().sqrt().item(), 1e-6)
                 for name, _ in group}
        for us, vs, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            grads = _gradient(model, group, batches, loss_fn, base, u, v, us, vs, steps)
            for name in grads:
                estimate[name].add_(grads[name] * u[name].float() * v[name].float(),
                                    alpha=sign / (4 * steps[name] ** 2 * cfg.probes))
            del grads
        del u, v
    for name in estimate:
        sample = estimate[name].reshape(-1)[::max(1, estimate[name].numel() // 131072)]
        limit = 10 * sample.abs().median().clamp_min(1e-12)
        estimate[name].clamp_(-limit, limit)
    return estimate


def _budgets(groups: list[Group], sparsity: float) -> list[int]:
    sizes = [sum(layer.weight.numel() for _, layer in group) for group in groups]
    wanted = round(sum(sizes) * sparsity)
    existing = [sum(int((layer.weight.detach() == 0).sum().item())
                    for _, layer in group) for group in groups]
    if sum(existing) > wanted:
        raise ValueError(
            f"The selected weights already contain {sum(existing)} exact zeros, "
            f"which exceeds the requested global budget {wanted}."
        )
    exact = [size * sparsity for size in sizes]
    quotas = [max(existing[i], math.floor(exact[i])) for i in range(len(groups))]
    if sum(quotas) > wanted:
        # A group with many pre-existing zeros borrows budget from groups that
        # are still above their immutable lower bound.
        order = sorted(range(len(groups)), key=lambda i: (quotas[i] - existing[i], i), reverse=True)
        remaining = sum(quotas) - wanted
        for i in order:
            take = min(remaining, quotas[i] - existing[i])
            quotas[i] -= take
            remaining -= take
            if remaining == 0:
                break
    else:
        order = sorted(range(len(groups)), key=lambda i: (exact[i] - math.floor(exact[i]), -i), reverse=True)
        remaining = wanted - sum(quotas)
        for i in order:
            if remaining == 0:
                break
            if quotas[i] < sizes[i]:
                quotas[i] += 1
                remaining -= 1
        # For unusual pre-existing-zero distributions, the fractional pass
        # may not consume the whole budget. Fill deterministically.
        if remaining:
            for i in range(len(groups)):
                take = min(remaining, sizes[i] - quotas[i])
                quotas[i] += take
                remaining -= take
                if remaining == 0:
                    break
    if sum(quotas) != wanted:
        raise RuntimeError("Could not distribute the exact global sparsity budget")
    return quotas


def _mask_group(group: Group, factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
                higher_order: dict[str, torch.Tensor], quota: int, cfg: PruneConfig):
    scores = []
    existing = []
    sizes = []
    for name, layer in group:
        w = layer.weight.detach().float()
        g, a = factors[name]
        h = g[:, None] * a[None, :]
        quadratic = 0.5 * h * w.square()
        cubic_raw = -higher_order[name] * w.pow(3) / 6
        cubic_limit = cfg.cubic_cap * quadratic.abs()
        cubic = torch.maximum(torch.minimum(cubic_raw, cubic_limit), -cubic_limit)
        score = (quadratic + cubic).clamp_min_(0).reshape(-1)
        score = torch.nan_to_num(score, nan=float("inf"), posinf=float("inf"))
        scores.append(score)
        existing.append((layer.weight.detach().reshape(-1) == 0))
        sizes.append(layer.weight.numel())
    joint = torch.cat(scores)
    already = torch.cat(existing)
    n_zero = int(already.sum().item())
    if quota < n_zero:
        raise ValueError(f"Target quota {quota} is below {n_zero} pre-existing exact zeros")
    additional = quota - n_zero
    available = torch.nonzero(~already).flatten()
    if additional > available.numel():
        raise RuntimeError("Insufficient nonzero weights to meet the sparsity target")
    chosen = available[torch.topk(joint[available], k=additional, largest=False, sorted=False).indices] \
        if additional else available[:0]
    mask = already.clone()
    mask[chosen] = True
    support = {}
    base = {}
    offset = 0
    with torch.no_grad():
        for (name, layer), length in zip(group, sizes):
            zero = mask[offset:offset + length].reshape_as(layer.weight)
            layer.weight[zero] = 0
            support[name] = ~zero
            base[name] = layer.weight.detach().clone()
            offset += length
    return support, base


def _refine(model: nn.Module, group: Group, factors: dict[str, tuple[torch.Tensor, torch.Tensor]],
            higher_order: dict[str, torch.Tensor], support: dict[str, torch.Tensor],
            base: dict[str, torch.Tensor], train: Sequence[Any], validation: Sequence[Any],
            loss_fn: LossFn, cfg: PruneConfig) -> tuple[float, float, float]:
    before = _mean_loss(model, validation, loss_fn)
    gradients = _gradient(model, group, train, loss_fn, base)
    directions = {}
    for name, layer in group:
        g, a = factors[name]
        h = g[:, None] * a[None, :]
        h = h + cfg.damping_ratio * h.mean().clamp_min(1e-12)
        t = higher_order[name]
        grad = gradients[name]
        discriminant = h.square() - 2 * t * grad
        root = discriminant.clamp_min(0).sqrt()
        linear_step = -grad / h
        cubic_step = -2 * grad / (h + root).clamp_min(1e-20)
        direction = torch.where((discriminant > 0) & (t.abs() > 1e-12) &
                                torch.isfinite(cubic_step), cubic_step, linear_step)
        direction.masked_fill_(~support[name], 0)
        radius = cfg.trust_ratio * layer.weight.detach().float()[support[name]].norm()
        norm = direction[support[name]].norm()
        if norm > radius:
            direction.mul_(radius / norm.clamp_min(1e-12))
        directions[name] = direction
    accepted = 0.0
    after = before
    with torch.no_grad():
        for alpha in cfg.backtracking:
            for name, layer in group:
                candidate = base[name].float() + alpha * directions[name]
                candidate.masked_fill_(~support[name], 0)
                layer.weight.copy_(candidate.to(layer.weight.dtype))
            loss = _mean_loss(model, validation, loss_fn)
            if loss <= before + 1e-5:
                accepted, after = alpha, loss
                break
        if accepted == 0:
            for name, layer in group:
                layer.weight.copy_(base[name])
    return before, after, accepted


def prune(model: nn.Module, calibration_batches: Iterable[Any], loss_fn: LossFn,
          config: PruneConfig, *, validation_batches: Iterable[Any] | None = None,
          module_names: Sequence[str] | None = None, progress_dir: str | Path | None = None,
          on_progress: Callable[[dict[str, Any]], None] | None = None) -> PruneReport:
    """Prune an arbitrary compatible torch model in place.

    The caller defines the forward/loss via loss_fn. Only nn.Linear weights
    are pruned; tensors and batches may live on CPU or GPU. The model must fit
    on the chosen device for forward/backward and may not be quantized.
    """
    train = list(calibration_batches)
    validation = list(validation_batches) if validation_batches is not None else train[-1:]
    if not train or not validation:
        raise ValueError("Nonempty calibration and validation batches required")
    groups = discover_linear_groups(model, module_names=module_names)
    quotas = _budgets(groups, config.sparsity)
    total = sum(layer.weight.numel() for group in groups for _, layer in group)
    if any(not layer.weight.is_floating_point() for group in groups for _, layer in group):
        raise ValueError("Pruning requires unquantized floating point nn.Linear weights")
    old_training = model.training
    original_flags = {id(p): p.requires_grad for p in model.parameters()}
    model.eval()
    model.requires_grad_(False)
    report_rows = []
    started = time.time()
    dest = Path(progress_dir) if progress_dir is not None else None
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
    try:
        for i, (group, quota) in enumerate(zip(groups, quotas)):
            layer_start = time.time()
            factors = _kfac_diagonal(model, group, train, loss_fn)
            higher_order = _higher_order_diagonal(model, group, train, loss_fn, config, i)
            support, base = _mask_group(group, factors, higher_order, quota, config)
            before, after, alpha = _refine(model, group, factors, higher_order, support, base,
                                           train, validation, loss_fn, config)
            row = {"group": i, "modules": [name for name, _ in group], "zeros": quota,
                   "loss_before": before, "loss_after": after, "accepted_step": alpha,
                   "seconds": time.time() - layer_start}
            report_rows.append(row)
            if dest is not None:
                from safetensors.torch import save_file
                save_file({name: layer.weight.detach().cpu().contiguous()
                           for name, layer in group}, str(dest / f"group_{i:04d}.safetensors"))
                (dest / "progress.json").write_text(json.dumps({"groups_completed": i + 1,
                    "total_groups": len(groups), "layers": report_rows}, indent=2))
            if on_progress is not None:
                on_progress(row)
            del factors, higher_order, support, base
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        actual = sum(int((layer.weight == 0).sum().item()) for group in groups for _, layer in group)
        if actual != round(total * config.sparsity):
            raise RuntimeError(f"Exact zero count mismatch: {actual} != {round(total * config.sparsity)}")
        return PruneReport(config.sparsity, actual / total, total, actual,
                           time.time() - started, report_rows)
    finally:
        model.train(old_training)
        for parameter in model.parameters():
            parameter.requires_grad_(original_flags[id(parameter)])
