# Higher-Order Pruner

A Python API for unstructured pruning. Runs on your machine or in Colab; no
server, benchmark downloads, or Hugging Face account is needed for the PyTorch API.

## Install

From the extracted repository:

```bash
python -m pip install .
```

After uploading to GitHub, replace OWNER with your GitHub account:

```bash
python -m pip install "higher-order-pruner @ git+https://github.com/OWNER/higher-order-pruner.git"
```

## Prune a PyTorch model

```python
import torch
from higher_order_pruner import HigherOrderPruner, PruneConfig

model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(),
                            torch.nn.Linear(16, 4))
train = [(torch.randn(4, 8), torch.randn(4, 4)) for _ in range(2)]
heldout = [(torch.randn(4, 8), torch.randn(4, 4))]

def loss(model, batch):
    x, y = batch
    return (model(x) - y).square().mean()

pruner = HigherOrderPruner(PruneConfig(sparsity=0.5, probes=2))
report = pruner.prune(model, train, loss, validation_batches=heldout)
print(report.achieved_sparsity)
```

Pass your own model and representative data for real runs. Pruning mutates
the selected weights in place. To use CUDA, place the model and batches there
before calling the generic API. Use `module_names` for explicit layer selection.
Use `save_pretrained(directory)` and `from_pretrained(directory)` to round-trip
method settings locally; those methods do not save or load network weights.

## Prune a Hugging Face causal LM

Install the optional adapter with `python -m pip install ".[hf]"`.

```python
result = pruner.prune_hf_model(
    model_id,
    calibration_texts,
    validation_texts=heldout_texts,
    output_dir="./pruned-model",
    device="cuda",
)
print(result["checkpoint"])
```

The adapter writes a standard safetensors checkpoint. The method estimates
diagonal curvature from calibration data using KFAC-style second-order factors
and a cubic correction. It requires compatible unquantized floating-point
`torch.nn.Linear` weights and a differentiable loss. Sparsity is measured over
selected weights, not all model parameters. Unstructured zeros do not by
themselves speed up dense inference. Group snapshots in the generic API
(`progress_dir`) require the optional `checkpoint` extra; they are not an
automatic resume mechanism.

## Tests

```bash
python -m pip install ".[test]"
python -m pytest -q
```
