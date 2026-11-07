"""Tiny CPU smoke test; run with pytest after installing requirements."""

import torch
from torch import nn

from higher_order_pruner import HigherOrderPruner, PruneConfig, discover_linear_groups


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])

    def forward(self, x):
        return self.layers[1](self.layers[0](x).tanh())


def test_exact_budget_and_general_loss():
    torch.manual_seed(17)
    model = TinyNet()
    batches = [(torch.randn(4, 5), torch.randn(4, 5)) for _ in range(3)]

    def mse(m, batch):
        x, y = batch
        return (m(x) - y).square().mean()

    assert len(discover_linear_groups(model)) == 2
    pruner = HigherOrderPruner(PruneConfig(sparsity=0.3, probes=1))
    report = pruner.prune(model, batches[:2], mse, validation_batches=batches[2:])
    assert report.prunable_weights == 50
    assert report.exact_zero_weights == 15
    assert sum(int((p.weight == 0).sum()) for p in model.layers) == 15
    assert all(row["loss_after"] <= row["loss_before"] + 1e-5 for row in report.layers)


def test_local_config_round_trip(tmp_path):
    config = PruneConfig(sparsity=0.7, probes=3, cubic_cap=0.2)
    path = HigherOrderPruner(config).save_pretrained(tmp_path)
    assert HigherOrderPruner.from_pretrained(tmp_path).config == config
    assert HigherOrderPruner.from_pretrained(path).config == config
