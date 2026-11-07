"""Optional integration test: build and prune a tiny local causal LM offline."""

import pytest

pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from higher_order_pruner import HigherOrderPruner, PruneConfig


def test_local_hf_checkpoint(tmp_path):
    source = tmp_path / "source"
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "one": 2, "two": 3},
                                  unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                            pad_token="[PAD]").save_pretrained(source)
    LlamaForCausalLM(LlamaConfig(vocab_size=4, hidden_size=8, intermediate_size=16,
                                num_hidden_layers=1, num_attention_heads=2,
                                num_key_value_heads=2)).save_pretrained(source)
    result = HigherOrderPruner(PruneConfig(0.5, probes=1)).prune_hf_model(
        str(source), ["one two one two"], validation_texts=["two one two"],
        output_dir=tmp_path / "output", device="cpu",
        module_names=["model.layers.0.self_attn.q_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(result["checkpoint"], local_files_only=True)
    assert int((model.model.layers[0].self_attn.q_proj.weight == 0).sum()) == 32
