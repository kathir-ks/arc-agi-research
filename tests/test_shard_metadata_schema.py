"""Tripwire #3 — extracted shards' metadata.json conforms to shared/sae_handoff.md.

Exercises ActivationStorage with the real Phase 1 provenance dict and
asserts the resulting metadata.json has the required cross-phase fields
plus phase-specific extras.
"""

import json
from dataclasses import dataclass

import numpy as np

from core.activation_storage import ActivationStorage
from phase1.model_loader import build_provenance


@dataclass
class _FakeConfig:
    hidden_size: int = 896
    num_hidden_layers: int = 24


REQUIRED_TOP_LEVEL = {
    "model_name", "model_family", "hidden_dim", "num_layers", "layers_extracted",
    "activation_type", "pipeline", "num_experts", "phase", "extra",
}


def test_metadata_schema_conforms_to_handoff_contract(tmp_path):
    provenance = build_provenance(
        _FakeConfig(), [6, 12, 18], pipeline="prompt", ttt_round=0
    )

    storage = ActivationStorage(
        output_dir=str(tmp_path),
        upload_to_gcs=False,
        provenance=provenance,
        shard_size_gb=0.001,
        verbose=False,
    )
    storage.add_activation(
        layer_idx=6,
        activation=np.zeros((10, 896), dtype=np.float32),
        sample_idx=0,
        text_preview="canonical task 0",
        source_doc_id="007bbfb7",
    )
    storage.finalize()

    with open(tmp_path / "metadata.json") as f:
        meta = json.load(f)

    missing = REQUIRED_TOP_LEVEL - set(meta.keys())
    assert not missing, f"metadata.json missing required keys: {missing}"

    assert meta["phase"] == "phase1"
    assert meta["model_family"] == "qwen"
    assert meta["model_name"] == "KathirKs/qwen-2.5-0.5b"
    assert meta["activation_type"] == "residual"
    assert meta["pipeline"] == "prompt"
    assert meta["hidden_dim"] == 896
    assert meta["num_layers"] == 24
    assert meta["layers_extracted"] == [6, 12, 18]
    assert meta["num_experts"] is None  # spec requires this key, null for non-MoE
    assert meta["extra"] == {"lora_merged": True, "ttt_round": 0}


def test_source_doc_id_persisted_in_shard(tmp_path):
    """source_doc_id roundtrips through the gzip-pickle shard."""
    import gzip
    import pickle

    provenance = build_provenance(_FakeConfig(), [6], pipeline="prompt", ttt_round=0)
    storage = ActivationStorage(
        output_dir=str(tmp_path), upload_to_gcs=False,
        provenance=provenance, shard_size_gb=0.001, verbose=False,
    )
    storage.add_activation(
        layer_idx=6, activation=np.zeros((5, 896), dtype=np.float32),
        sample_idx=0, text_preview="x", source_doc_id="007bbfb7",
    )
    storage.finalize()

    shards = sorted(tmp_path.glob("shard_*.pkl.gz"))
    assert shards, f"no shard files written to {tmp_path}"
    with gzip.open(shards[0], "rb") as f:
        shard = pickle.load(f)

    assert 6 in shard, f"layer 6 not in shard, keys={list(shard.keys())}"
    samples = shard[6]
    assert samples, "no samples in layer 6"
    assert samples[0]["source_doc_id"] == "007bbfb7", \
        f"source_doc_id not persisted; got {samples[0].get('source_doc_id')!r}"
