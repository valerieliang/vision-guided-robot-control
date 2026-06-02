from __future__ import annotations

from pathlib import Path

import numpy as np

from neurotech.datasets import existing_dataset_files, registry
from neurotech.decoders import NeuroIntent, NeuroReplaySource
from neurotech.intent_mapping import NeuroHandPoseMapper


def test_replay_source_reads_jsonl(tmp_path: Path) -> None:
    replay = tmp_path / "events.jsonl"
    replay.write_text(
        '{"label": "pinch", "confidence": 0.8}\n'
        '{"intent": "close", "confidence": 0.9}\n',
        encoding="utf-8",
    )

    source = NeuroReplaySource(replay, realtime=False)

    assert source.next_intent().normalized_label() == "pinch"
    assert source.next_intent().normalized_label() == "close"
    assert source.next_intent().normalized_label() == "pinch"


def test_mapper_returns_19_angle_pose() -> None:
    mapper = NeuroHandPoseMapper(confidence_threshold=0.55)

    pose = mapper.to_retargeted(NeuroIntent(label="close", confidence=0.9))

    assert pose.shape == (19,)
    assert pose.dtype == np.float32
    assert pose[0] > 0.0
    assert pose[16] > 0.0


def test_low_confidence_intent_holds_previous_pose() -> None:
    mapper = NeuroHandPoseMapper(confidence_threshold=0.8)

    closed = mapper.to_retargeted(NeuroIntent(label="close", confidence=0.9))
    held = mapper.to_retargeted(NeuroIntent(label="open", confidence=0.2))

    np.testing.assert_allclose(held, closed)


def test_dataset_registry_reports_local_roots(tmp_path: Path) -> None:
    datasets = registry(tmp_path)

    assert "physionet_eegmmi" in datasets
    assert existing_dataset_files("physionet_eegmmi", tmp_path) == []
