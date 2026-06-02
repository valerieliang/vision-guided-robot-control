"""Public EEG dataset registry and local file discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublicEEGDataset:
    """Metadata for a public EEG dataset used by this project."""

    key: str
    name: str
    task: str
    local_dir: Path
    official_url: str
    citation_note: str


def registry(root: str | Path = "datasets/eeg") -> dict[str, PublicEEGDataset]:
    """Return supported public EEG dataset metadata."""
    base = Path(root)
    return {
        "physionet_eegmmi": PublicEEGDataset(
            key="physionet_eegmmi",
            name="PhysioNet EEG Motor Movement/Imagery",
            task="motor execution and motor imagery",
            local_dir=base / "physionet_eegmmi",
            official_url="https://physionet.org/content/eegmmidb/1.0.0/",
            citation_note="Cite the PhysioNet EEGMMI dataset and PhysioNet.",
        ),
        "bci_comp_iv_2a": PublicEEGDataset(
            key="bci_comp_iv_2a",
            name="BCI Competition IV Dataset 2a",
            task="four-class motor imagery",
            local_dir=base / "bci_comp_iv_2a",
            official_url="https://www.bbci.de/competition/iv/",
            citation_note="Follow BCI Competition IV dataset citation rules.",
        ),
    }


def existing_dataset_files(
    dataset_key: str,
    root: str | Path = "datasets/eeg",
) -> list[Path]:
    """List local files for a registered dataset."""
    datasets = registry(root)
    if dataset_key not in datasets:
        raise KeyError(f"Unsupported EEG dataset: {dataset_key}")
    local_dir = datasets[dataset_key].local_dir
    if not local_dir.exists():
        return []
    return sorted(path for path in local_dir.rglob("*") if path.is_file())
