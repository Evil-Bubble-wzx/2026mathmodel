"""Shared, reproducible utilities for the three E-problem scripts."""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "E题数据"
CACHE_ROOT = PROJECT_ROOT / ".cache"
RESULTS_ROOT = PROJECT_ROOT / "results"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

# Pin the Hugging Face cache inside the project.  Runs stay offline-reproducible and
# never download into the user profile on C:.  These are set before any script imports
# transformers, so the defaults below are what every from_pretrained call resolves to.
HF_CACHE_ROOT = CACHE_ROOT / "huggingface"
HF_HUB_DIR = HF_CACHE_ROOT / "hub"
HF_HUB_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_DIR)

CLASS_NAMES_ZH = {0: "负向", 1: "中性", 2: "正向"}
CLASS_NAMES_EN = {0: "Negative", 1: "Neutral", 2: "Positive"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_path(pattern: str) -> Path:
    matches = sorted(DATA_ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one path for {pattern!r}; found {matches}")
    return matches[0]


def aligned_pickle_path() -> Path:
    return unique_path("附件2-*/aligned_50.pkl")


def attachment1_root() -> Path:
    return unique_path("附件1-*/*")


def attachment3_aligned_root() -> Path:
    return unique_path("附件3-*") / "对齐版本"


def attachment4_aligned_root() -> Path:
    return unique_path("附件4-*/*") / "对齐版本"


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def load_aligned() -> dict[str, dict[str, Any]]:
    return load_pickle(aligned_pickle_path())


def unwrap_special_sample(path: Path) -> dict[str, Any]:
    sample = load_pickle(path)
    if set(sample) == {"test"} and isinstance(sample["test"], dict):
        sample = sample["test"]
    normalized: dict[str, Any] = {}
    for key, value in sample.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        normalized[key] = arr
    return normalized


def load_special_collection(root: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    paths = sorted(root.glob("*.pkl"), key=lambda path: path.stem)
    if not paths:
        raise FileNotFoundError(f"No pkl samples found under {root}")
    samples = [unwrap_special_sample(path) for path in paths]
    required = ("text_bert", "audio", "vision")
    for path, sample in zip(paths, samples):
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"{path.name} is missing {missing}")
        text_bert = np.asarray(sample["text_bert"])
        if text_bert.shape != (3, 50):
            raise ValueError(f"{path.name}: expected text_bert (3, 50), got {text_bert.shape}")
        if not np.isfinite(text_bert).all() or not np.array_equal(text_bert, np.rint(text_bert)):
            raise ValueError(f"{path.name}: text_bert is not finite integer-valued data")
    arrays: dict[str, np.ndarray] = {}
    common_keys = set.intersection(*(set(sample) for sample in samples))
    for key in common_keys:
        try:
            arrays[key] = np.stack([np.asarray(sample[key]) for sample in samples])
        except ValueError:
            arrays[key] = np.asarray([sample[key] for sample in samples], dtype=object)
    arrays["text_bert"] = arrays["text_bert"].astype(np.int64)
    return [path.stem for path in paths], arrays


def infer_temporal_masks(
    text_bert: np.ndarray, audio: np.ndarray, vision: np.ndarray
) -> dict[str, np.ndarray]:
    """Separate content extent, observed values, and padding.

    The union of three modalities estimates the final active index.  Position 0
    ([CLS]) and that final position ([SEP]) are structural boundaries rather
    than content.  A zero row inside the resulting content interval is recorded
    as unobserved/suspicious but is not claimed to have a known physical cause.
    """

    text_bert = np.asarray(text_bert)
    audio = np.asarray(audio)
    vision = np.asarray(vision)
    if text_bert.ndim == 2:
        text_bert = text_bert[None, ...]
        audio = audio[None, ...]
        vision = vision[None, ...]
    text_observed_raw = text_bert[:, 1, :] > 0
    audio_observed_raw = np.any(audio != 0, axis=-1)
    vision_observed_raw = np.any(vision != 0, axis=-1)
    union = text_observed_raw | audio_observed_raw | vision_observed_raw
    positions = np.arange(union.shape[1])[None, :]
    last_active = np.max(np.where(union, positions, -1), axis=1)
    if np.any(last_active < 2):
        bad = np.flatnonzero(last_active < 2).tolist()
        raise ValueError(f"Samples without a valid content interval: {bad}")
    timeline = (positions > 0) & (positions < last_active[:, None])
    return {
        "timeline": timeline,
        "text_observed": text_observed_raw & timeline,
        "audio_observed": audio_observed_raw & timeline,
        "vision_observed": vision_observed_raw & timeline,
        "last_active": last_active,
    }


@dataclass
class FeatureScaler:
    audio_mean: list[float]
    audio_std: list[float]
    vision_mean: list[float]
    vision_std: list[float]

    @classmethod
    def fit(
        cls,
        audio: np.ndarray,
        vision: np.ndarray,
        masks: dict[str, np.ndarray],
    ) -> "FeatureScaler":
        audio_values = np.asarray(audio, dtype=np.float64)[masks["audio_observed"]]
        vision_values = np.asarray(vision, dtype=np.float64)[masks["vision_observed"]]
        audio_mean = audio_values.mean(axis=0)
        audio_std = audio_values.std(axis=0)
        vision_mean = vision_values.mean(axis=0)
        vision_std = vision_values.std(axis=0)
        audio_std = np.where(audio_std < 1e-6, 1.0, audio_std)
        vision_std = np.where(vision_std < 1e-6, 1.0, vision_std)
        return cls(
            audio_mean=audio_mean.tolist(),
            audio_std=audio_std.tolist(),
            vision_mean=vision_mean.tolist(),
            vision_std=vision_std.tolist(),
        )

    def transform_audio(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.audio_mean, dtype=np.float32)
        std = np.asarray(self.audio_std, dtype=np.float32)
        return np.clip((np.asarray(values, dtype=np.float32) - mean) / std, -8, 8)

    def transform_vision(self, values: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.vision_mean, dtype=np.float32)
        std = np.asarray(self.vision_std, dtype=np.float32)
        return np.clip((np.asarray(values, dtype=np.float32) - mean) / std, -8, 8)

    def save(self, path: Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        return cls(**read_json(path))


def regression_to_class(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.where(values < 0, 0, np.where(values > 0, 2, 1)).astype(np.int64)


def project_intensity_to_class(
    values: np.ndarray, predicted_class: np.ndarray, epsilon: float = 1e-3
) -> np.ndarray:
    """Project intensities onto strict class-consistent sign intervals.

    The task defines negative, neutral and positive as y<0, y=0 and y>0.
    A small epsilon therefore keeps non-neutral predictions away from the
    otherwise ambiguous zero boundary while changing magnitudes minimally.
    """

    values = np.asarray(values, dtype=np.float32)
    predicted_class = np.asarray(predicted_class, dtype=np.int64)
    return np.where(
        predicted_class == 0,
        np.minimum(values, -epsilon),
        np.where(predicted_class == 2, np.maximum(values, epsilon), 0.0),
    ).astype(np.float32)


def contiguous_runs(mask: Iterable[bool]) -> list[tuple[int, int]]:
    values = np.asarray(list(mask), dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]
