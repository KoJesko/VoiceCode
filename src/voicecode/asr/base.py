"""ASR backend interface.

Transcription happens once, on a complete utterance, after the VAD has endpointed the
turn. It is not streamed. See docs/DESIGN.md 1.5: parakeet-unified's streaming mode
exposes no feed-a-chunk API and recomputes 5.6s of left context per chunk, and since
the latency budget is measured from end-of-speech, offline transcription starts exactly
when the clock does -- while scoring better (5.91 vs 6.29 WER).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    duration_ms: float
    backend: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@runtime_checkable
class ASREngine(Protocol):
    name: str

    def load(self) -> None:
        """Load weights and warm up. Called once at startup, never in the hot path."""

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe 16 kHz mono float32 audio."""

    def describe(self) -> str: ...

    def unload(self) -> None: ...


class ASRUnavailable(RuntimeError):
    pass


def build_engine(backend: str, model_id: str, device: str = "cuda") -> ASREngine:
    """Construct the configured backend without importing the other one's deps."""
    if backend == "nemo_unified":
        from .nemo_unified import NemoUnifiedASR

        return NemoUnifiedASR(model_id=model_id, device=device)
    if backend == "hf_tdt":
        from .hf_tdt import HFParakeetASR

        return HFParakeetASR(model_id=model_id, device=device)
    raise ASRUnavailable(f"unknown ASR backend {backend!r}")
