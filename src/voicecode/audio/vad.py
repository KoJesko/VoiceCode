"""Voice activity detection with silero-vad.

silero's 16 kHz model takes a fixed 512-sample window (32 ms) -- that is a property of
the model, not a tunable. Discord frames arrive as 20 ms, which resamples to 320
samples at 16 kHz, so windows never line up with frames. This class absorbs that
mismatch: feed it whatever arrives, get back one probability per completed window.

Discord's own speaking-state events are used elsewhere as a coarse gate, but they are
driven by a 200 ms timeout in the receive extension and by a client-controlled flag, so
they cannot endpoint a turn. This can.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

WINDOW_SAMPLES = 512
SAMPLE_RATE = 16_000


class SileroVad:
    """Thin stateful wrapper. Loads the model once; instances are per-user."""

    _model = None  # class-level: the model is shared, the RNN state is not

    __slots__ = ("_buffer", "_threshold", "_torch")

    def __init__(self, threshold: float = 0.5):
        self._threshold = threshold
        self._buffer = np.zeros(0, dtype=np.float32)
        self._torch = None

    @classmethod
    def load(cls) -> None:
        """Load the shared model. Safe to call more than once."""
        if cls._model is not None:
            return
        from silero_vad import load_silero_vad

        cls._model = load_silero_vad()
        log.info("silero-vad loaded")

    def _ensure(self):
        if SileroVad._model is None:
            SileroVad.load()
        if self._torch is None:
            import torch

            self._torch = torch
        return SileroVad._model

    def reset(self) -> None:
        """Clear RNN state between utterances. Required, or confidence drifts."""
        self._buffer = np.zeros(0, dtype=np.float32)
        model = SileroVad._model
        if model is not None and hasattr(model, "reset_states"):
            model.reset_states()

    def feed(self, audio: np.ndarray) -> list[float]:
        """Feed 16 kHz mono float32. Returns a probability per completed window."""
        if audio.size:
            self._buffer = np.concatenate([self._buffer, audio.astype(np.float32)])

        if self._buffer.size < WINDOW_SAMPLES:
            return []

        model = self._ensure()
        torch = self._torch
        probabilities: list[float] = []
        offset = 0
        with torch.no_grad():
            while offset + WINDOW_SAMPLES <= self._buffer.size:
                window = self._buffer[offset : offset + WINDOW_SAMPLES]
                tensor = torch.from_numpy(np.ascontiguousarray(window))
                probabilities.append(float(model(tensor, SAMPLE_RATE).item()))
                offset += WINDOW_SAMPLES
        self._buffer = self._buffer[offset:]
        return probabilities

    def is_speech(self, probability: float) -> bool:
        return probability >= self._threshold


class EnergyVad:
    """Fallback used when silero cannot be loaded.

    Deliberately crude. It exists so a missing model degrades endpointing quality
    instead of taking the voice connection down, per the graceful-degradation
    requirement -- not because RMS gating is good enough on its own.
    """

    __slots__ = ("_buffer", "_threshold")

    def __init__(self, threshold: float = 0.01):
        self._threshold = threshold
        self._buffer = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    def feed(self, audio: np.ndarray) -> list[float]:
        if audio.size:
            self._buffer = np.concatenate([self._buffer, audio.astype(np.float32)])
        out: list[float] = []
        offset = 0
        while offset + WINDOW_SAMPLES <= self._buffer.size:
            window = self._buffer[offset : offset + WINDOW_SAMPLES]
            rms = float(np.sqrt(np.mean(window**2)))
            out.append(min(1.0, rms / max(self._threshold, 1e-6)))
            offset += WINDOW_SAMPLES
        self._buffer = self._buffer[offset:]
        return out

    def is_speech(self, probability: float) -> bool:
        return probability >= 1.0
