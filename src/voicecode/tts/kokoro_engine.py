"""Kokoro-82M text-to-speech.

The pipeline is built once and shared. Kokoro emits 24 kHz mono float32; conversion to
Discord's 48 kHz stereo s16le happens in audio/resample.py.

Graceful degradation is a requirement here, not a nicety: if the GPU runs out of memory
the bot disables TTS and keeps the voice connection and the text mirror alive rather
than taking the whole session down. ASR is the more important of the two models -- a
bot that hears you and answers only in text is degraded; one that has dropped the voice
connection is off.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

KOKORO_RATE = 24_000
DEFAULT_VOICE = "af_heart"
DEFAULT_LANG = "a"  # American English


class TTSUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Speech:
    audio: np.ndarray  # 24 kHz mono float32
    duration_ms: float
    synth_ms: float


class KokoroTTS:
    def __init__(
        self,
        lang_code: str = DEFAULT_LANG,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
    ):
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        self._disabled_reason: str | None = None

    # -- lifecycle ---------------------------------------------------------------

    def load(self) -> None:
        if self._pipeline is not None or self._disabled_reason:
            return
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise TTSUnavailable(
                f"the 'kokoro' package is not installed ({exc}). "
                "Kokoro also needs espeak-ng on the system for out-of-dictionary words."
            ) from exc

        log.info("loading Kokoro (lang_code=%r, voice=%r)", self.lang_code, self.voice)
        started = time.perf_counter()
        self._pipeline = KPipeline(lang_code=self.lang_code)
        log.info("Kokoro loaded in %.1fs", time.perf_counter() - started)
        self._warmup()

    def _warmup(self) -> None:
        started = time.perf_counter()
        try:
            self.synthesize("Ready.")
        except Exception:
            log.exception("Kokoro warmup failed; the first spoken turn may be slow")
        else:
            log.info("Kokoro warmup complete in %.0fms", (time.perf_counter() - started) * 1000)

    @property
    def enabled(self) -> bool:
        return self._pipeline is not None and self._disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def describe(self) -> str:
        if self._disabled_reason:
            return f"Kokoro disabled: {self._disabled_reason}"
        if self._pipeline is None:
            return "Kokoro not loaded"
        return f"Kokoro voice={self.voice} lang={self.lang_code} speed={self.speed:g}"

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    def disable(self, reason: str) -> None:
        log.error("disabling TTS: %s", reason)
        self._disabled_reason = reason
        self._pipeline = None
        _empty_cuda_cache()

    # -- inference ---------------------------------------------------------------

    def synthesize(self, text: str) -> Speech:
        """Synthesize one sentence. Returns 24 kHz mono float32."""
        if self._disabled_reason:
            raise TTSUnavailable(self._disabled_reason)
        if self._pipeline is None:
            raise TTSUnavailable("Kokoro is not loaded")
        if not text.strip():
            return Speech(np.zeros(0, dtype=np.float32), 0.0, 0.0)

        started = time.perf_counter()
        try:
            chunks = [
                _to_numpy(audio)
                for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed)
            ]
        except Exception as exc:
            if _is_oom(exc):
                self.disable(f"GPU out of memory during synthesis ({exc})")
                raise TTSUnavailable(self._disabled_reason or "out of memory") from exc
            raise

        chunks = [c for c in chunks if c.size]
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return Speech(
            audio=audio,
            duration_ms=audio.size / KOKORO_RATE * 1000.0,
            synth_ms=(time.perf_counter() - started) * 1000.0,
        )


def _to_numpy(audio) -> np.ndarray:
    """Kokoro yields a torch tensor on GPU builds and a numpy array on others."""
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32, copy=False)
    detach = getattr(audio, "detach", None)
    if detach is not None:
        return detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(audio, dtype=np.float32)


def _is_oom(exc: BaseException) -> bool:
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
