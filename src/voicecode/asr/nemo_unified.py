"""Parakeet via NVIDIA NeMo. The default backend.

Model: nvidia/parakeet-unified-en-0.6b (offline+streaming in one checkpoint; we use it
offline). Loaded once at startup, warmed with a dummy tensor so the first real
utterance does not pay CUDA kernel autotuning, and kept resident on the GPU.

NeMo's `transcribe()` is documented against file paths. Recent versions also accept
in-memory arrays, which avoids a disk round-trip per turn -- worth roughly the write
plus the read on the critical path. We try the array form once and fall back to a temp
wav permanently if this NeMo build rejects it, rather than probing every turn.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .base import ASR_SAMPLE_RATE, ASRUnavailable, Transcript

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/parakeet-unified-en-0.6b"


class NemoUnifiedASR:
    name = "nemo"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._accepts_arrays: bool | None = None
        # Inference runs on a worker thread (asyncio.to_thread), and two people
        # talking at once means two concurrent calls into the same model object.
        # NeMo models are not thread-safe, so serialise here rather than relying
        # on every caller to remember.
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ASRUnavailable(
                "NeMo is not installed. Install the ASR extra:\n"
                "  uv pip install -e '.[nemo]'\n"
                f"(original error: {exc})"
            ) from exc

        log.info("loading ASR model %s (this downloads ~2.5GB on first run)", self.model_id)
        started = time.perf_counter()
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)

        try:
            import torch

            if self.device.startswith("cuda") and torch.cuda.is_available():
                model = model.to(self.device)
            elif self.device.startswith("cuda"):
                log.warning("CUDA requested but unavailable; ASR will run on CPU")
                self.device = "cpu"
        except ImportError:  # pragma: no cover
            pass

        model.eval()
        self._model = model
        log.info("ASR model loaded in %.1fs on %s", time.perf_counter() - started, self.device)
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy utterance so the first real turn is not the slow one."""
        dummy = np.zeros(ASR_SAMPLE_RATE, dtype=np.float32)
        started = time.perf_counter()
        try:
            self.transcribe(dummy)
        except Exception:
            log.exception("ASR warmup failed; the first real turn may be slow")
        else:
            log.info("ASR warmup complete in %.0fms", (time.perf_counter() - started) * 1000)

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def describe(self) -> str:
        state = "loaded" if self._model is not None else "not loaded"
        return f"NeMo {self.model_id} on {self.device} ({state})"

    # -- inference ---------------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._model is None:
            raise ASRUnavailable("ASR model is not loaded")

        started = time.perf_counter()
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        with self._lock:
            outputs = self._run(audio)

        return Transcript(
            text=_extract_text(outputs),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            backend=self.name,
        )

    def _run(self, audio: np.ndarray):
        if self._accepts_arrays is not False:
            try:
                outputs = self._model.transcribe([audio], batch_size=1, verbose=False)
                self._accepts_arrays = True
                return outputs
            except Exception as exc:
                if self._accepts_arrays is True:
                    raise
                log.info(
                    "this NeMo build does not accept in-memory audio (%s); "
                    "using a temp wav per turn",
                    type(exc).__name__,
                )
                self._accepts_arrays = False
                return self._transcribe_via_file(audio)
        return self._transcribe_via_file(audio)

    def _transcribe_via_file(self, audio: np.ndarray):
        with tempfile.TemporaryDirectory(prefix="voicecode-asr-") as tmp:
            path = Path(tmp) / "utterance.wav"
            _write_wav(path, audio)
            return self._model.transcribe([str(path)], batch_size=1, verbose=False)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    ints = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(ASR_SAMPLE_RATE)
        handle.writeframes(ints.tobytes())


def _extract_text(outputs) -> str:
    """NeMo returns Hypothesis objects on modern versions and plain strings on older."""
    if not outputs:
        return ""
    first = outputs[0]
    if isinstance(first, str):
        return first.strip()
    text = getattr(first, "text", None)
    if isinstance(text, str):
        return text.strip()
    if isinstance(first, (list, tuple)) and first:
        return _extract_text(first)
    return str(first).strip()
