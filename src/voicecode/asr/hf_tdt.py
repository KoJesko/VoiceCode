"""Parakeet via Transformers. The alternative backend.

Model: nvidia/parakeet-tdt-0.6b-v3, which the Hub now serves with
`library_name: transformers` through AutoModelForTDT. Far lower install friction than
NeMo (2.4M downloads vs 949 for the unified model) and CC-BY-4.0 rather than the
NVIDIA Open Model License.

Caveat from the model card, current as of 2026-08-29: *"Until Parakeet TDT is part of
an official Transformers release, you can use it by installing from source."* If
AutoModelForTDT is missing, that is the reason, and the error below says so.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .base import ASR_SAMPLE_RATE, ASRUnavailable, Transcript

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class HFParakeetASR:
    name = "transformers"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        # See nemo_unified: inference runs on worker threads and must be
        # serialised per model object.
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ASRUnavailable(
                "transformers is not installed. Install the extra:\n"
                "  uv pip install -e '.[hf]'\n"
                f"(original error: {exc})"
            ) from exc

        try:
            from transformers import AutoModelForTDT
        except ImportError as exc:
            raise ASRUnavailable(
                "transformers has no AutoModelForTDT. Parakeet TDT is not in an "
                "official release yet; install from source:\n"
                "  uv pip install 'git+https://github.com/huggingface/transformers'"
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; ASR will run on CPU")
            self.device = "cpu"

        log.info("loading ASR model %s", self.model_id)
        started = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForTDT.from_pretrained(
            self.model_id, dtype="auto", device_map=self.device
        )
        log.info("ASR model loaded in %.1fs on %s", time.perf_counter() - started, self.device)
        self._warmup()

    def _warmup(self) -> None:
        try:
            self.transcribe(np.zeros(ASR_SAMPLE_RATE, dtype=np.float32))
        except Exception:
            log.exception("ASR warmup failed; the first real turn may be slow")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def describe(self) -> str:
        # See nemo_unified.describe: residency is reported by ManagedASR, not here.
        return f"transformers {self.model_id} on {self.device}"

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._model is None or self._processor is None:
            raise ASRUnavailable("ASR model is not loaded")

        import torch

        started = time.perf_counter()
        rate = getattr(
            getattr(self._processor, "feature_extractor", None),
            "sampling_rate",
            ASR_SAMPLE_RATE,
        )
        with self._lock:
            inputs = self._processor(
                [np.asarray(audio, dtype=np.float32)], sampling_rate=rate
            )
            inputs = inputs.to(self._model.device, dtype=self._model.dtype)
            with torch.no_grad():
                output = self._model.generate(**inputs, return_dict_in_generate=True)
            decoded = self._processor.decode(output.sequences, skip_special_tokens=True)

        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0] if decoded else ""
        return Transcript(
            text=str(decoded).strip(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            backend=self.name,
        )
