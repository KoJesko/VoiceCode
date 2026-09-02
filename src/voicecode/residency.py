"""Keep the ASR and TTS models resident only while they are being used.

Both models are heavy and idle most of the time: a bot sitting in a voice channel
overnight holds well over a gigabyte of VRAM for nothing. `ManagedModel` loads a model
on first use and unloads it after a configurable idle period, so the GPU is free
between conversations and the models come back on the next thing anyone says.

Three things make this safe rather than merely convenient:

**A model is never unloaded mid-inference.** Every call takes a lease. The janitor
unloads only when the lease count is zero, under the same lock that grants leases, so
an unload can never land between "check whether it is loaded" and "call it".

**Availability is not residency.** `ManagedTTS.enabled` means "could speak", not "is on
the GPU right now". The session checks it before every sentence; if it meant residency,
the first unload would silently turn the bot text-only for the rest of its life. Only a
real failure -- an OOM, a missing package -- disables TTS, exactly as before.

**Reloading is visible.** Every load and unload logs, and `describe()` reports
residency, so a `/status` during a cold period reads "unloaded" rather than "broken".

The cost is latency. The first turn after an unload pays the full load plus warmup --
several seconds for NeMo, well outside the 1.5 s end-of-speech budget. That is the
trade being made, and it is why MODEL_IDLE_UNLOAD_MINUTES defaults to 0, meaning the
old always-resident behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager

import numpy as np

from .asr.base import ASREngine, Transcript
from .tts.kokoro_engine import KokoroTTS, Speech, TTSUnavailable

log = logging.getLogger(__name__)

JANITOR_INTERVAL_S = 30.0


class ManagedModel:
    """Load-on-demand, unload-when-idle bookkeeping for one heavy model.

    Deliberately knows nothing about what it is managing: it holds a load callable, an
    unload callable, a lease count and an idle clock. The adapters below supply the
    rest.
    """

    def __init__(self, *, name: str, load: Callable[[], None], unload: Callable[[], None]):
        self.name = name
        self._load_fn = load
        self._unload_fn = unload
        # Guards every field below. Held across load and unload -- both are slow, but
        # holding it is the point: it is what stops a lease starting mid-unload.
        self._lock = threading.Lock()
        self._loaded = False
        self._leases = 0
        self._idle_since: float | None = None  # monotonic; None while in use
        self.loads = 0
        self.unloads = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    @contextmanager
    def lease(self):
        """Hold the model resident for the duration of one call.

        Blocking, and called from worker threads. Loads first if it has to, which is
        why a cold turn is slow.
        """
        self._acquire()
        try:
            yield
        finally:
            self._release()

    def _acquire(self) -> None:
        with self._lock:
            if not self._loaded:
                started = time.perf_counter()
                self._load_fn()
                self._loaded = True
                self.loads += 1
                log.info(
                    "%s loaded on demand in %.1fs (load #%d)",
                    self.name,
                    time.perf_counter() - started,
                    self.loads,
                )
            self._leases += 1
            self._idle_since = None

    def _release(self) -> None:
        with self._lock:
            self._leases = max(0, self._leases - 1)
            if self._leases == 0:
                self._idle_since = time.monotonic()

    def preload(self) -> None:
        """Load now and start the idle clock. Propagates load failures to the caller."""
        with self.lease():
            pass

    def mark_unloaded(self) -> None:
        """Record that the model went away behind our back (e.g. TTS disabled itself)."""
        with self._lock:
            self._loaded = False
            self._idle_since = None

    def unload_now(self) -> bool:
        """Unload regardless of the idle clock. Refuses while a lease is held."""
        with self._lock:
            return self._unload_locked(reason="on request")

    def maybe_unload(self, idle_timeout_s: float) -> bool:
        """Unload if loaded, unused, and idle for longer than the timeout."""
        if idle_timeout_s <= 0:
            return False
        with self._lock:
            if self._idle_since is None:
                return False
            idle_for = time.monotonic() - self._idle_since
            if idle_for < idle_timeout_s:
                return False
            return self._unload_locked(reason=f"after {idle_for / 60:.0f} min idle")

    def _unload_locked(self, *, reason: str) -> bool:
        if not self._loaded or self._leases:
            return False
        self._unload_fn()
        self._loaded = False
        self._idle_since = None
        self.unloads += 1
        log.info("%s unloaded %s, freeing its memory", self.name, reason)
        return True

    def residency(self) -> str:
        """One phrase for /status. Says what is true now, not what is configured."""
        with self._lock:
            if not self._loaded:
                return "unloaded" if self.unloads else "not loaded"
            if self._leases:
                return "in use"
            if self._idle_since is None:
                return "resident"
            return f"resident, idle {time.monotonic() - self._idle_since:.0f}s"


class ManagedASR:
    """An ASREngine that loads on demand and survives being unloaded between turns."""

    def __init__(self, engine: ASREngine):
        self.engine = engine
        self.model = ManagedModel(name="ASR", load=engine.load, unload=engine.unload)

    @property
    def name(self) -> str:
        return self.engine.name

    def load(self) -> None:
        self.model.preload()

    def unload(self) -> None:
        self.model.unload_now()

    def transcribe(self, audio: np.ndarray) -> Transcript:
        with self.model.lease():
            return self.engine.transcribe(audio)

    def describe(self) -> str:
        return f"{self.engine.describe()} | {self.model.residency()}"


class ManagedTTS:
    """A KokoroTTS that loads on demand and survives being unloaded between turns."""

    def __init__(self, engine: KokoroTTS):
        self.engine = engine
        self.model = ManagedModel(name="TTS", load=engine.load, unload=engine.unload)

    @property
    def enabled(self) -> bool:
        # "Could speak", not "is on the GPU". The session checks this before every
        # sentence; see the module docstring for why residency must not leak in here.
        return self.engine.disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self.engine.disabled_reason

    @property
    def voice(self) -> str:
        return self.engine.voice

    def load(self) -> None:
        self.model.preload()

    def unload(self) -> None:
        self.model.unload_now()

    def set_voice(self, voice: str) -> None:
        self.engine.set_voice(voice)

    def disable(self, reason: str) -> None:
        self.engine.disable(reason)
        self.model.mark_unloaded()

    def synthesize(self, text: str) -> Speech:
        reason = self.engine.disabled_reason
        if reason:
            # Fail here rather than leasing: loading a disabled engine is a no-op that
            # would leave the bookkeeping claiming a model that is not there.
            raise TTSUnavailable(reason)
        with self.model.lease():
            try:
                return self.engine.synthesize(text)
            finally:
                # An OOM inside synthesize() disables the engine and drops the
                # pipeline. Resync rather than believing our own stale flag.
                if self.engine.disabled_reason is not None:
                    self.model.mark_unloaded()

    def describe(self) -> str:
        if self.engine.disabled_reason:
            return self.engine.describe()
        return f"{self.engine.describe()} | {self.model.residency()}"


async def run_janitor(
    models: Sequence[ManagedModel],
    idle_minutes: Callable[[], int],
    *,
    interval_s: float = JANITOR_INTERVAL_S,
) -> None:
    """Unload idle models on a timer.

    Polled rather than a sleep-until-deadline so the threshold stays live-reloadable
    through /reload, and so a model that is picked up again simply resets its clock.
    Unloading runs on a worker thread: freeing CUDA memory is slow enough to be worth
    keeping off the event loop that is also decoding audio.
    """
    try:
        while True:
            await asyncio.sleep(interval_s)
            timeout_s = idle_minutes() * 60
            if timeout_s <= 0:
                continue
            for model in models:
                try:
                    await asyncio.to_thread(model.maybe_unload, timeout_s)
                except Exception:
                    log.exception("failed to unload %s; leaving it resident", model.name)
    except asyncio.CancelledError:
        pass
