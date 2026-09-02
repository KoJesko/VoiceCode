"""Load-on-demand / unload-when-idle behaviour.

The expensive bugs here are not "it failed to unload". They are unloading a model out
from under a call that is using it, and letting a routine unload look like the
permanent TTS-disabled state -- which would silently make the bot text-only forever.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from voicecode.asr.base import Transcript
from voicecode.residency import ManagedASR, ManagedModel, ManagedTTS, run_janitor
from voicecode.tts.kokoro_engine import KokoroTTS, Speech, TTSUnavailable


class FakeEngine:
    """Stands in for either heavy model: counts loads, unloads, and concurrent calls."""

    name = "fake"

    def __init__(self, *, load_error: Exception | None = None):
        self.loaded = False
        self.loads = 0
        self.unloads = 0
        self.calls = 0
        self.load_error = load_error
        self.loaded_during_calls: list[bool] = []

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True
        self.loads += 1

    def unload(self) -> None:
        self.loaded = False
        self.unloads += 1

    def transcribe(self, audio) -> Transcript:
        self.calls += 1
        self.loaded_during_calls.append(self.loaded)
        return Transcript("hello", 1.0, self.name)

    def describe(self) -> str:
        return "fake engine"


def managed(engine: FakeEngine) -> ManagedModel:
    return ManagedModel(name="fake", load=engine.load, unload=engine.unload)


def make_idle(model: ManagedModel, seconds: float = 3600.0) -> None:
    """Backdate the idle clock instead of sleeping for the threshold.

    The alternative is a real sleep long enough to beat a real timeout, which either
    makes the suite slow or makes it flaky when the machine is loaded.
    """
    model._idle_since = time.monotonic() - seconds


# -- load on demand -----------------------------------------------------------

def test_first_use_loads_and_no_use_does_not():
    engine = FakeEngine()
    asr = ManagedASR(engine)
    assert engine.loads == 0

    asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert engine.loads == 1
    assert engine.loaded_during_calls == [True]


def test_repeated_use_loads_once():
    engine = FakeEngine()
    asr = ManagedASR(engine)
    for _ in range(3):
        asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert engine.loads == 1


def test_use_after_unload_reloads():
    engine = FakeEngine()
    asr = ManagedASR(engine)
    asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert asr.model.unload_now()
    assert not engine.loaded

    asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert engine.loads == 2
    assert engine.loaded_during_calls == [True, True]


def test_load_failure_propagates_and_leaves_nothing_leased():
    engine = FakeEngine(load_error=RuntimeError("no GPU"))
    model = managed(engine)
    with pytest.raises(RuntimeError, match="no GPU"):
        model.preload()
    assert not model.loaded
    # A failed acquire must not leave a lease behind, or the model can never unload.
    assert model.maybe_unload(0.0) is False
    engine.load_error = None
    model.preload()
    assert model.loaded


# -- unload only when genuinely idle -------------------------------------------

def test_unload_waits_for_the_idle_threshold():
    engine = FakeEngine()
    model = managed(engine)
    model.preload()

    assert model.maybe_unload(idle_timeout_s=60) is False
    assert engine.unloads == 0

    make_idle(model, seconds=120)
    assert model.maybe_unload(idle_timeout_s=60) is True
    assert engine.unloads == 1


def test_zero_timeout_never_unloads():
    """0 is the documented "stay resident" default, not "unload immediately"."""
    engine = FakeEngine()
    model = managed(engine)
    model.preload()
    make_idle(model)
    assert model.maybe_unload(idle_timeout_s=0) is False
    assert engine.loaded


def test_unloading_an_unloaded_model_is_a_no_op():
    engine = FakeEngine()
    model = managed(engine)
    assert model.unload_now() is False
    assert engine.unloads == 0


def test_idle_clock_resets_on_use():
    engine = FakeEngine()
    model = managed(engine)
    model.preload()
    time.sleep(0.02)
    with model.lease():
        pass
    # The clock restarted at the lease release, so a threshold longer than the time
    # since then must not fire even though the model is older than it.
    assert model.maybe_unload(idle_timeout_s=0.01) is False


# -- never unload mid-inference ------------------------------------------------

def test_unload_refuses_while_a_lease_is_held():
    engine = FakeEngine()
    model = managed(engine)
    with model.lease():
        make_idle(model)  # even a stale clock must not beat a live lease
        assert model.maybe_unload(idle_timeout_s=60) is False
        assert model.unload_now() is False
        assert engine.loaded
    make_idle(model)
    assert model.maybe_unload(idle_timeout_s=60) is True


def test_concurrent_calls_never_see_an_unloaded_model():
    """The race the lease exists to prevent, run for real on threads."""
    engine = FakeEngine()
    asr = ManagedASR(engine)
    stop = threading.Event()
    errors: list[str] = []

    def call() -> None:
        while not stop.is_set():
            asr.transcribe(np.zeros(160, dtype=np.float32))

    def evict() -> None:
        while not stop.is_set():
            # Any gap at all counts as idle, so eviction hammers the lease boundary.
            asr.model.maybe_unload(idle_timeout_s=1e-9)

    threads = [threading.Thread(target=call) for _ in range(4)]
    threads.append(threading.Thread(target=evict))
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join()

    if not all(engine.loaded_during_calls):
        errors.append("a transcribe() ran against an unloaded model")
    assert not errors, errors
    assert engine.calls > 0
    # The point of the test is the interleaving; if nothing was evicted it proved
    # nothing, so make that visible rather than passing quietly.
    assert engine.unloads > 0, "no unload raced a call; the test proved nothing"


# -- TTS: unloaded is not disabled ---------------------------------------------

class FakeKokoro(KokoroTTS):
    """Real KokoroTTS with the model swapped for a stub, so disable/unload are real."""

    def __init__(self, *, oom_on_synth: bool = False):
        super().__init__()
        self.oom_on_synth = oom_on_synth
        self.synth_calls = 0

    def load(self) -> None:
        if self._disabled_reason:
            return
        self._pipeline = object()

    def synthesize(self, text: str) -> Speech:
        if self._disabled_reason:
            raise TTSUnavailable(self._disabled_reason)
        if self._pipeline is None:
            raise TTSUnavailable("Kokoro is not loaded")
        self.synth_calls += 1
        if self.oom_on_synth:
            self.disable("GPU out of memory during synthesis (fake)")
            raise TTSUnavailable(self._disabled_reason or "out of memory")
        return Speech(np.zeros(2400, dtype=np.float32), 100.0, 1.0)


def test_tts_stays_enabled_across_an_unload():
    """The regression this guards: session.py checks .enabled before every sentence."""
    engine = FakeKokoro()
    tts = ManagedTTS(engine)
    tts.synthesize("one")
    assert tts.model.unload_now()

    assert tts.enabled, "an unloaded TTS must still report itself as able to speak"
    tts.synthesize("two")
    assert engine.synth_calls == 2


def test_tts_disabled_by_oom_stays_disabled():
    engine = FakeKokoro(oom_on_synth=True)
    tts = ManagedTTS(engine)
    with pytest.raises(TTSUnavailable):
        tts.synthesize("one")

    assert not tts.enabled
    assert not tts.model.loaded, "bookkeeping must follow the engine dropping its pipeline"
    with pytest.raises(TTSUnavailable):
        tts.synthesize("two")
    assert engine.synth_calls == 1, "a disabled engine must not be reloaded and retried"


def test_disabling_through_the_wrapper_clears_residency():
    engine = FakeKokoro()
    tts = ManagedTTS(engine)
    tts.load()
    tts.disable("no espeak-ng")
    assert not tts.enabled
    assert not tts.model.loaded
    assert tts.disabled_reason == "no espeak-ng"


def test_set_voice_reaches_the_engine_while_unloaded():
    engine = FakeKokoro()
    tts = ManagedTTS(engine)
    tts.set_voice("af_bella")
    assert engine.voice == "af_bella"
    assert tts.voice == "af_bella"


# -- describe ------------------------------------------------------------------

def test_describe_reports_residency():
    engine = FakeEngine()
    asr = ManagedASR(engine)
    assert "not loaded" in asr.describe()
    asr.load()
    assert "resident" in asr.describe()
    asr.unload()
    assert "unloaded" in asr.describe()


def test_disabled_tts_describes_the_reason_not_its_residency():
    engine = FakeKokoro()
    tts = ManagedTTS(engine)
    tts.disable("no espeak-ng")
    assert tts.describe() == "Kokoro disabled: no espeak-ng"


# -- janitor -------------------------------------------------------------------

async def test_janitor_unloads_idle_models():
    engine = FakeEngine()
    model = managed(engine)
    model.preload()

    make_idle(model)
    task = asyncio.create_task(run_janitor([model], lambda: 1, interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    await task

    assert engine.unloads > 0


async def test_janitor_leaves_models_alone_when_disabled():
    engine = FakeEngine()
    model = managed(engine)
    model.preload()
    make_idle(model)

    task = asyncio.create_task(run_janitor([model], lambda: 0, interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    await task

    assert engine.unloads == 0
    assert engine.loaded


async def test_janitor_reads_the_threshold_live():
    """/reload retunes the janitor without a restart."""
    engine = FakeEngine()
    model = managed(engine)
    model.preload()
    make_idle(model)
    minutes = 0

    task = asyncio.create_task(run_janitor([model], lambda: minutes, interval_s=0.01))
    await asyncio.sleep(0.05)
    assert engine.unloads == 0

    minutes = 1
    await asyncio.sleep(0.05)
    task.cancel()
    await task

    assert engine.unloads > 0


async def test_janitor_survives_a_failing_unload():
    """A model that cannot be freed must not kill the loop that frees the other one."""
    class Angry(FakeEngine):
        def unload(self) -> None:
            raise RuntimeError("cuda is unhappy")

    angry, calm = Angry(), FakeEngine()
    models = [managed(angry), managed(calm)]
    for model in models:
        model.preload()
        make_idle(model)

    task = asyncio.create_task(run_janitor(models, lambda: 1, interval_s=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    await task

    assert calm.unloads > 0
