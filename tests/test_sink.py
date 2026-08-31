"""The sink's gates and its per-user state lifecycle.

The sink is the only place that touches raw Discord audio, and it runs entirely
off the event loop, so a mistake here is invisible in normal use: dropped frames
look like a quiet channel and an exception in the reader thread is swallowed by
the extension. Gate #7 (the user allowlist) and the idle-buffer reclamation both
sit on that path.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from voicecode.audio import sink as sink_mod
from voicecode.audio.sink import VoiceCodeSink
from voicecode.config import AllowlistSnapshot

GUILD, VOICE, TEXT, ALLOWED, DENIED, OTHER = 1, 2, 5, 3, 4, 7


def two_speaker_snapshot() -> AllowlistSnapshot:
    """Two allowed speakers, for the tests that need two live buffers."""
    return AllowlistSnapshot(
        frozenset({GUILD}),
        frozenset({VOICE}),
        frozenset({ALLOWED, OTHER}),
        {VOICE: TEXT},
        False,
        0,
    )


def snapshot() -> AllowlistSnapshot:
    return AllowlistSnapshot(
        frozenset({GUILD}), frozenset({VOICE}), frozenset({ALLOWED}), {VOICE: TEXT}, False, 0
    )


class FakeDecryptor:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.forgotten: list[int] = []

    def decrypt(self, user_id, payload):
        self.calls.append(user_id)
        return payload

    def forget(self, user_id):
        self.forgotten.append(user_id)


class FakeDecoders:
    """Stands in for the Opus pool: real decoding needs libopus and a real frame."""

    def __init__(self) -> None:
        self.dropped: list[int] = []
        # 20 ms of 48 kHz stereo silence, which is what a real decode returns.
        self.pcm = b"\x00" * 3840

    def decode(self, ssrc, payload):
        return self.pcm

    def drop(self, ssrc):
        self.dropped.append(ssrc)

    def clear(self):
        pass


class FakeConsumer:
    def __init__(self) -> None:
        self.speech_starts: list[int] = []
        self.utterances: list = []

    async def on_speech_start(self, user_id):
        self.speech_starts.append(user_id)

    async def on_utterance(self, event):
        self.utterances.append(event)


def make_sink(loop) -> tuple[VoiceCodeSink, FakeDecryptor, FakeDecoders, FakeConsumer]:
    decryptor, consumer = FakeDecryptor(), FakeConsumer()
    sink = VoiceCodeSink(
        loop=loop,
        snapshot_provider=snapshot,
        decryptor=decryptor,
        consumer=consumer,
        use_silero=False,
    )
    decoders = FakeDecoders()
    sink._decoders = decoders
    return sink, decryptor, decoders, consumer


def voice_data(payload=b"\x01\x02\x03", ssrc=99):
    return SimpleNamespace(opus=payload, pcm=None, packet=SimpleNamespace(ssrc=ssrc), source=None)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# -- gate #7 ------------------------------------------------------------------


def test_denied_user_audio_never_reaches_the_decryptor(loop):
    sink, decryptor, _, _ = make_sink(loop)
    sink.write(SimpleNamespace(id=DENIED), voice_data())
    assert decryptor.calls == []
    assert sink._buffers == {}


def test_unknown_speaker_is_dropped(loop):
    sink, decryptor, _, _ = make_sink(loop)
    sink.write(None, voice_data())
    assert decryptor.calls == []


def test_muted_sink_drops_allowed_audio_too(loop):
    sink, decryptor, _, _ = make_sink(loop)
    sink.set_muted(True)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data())
    assert decryptor.calls == []


def test_allowed_user_audio_is_buffered(loop):
    sink, decryptor, _, _ = make_sink(loop)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data())
    assert decryptor.calls == [ALLOWED]
    assert ALLOWED in sink._buffers


# -- speaking-state coarse gate -----------------------------------------------


def test_speaking_start_gates_on_the_allowlist(loop):
    sink, _, _, consumer = make_sink(loop)
    sink.on_voice_member_speaking_start(SimpleNamespace(id=DENIED))
    loop.run_until_complete(asyncio.sleep(0))
    assert consumer.speech_starts == []


def test_speaking_start_reaches_the_consumer_for_an_allowed_user(loop):
    sink, _, _, consumer = make_sink(loop)

    async def drive():
        sink.on_voice_member_speaking_start(SimpleNamespace(id=ALLOWED))
        await asyncio.sleep(0)

    loop.run_until_complete(drive())
    assert consumer.speech_starts == [ALLOWED]


def test_speaking_stop_does_not_endpoint_the_turn(loop):
    sink, _, _, consumer = make_sink(loop)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data())
    sink.on_voice_member_speaking_stop(SimpleNamespace(id=ALLOWED))
    assert consumer.utterances == []
    assert ALLOWED in sink._buffers


# -- idle reclamation ---------------------------------------------------------


def test_write_records_activity_and_reclaims_nothing_while_recent(loop):
    # Regression: _reclaim_idle_buffers used time.monotonic() with no `time`
    # import, so every second frame raised NameError inside the reader thread.
    sink, _, _, _ = make_sink(loop)
    for user_id in (ALLOWED, ALLOWED):
        sink.write(SimpleNamespace(id=user_id), voice_data())
    assert sink._last_audio_at[ALLOWED] == pytest.approx(time.monotonic(), abs=5.0)
    assert ALLOWED in sink._buffers


def test_idle_buffer_is_reclaimed_with_its_decoder(loop, monkeypatch):
    sink, _, decoders, _ = make_sink(loop)
    monkeypatch.setattr(sink, "_snapshot", two_speaker_snapshot)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data(ssrc=11))
    sink.write(SimpleNamespace(id=OTHER), voice_data(ssrc=22))
    assert set(sink._buffers) == {ALLOWED, OTHER}

    # Age one speaker out. Reclamation only runs on the next inbound frame.
    sink._last_audio_at[OTHER] = time.monotonic() - sink_mod.IDLE_BUFFER_TTL_S - 1
    sink.write(SimpleNamespace(id=ALLOWED), voice_data(ssrc=11))

    assert set(sink._buffers) == {ALLOWED}
    assert decoders.dropped == [22]


def test_a_lone_speaker_is_never_reclaimed(loop):
    # Reclaiming the only speaker would throw away resampler state for no gain.
    sink, _, _, _ = make_sink(loop)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data())
    sink._last_audio_at[ALLOWED] = time.monotonic() - sink_mod.IDLE_BUFFER_TTL_S - 1
    sink.write(SimpleNamespace(id=ALLOWED), voice_data())
    assert ALLOWED in sink._buffers


def test_mid_utterance_buffer_survives_reclamation(loop, monkeypatch):
    sink, _, _, _ = make_sink(loop)
    monkeypatch.setattr(sink, "_snapshot", two_speaker_snapshot)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data(ssrc=11))
    sink.write(SimpleNamespace(id=OTHER), voice_data(ssrc=22))

    # Speech in flight: the VAD has opened a turn we have not endpointed yet.
    sink._buffers[OTHER]._speaking = True
    assert sink._buffers[OTHER].speaking
    sink._last_audio_at[OTHER] = time.monotonic() - sink_mod.IDLE_BUFFER_TTL_S - 1

    sink.write(SimpleNamespace(id=ALLOWED), voice_data(ssrc=11))
    assert OTHER in sink._buffers


# -- disconnect / cleanup -----------------------------------------------------


def test_disconnect_drops_state_and_forgets_the_decrypt_counter(loop):
    sink, decryptor, decoders, _ = make_sink(loop)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data(ssrc=11))
    sink.on_voice_member_disconnect(SimpleNamespace(id=ALLOWED))
    assert ALLOWED not in sink._buffers
    assert ALLOWED not in sink._last_audio_at
    assert decryptor.forgotten == [ALLOWED]
    assert decoders.dropped == [11]


def test_cleanup_clears_every_per_user_map(loop):
    sink, _, _, _ = make_sink(loop)
    sink.write(SimpleNamespace(id=ALLOWED), voice_data())
    sink.cleanup()
    assert sink._buffers == {} and sink._last_audio_at == {} and sink._ssrc_for_user == {}
