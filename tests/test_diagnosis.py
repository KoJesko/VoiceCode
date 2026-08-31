"""The receive path's self-diagnosis.

Four unrelated faults produce one symptom -- the bot never answers -- because
the receive path fails without raising: a wrong DAVE key yields bytes that are
not Opus, the decoder rejects them at DEBUG, and the channel goes quiet. These
tests pin each verdict to the fault it is supposed to name, so the message a
user acts on cannot drift away from the condition that produced it.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from voicecode.audio.opus_decode import OpusDecoderPool
from voicecode.audio.sink import DIAGNOSIS_MIN_FRAMES, ReceiveDiagnosis


def report(**over) -> ReceiveDiagnosis:
    base = dict(
        frames_in=0,
        decrypted=0,
        passthrough=0,
        decrypt_dropped=0,
        decoded=0,
        decode_failed=0,
        utterances=0,
        dave="active (protocol v1)",
    )
    base.update(over)
    return ReceiveDiagnosis(**base)


# -- verdicts -----------------------------------------------------------------


def test_no_frames_names_the_causes_that_drop_audio_before_decryption():
    verdict = report().verdict()
    assert "no audio received" in verdict
    # The three things that silence the path upstream of us must all be named,
    # because none of them leave any other trace.
    assert "deafened" in verdict
    assert "voice_states" in verdict
    assert "USER_ALLOWLIST" in verdict


def test_a_short_sample_refuses_to_judge():
    r = report(frames_in=10, decoded=0, decode_failed=10)
    assert not r.conclusive
    assert "too early to judge" in r.verdict()


def test_every_decode_failing_is_called_a_skipped_decrypt_not_a_codec_bug():
    r = report(
        frames_in=DIAGNOSIS_MIN_FRAMES,
        passthrough=DIAGNOSIS_MIN_FRAMES,
        decode_failed=DIAGNOSIS_MIN_FRAMES,
    )
    verdict = r.verdict()
    assert not r.healthy and r.conclusive
    assert "still encrypted" in verdict
    assert "skipped, not performed" in verdict


def test_active_dave_rejecting_us_is_distinguished_from_passthrough():
    # Both leave decoded==0. They have different fixes, so they must not share
    # a message: this one means davey rejected the call, not that we bypassed it.
    r = report(frames_in=DIAGNOSIS_MIN_FRAMES, decrypt_dropped=DIAGNOSIS_MIN_FRAMES)
    verdict = r.verdict()
    assert "DAVE decryption is failing" in verdict
    assert "still encrypted" not in verdict


def test_clean_decodes_with_no_endpoint_blames_the_vad_and_clears_the_audio_path():
    r = report(frames_in=DIAGNOSIS_MIN_FRAMES, decrypted=250, decoded=250)
    verdict = r.verdict()
    assert r.healthy
    assert "VAD_THRESHOLD" in verdict
    assert "audio is healthy" in verdict


def test_a_working_path_says_so():
    r = report(frames_in=300, decrypted=300, decoded=300, utterances=2)
    assert r.healthy
    assert r.verdict().startswith("healthy")


def test_describe_carries_every_counter_for_a_bug_report():
    line = report(frames_in=1, decrypted=2, passthrough=3, decrypt_dropped=4,
                  decoded=5, decode_failed=6, utterances=7).describe()
    for token in ("in=1", "decrypted=2", "passthrough=3", "dropped=4",
                  "decoded=5", "decode_failed=6", "utterances=7"):
        assert token in line


# -- the counters the verdict is computed from --------------------------------


class FakeDecoder:
    """Real Opus decoding needs libopus and a genuine frame; the pool's
    bookkeeping is what these tests are about."""

    def __init__(self, fail=False):
        self.fail = fail

    def decode(self, payload):
        if self.fail:
            raise _opus_error()
        return b"\x00" * 3840


def _opus_error():
    """An OpusError instance that does not need libopus to exist.

    Its __init__ calls opus_strerror through the loaded library, so building
    one normally would make these tests pass or fail on whether libopus is
    installed -- which has nothing to do with what they check.
    """
    from discord.opus import OpusError

    exc = OpusError.__new__(OpusError)
    exc.code = -4
    return exc


def test_pool_counts_successes_and_failures(monkeypatch):
    pool = OpusDecoderPool()
    pool._decoders[1] = FakeDecoder()
    pool._decoders[2] = FakeDecoder(fail=True)
    pool.decode(1, b"good")
    pool.decode(2, b"bad")
    pool.decode(2, b"bad")
    assert (pool.decoded, pool.failed) == (1, 2)


def test_concealment_frames_are_not_counted_as_received_audio():
    # Concealment is our own synthesis. Counting it as a decode would let a
    # fully broken path report healthy on nothing but invented silence.
    pool = OpusDecoderPool()
    pool._decoders[1] = FakeDecoder()
    pool.conceal(1)
    pool.conceal(1)
    assert pool.decoded == 0


def test_failed_concealment_is_not_counted_either():
    pool = OpusDecoderPool()
    pool._decoders[1] = FakeDecoder(fail=True)
    pool.conceal(1)
    assert pool.failed == 0


# -- the warning ---------------------------------------------------------------


def make_sink_with_broken_decrypt():
    from voicecode.audio.sink import VoiceCodeSink
    from voicecode.config import AllowlistSnapshot

    snap = AllowlistSnapshot(
        frozenset({1}), frozenset({2}), frozenset({3}), {2: 5}, False, 0
    )

    class NeverDecrypts:
        decrypted = passthrough = dropped = 0
        status = SimpleNamespace(describe=lambda: "active (protocol v1)")

        def decrypt(self, user_id, payload):
            self.passthrough += 1
            return payload

        def forget(self, user_id):
            pass

    class AlwaysFails:
        decoded = 0

        def __init__(self):
            self.failed = 0

        def decode(self, ssrc, payload):
            self.failed += 1
            return b""

        def drop(self, ssrc):
            pass

        def clear(self):
            pass

    import asyncio

    loop = asyncio.new_event_loop()
    sink = VoiceCodeSink(
        loop=loop,
        snapshot_provider=lambda: snap,
        decryptor=NeverDecrypts(),
        consumer=SimpleNamespace(),
        use_silero=False,
    )
    sink._decoders = AlwaysFails()
    return sink, loop


def frame():
    return SimpleNamespace(
        opus=b"\x01\x02", pcm=None, packet=SimpleNamespace(ssrc=7), source=None
    )


def test_a_broken_path_warns_once_and_names_the_stage(caplog):
    sink, loop = make_sink_with_broken_decrypt()
    try:
        with caplog.at_level(logging.WARNING, logger="voicecode.audio.sink"):
            for _ in range(DIAGNOSIS_MIN_FRAMES + 50):
                sink.write(SimpleNamespace(id=3), frame())
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "one warning, not one per 20 ms frame"
        assert "still encrypted" in warnings[0].getMessage()
    finally:
        loop.close()


def test_a_path_that_has_not_seen_enough_frames_stays_quiet(caplog):
    sink, loop = make_sink_with_broken_decrypt()
    try:
        with caplog.at_level(logging.WARNING, logger="voicecode.audio.sink"):
            for _ in range(DIAGNOSIS_MIN_FRAMES - 1):
                sink.write(SimpleNamespace(id=3), frame())
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    finally:
        loop.close()


@pytest.mark.parametrize("field", ["frames_in", "decoded", "utterances"])
def test_diagnosis_property_reflects_live_counters(field):
    sink, loop = make_sink_with_broken_decrypt()
    try:
        sink.write(SimpleNamespace(id=3), frame())
        assert hasattr(sink.diagnosis, field)
        assert sink.diagnosis.frames_in == 1
    finally:
        loop.close()
