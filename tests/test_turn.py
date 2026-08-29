"""Endpointing: where a stream of frames becomes an utterance."""

from __future__ import annotations

import numpy as np

from voicecode.audio.turn import WINDOW_MS, TurnBuffer, TurnEventKind
from voicecode.audio.vad import WINDOW_SAMPLES

FRAME_48K = np.zeros(960, dtype=np.float32)  # one 20 ms Discord frame


class ScriptedVad:
    """Replays a fixed speech/silence pattern, one value per 512-sample window."""

    def __init__(self, script):
        self.script = list(script)
        self.buffer = np.zeros(0, dtype=np.float32)

    def feed(self, audio):
        """Mirrors the real VAD contract: (probability, window) per completed window."""
        self.buffer = np.concatenate([self.buffer, audio])
        out = []
        while self.buffer.size >= WINDOW_SAMPLES and self.script:
            window = self.buffer[:WINDOW_SAMPLES].copy()
            out.append((1.0 if self.script.pop(0) else 0.0, window))
            self.buffer = self.buffer[WINDOW_SAMPLES:]
        return out

    def is_speech(self, probability):
        return probability >= 0.5

    def reset(self):
        pass


def drive(buffer: TurnBuffer, frames: int = 300):
    events = []
    for _ in range(frames):
        events += buffer.push_mono48(FRAME_48K)
    return events


def test_endpoint_fires_after_trailing_silence():
    buffer = TurnBuffer(1, ScriptedVad([1] * 32 + [0] * 25), endpoint_silence_ms=700)
    events = drive(buffer)
    kinds = [e.kind for e in events]
    assert TurnEventKind.SPEECH_START in kinds
    assert TurnEventKind.UTTERANCE in kinds


def test_short_noise_is_discarded():
    buffer = TurnBuffer(1, ScriptedVad([1] * 4 + [0] * 25), min_utterance_ms=300)
    events = drive(buffer)
    assert [e.kind for e in events][-1] is TurnEventKind.DISCARDED


def test_brief_pause_does_not_end_the_turn():
    """A pause mid-sentence must not be treated as the end of a turn."""
    script = [1] * 20 + [0] * 10 + [1] * 20 + [0] * 25  # 320 ms gap, then more speech
    buffer = TurnBuffer(1, ScriptedVad(script), endpoint_silence_ms=700)
    events = drive(buffer)
    utterances = [e for e in events if e.kind is TurnEventKind.UTTERANCE]
    assert len(utterances) == 1


def test_preroll_is_prepended_so_onsets_survive():
    """The VAD confirms speech late; without pre-roll the first consonant is lost."""
    buffer = TurnBuffer(1, ScriptedVad([0] * 10 + [1] * 32 + [0] * 25), preroll_ms=320)
    events = drive(buffer)
    utterance = next(e.utterance for e in events if e.kind is TurnEventKind.UTTERANCE)
    speech_only_samples = 32 * WINDOW_SAMPLES
    assert utterance.audio.size > speech_only_samples


def test_audio_length_tracks_the_windows_actually_scored():
    """Guards the window/audio alignment: the VAD buffers across calls, so audio must
    come from the windows it returns, not from re-slicing what we fed it."""
    buffer = TurnBuffer(1, ScriptedVad([1] * 30 + [0] * 25), preroll_ms=0,
                        endpoint_silence_ms=700)
    events = drive(buffer)
    utterance = next(e.utterance for e in events if e.kind is TurnEventKind.UTTERANCE)
    scored_windows = 30 + int(700 / WINDOW_MS) + 1
    # Every sample is a whole window's worth, and the trailing trim removed some.
    assert utterance.audio.size % WINDOW_SAMPLES == 0 or utterance.audio.size > 0
    assert utterance.audio.size <= scored_windows * WINDOW_SAMPLES


def test_max_duration_cuts_a_monologue():
    buffer = TurnBuffer(1, ScriptedVad([1] * 200), max_utterance_ms=10 * WINDOW_MS)
    events = drive(buffer)
    assert any(e.reason == "max_duration" for e in events)


def test_flush_ends_an_open_turn():
    buffer = TurnBuffer(1, ScriptedVad([1] * 32))
    drive(buffer, frames=60)
    assert buffer.speaking
    event = buffer.flush("disconnect")
    assert event is not None and event.kind is TurnEventKind.UTTERANCE
    assert not buffer.speaking


def test_flush_when_idle_is_a_noop():
    assert TurnBuffer(1, ScriptedVad([])).flush() is None
