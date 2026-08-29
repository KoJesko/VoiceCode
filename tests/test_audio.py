"""Format conversion. Getting a rate or a channel count wrong here is silent."""

from __future__ import annotations

import numpy as np

from voicecode.audio.resample import (
    ASR_RATE,
    DISCORD_FRAME_BYTES,
    DISCORD_RATE,
    KOKORO_RATE,
    StreamResampler,
    discord_frames_from_float,
    float_to_pcm_bytes,
    pcm_bytes_to_mono_float,
    resample,
)


def tone(seconds=1.0, hz=440.0, rate=DISCORD_RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_discord_frame_geometry():
    """20 ms of 48 kHz stereo s16le is 3840 bytes; discord.opus agrees."""
    frame = float_to_pcm_bytes(np.zeros(960, dtype=np.float32))
    assert len(frame) == DISCORD_FRAME_BYTES


def test_pcm_round_trip_is_lossless_to_quantisation():
    signal = tone(0.1)
    back = pcm_bytes_to_mono_float(float_to_pcm_bytes(signal))
    assert back.shape == signal.shape
    assert np.abs(back - signal).max() < 1e-4


def test_downmix_averages_channels():
    """Not a left-channel pick: a panned client would otherwise lose half its level."""
    interleaved = np.array([1000, 3000, -1000, -3000], dtype="<i2").tobytes()
    mono = pcm_bytes_to_mono_float(interleaved, channels=2)
    assert np.allclose(mono, np.array([2000, -2000]) / 32768.0, atol=1e-4)


def test_clipping_prevents_wraparound():
    """An unclipped cast turns an overshoot into full-scale negative -- a loud pop."""
    loud = np.array([1.5, -1.5], dtype=np.float32)
    ints = np.frombuffer(float_to_pcm_bytes(loud, channels=1), dtype="<i2")
    assert ints[0] > 0 and ints[1] < 0


def test_odd_length_buffer_does_not_crash():
    """A truncated frame must not raise inside the reader thread."""
    assert pcm_bytes_to_mono_float(b"\x01\x00\x02\x00\x03\x00", channels=2).size == 1


def test_resample_lengths():
    assert resample(tone(1.0), DISCORD_RATE, ASR_RATE).size == ASR_RATE
    upsampled = resample(np.zeros(KOKORO_RATE, dtype=np.float32), KOKORO_RATE, DISCORD_RATE)
    assert upsampled.size == DISCORD_RATE


def test_resample_preserves_the_tone():
    """A real resample, not decimation: 440 Hz must survive 48k -> 16k."""
    downsampled = resample(tone(1.0, 440.0), DISCORD_RATE, ASR_RATE)
    spectrum = np.abs(np.fft.rfft(downsampled))
    peak_hz = np.fft.rfftfreq(downsampled.size, 1 / ASR_RATE)[spectrum.argmax()]
    assert abs(peak_hz - 440.0) < 5.0


def test_stream_resampler_matches_one_shot_length_closely():
    """The shortfall is the filter's internal latency, held in state."""
    signal = tone(1.0)
    stream = StreamResampler(DISCORD_RATE, ASR_RATE)
    streamed = np.concatenate(
        [stream.feed(signal[i : i + 960]) for i in range(0, signal.size, 960)]
    )
    assert 0 < ASR_RATE - streamed.size < 512


def test_frames_are_whole_and_padded():
    frames = discord_frames_from_float(np.zeros(KOKORO_RATE, dtype=np.float32), KOKORO_RATE)
    assert frames
    assert {len(f) for f in frames} == {DISCORD_FRAME_BYTES}


def test_ragged_tail_is_padded_not_truncated():
    """A short final frame would be played as a click."""
    frames = discord_frames_from_float(np.zeros(KOKORO_RATE + 137, dtype=np.float32), KOKORO_RATE)
    assert {len(f) for f in frames} == {DISCORD_FRAME_BYTES}


def test_empty_input_produces_no_frames():
    assert discord_frames_from_float(np.zeros(0, dtype=np.float32), KOKORO_RATE) == []
