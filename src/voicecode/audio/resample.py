"""Format conversion between Discord's wire audio and the model sample rates.

Rates in play:
  Discord RX : 48 kHz, 2 ch, s16le, 20 ms frames (3840 bytes)
  Parakeet   : 16 kHz, 1 ch, float32
  Kokoro     : 24 kHz, 1 ch, float32
  Discord TX : 48 kHz, 2 ch, s16le

All rate conversion goes through soxr. Nothing here decimates by dropping samples --
48k -> 16k by taking every third sample aliases everything above 8 kHz straight into
the speech band, which is exactly where the consonants are.
"""

from __future__ import annotations

import numpy as np
import soxr

DISCORD_RATE = 48_000
DISCORD_CHANNELS = 2
ASR_RATE = 16_000
KOKORO_RATE = 24_000

# 20 ms of 48 kHz stereo s16le.
DISCORD_FRAME_BYTES = 3840
DISCORD_FRAME_SAMPLES = 960

_INT16_SCALE = 32768.0


def pcm_bytes_to_mono_float(data: bytes, channels: int = DISCORD_CHANNELS) -> np.ndarray:
    """s16le interleaved PCM -> mono float32 in [-1, 1].

    Downmix is a mean across channels, not a left-channel pick: Discord sends the same
    signal on both channels for most clients, but a client that pans would otherwise
    lose half its level.
    """
    if not data:
        return np.zeros(0, dtype=np.float32)

    samples = np.frombuffer(data, dtype="<i2")
    if channels > 1:
        usable = (samples.size // channels) * channels
        if usable != samples.size:
            samples = samples[:usable]
        samples = samples.reshape(-1, channels).mean(axis=1)
    return (samples.astype(np.float32) / _INT16_SCALE).astype(np.float32)


def float_to_pcm_bytes(audio: np.ndarray, channels: int = DISCORD_CHANNELS) -> bytes:
    """mono float32 -> interleaved s16le, duplicated across `channels`.

    Clipped before the cast: Kokoro occasionally overshoots 1.0 on plosives, and an
    unclipped cast wraps that to full-scale negative, which sounds like a gunshot.
    """
    if audio.size == 0:
        return b""

    mono = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    ints = (mono * (_INT16_SCALE - 1)).astype("<i2")
    if channels > 1:
        ints = np.repeat(ints[:, None], channels, axis=1).reshape(-1)
    return ints.tobytes()


def resample(audio: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """One-shot high-quality resample. Use for complete buffers."""
    if in_rate == out_rate or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    return soxr.resample(
        np.asarray(audio, dtype=np.float32), in_rate, out_rate, quality="HQ"
    ).astype(np.float32)


class StreamResampler:
    """Stateful resampler for a continuous stream.

    The receive path needs this: 20 ms frames arrive indefinitely and the VAD has to
    see 16 kHz audio as it comes, not at end of utterance. Feeding independent one-shot
    resamples per frame would leave a filter discontinuity at every 20 ms boundary --
    audible as a buzz, and enough to disturb the VAD.
    """

    __slots__ = ("_stream", "in_rate", "out_rate")

    def __init__(self, in_rate: int, out_rate: int, channels: int = 1):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = soxr.ResampleStream(
            in_rate, out_rate, channels, dtype="float32", quality="HQ"
        )

    def feed(self, audio: np.ndarray, last: bool = False) -> np.ndarray:
        if audio.size == 0 and not last:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(np.asarray(audio, dtype=np.float32), last=last)
        return np.asarray(out, dtype=np.float32)


def discord_frames_from_float(audio: np.ndarray, rate: int = KOKORO_RATE) -> list[bytes]:
    """Kokoro output -> a list of Discord-ready 20 ms stereo frames.

    The tail is zero-padded to a whole frame. Discord's sender expects exactly
    3840 bytes per frame and a short final frame would be played as a click.
    """
    upsampled = resample(audio, rate, DISCORD_RATE)
    if upsampled.size == 0:
        return []

    remainder = upsampled.size % DISCORD_FRAME_SAMPLES
    if remainder:
        upsampled = np.concatenate(
            [upsampled, np.zeros(DISCORD_FRAME_SAMPLES - remainder, dtype=np.float32)]
        )

    raw = float_to_pcm_bytes(upsampled)
    return [
        raw[i : i + DISCORD_FRAME_BYTES] for i in range(0, len(raw), DISCORD_FRAME_BYTES)
    ]
