"""Per-user Opus decoding.

The sink asks for Opus rather than PCM (see audio/dave.py for why), so decoding is
ours to do. One decoder per SSRC is required, not merely tidy: an Opus decoder carries
inter-frame state, and feeding two speakers' packets through one decoder corrupts both.

Passing None to `decode()` invokes packet-loss concealment, which is how a gap in the
stream should be handled -- inserting silence instead leaves an audible discontinuity
and can clip the start of the next word.
"""

from __future__ import annotations

import logging

from discord.opus import Decoder, OpusError

log = logging.getLogger(__name__)

SAMPLE_RATE = Decoder.SAMPLING_RATE  # 48000
CHANNELS = Decoder.CHANNELS  # 2
FRAME_BYTES = Decoder.FRAME_SIZE  # 3840 == 20 ms stereo s16le


class OpusDecoderPool:
    """Keeps one Opus decoder per SSRC and disposes of them on disconnect."""

    __slots__ = ("_decoders", "decoded", "failed")

    def __init__(self) -> None:
        self._decoders: dict[int, Decoder] = {}
        # A sustained run of failures here is the signature of feeding the
        # decoder ciphertext, which is what a broken DAVE decrypt looks like
        # from downstream. Counted so ReceiveDiagnosis can say so out loud
        # instead of leaving the channel quiet with only DEBUG lines.
        # Concealment frames are excluded: they are our own synthesis, not
        # evidence that anything arrived intact.
        self.decoded = 0
        self.failed = 0

    def decode(self, ssrc: int, payload: bytes | None) -> bytes:
        """Decode one frame to 48 kHz stereo s16le. Returns b"" on failure."""
        decoder = self._decoders.get(ssrc)
        if decoder is None:
            decoder = Decoder()
            self._decoders[ssrc] = decoder
        try:
            pcm = decoder.decode(payload)
        except OpusError as exc:
            if payload is not None:
                self.failed += 1
            log.debug("opus decode failed for ssrc %s: %s", ssrc, exc)
            return b""
        if payload is not None:
            self.decoded += 1
        return pcm

    def conceal(self, ssrc: int) -> bytes:
        """Generate one frame of loss concealment for a gap in the stream."""
        return self.decode(ssrc, None)

    def drop(self, ssrc: int) -> None:
        self._decoders.pop(ssrc, None)

    def clear(self) -> None:
        self._decoders.clear()

    def __len__(self) -> int:
        return len(self._decoders)
