"""Per-user buffering and endpointing: where a stream of frames becomes an utterance.

One TurnBuffer per speaking user. It owns that user's resampler state, VAD state, and
audio buffer, and emits an Utterance once the speaker has been quiet for long enough.

Two details that matter more than they look:

* **Pre-roll.** VAD confirms speech a window or two after it actually began, so a
  buffer that starts filling at the trigger clips the first consonant -- "delete" and
  "elite" is the kind of pair that costs you. A rolling pre-roll buffer is retained at
  all times and prepended when speech starts.

* **Trailing trim.** The endpoint fires after `endpoint_silence_ms` of quiet, but all
  of that silence is in the buffer. Most is trimmed before ASR, leaving a short tail --
  transducer models want a little run-out to finalise the last token, but not 700 ms
  of it.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum, auto

import numpy as np

from .resample import ASR_RATE, DISCORD_RATE, StreamResampler, pcm_bytes_to_mono_float
from .vad import WINDOW_SAMPLES

WINDOW_MS = WINDOW_SAMPLES / ASR_RATE * 1000.0  # 32 ms
# Kept after the endpoint so the ASR has run-out without a long silent tail.
TAIL_KEEP_MS = 200.0


class TurnEventKind(StrEnum):
    SPEECH_START = auto()
    UTTERANCE = auto()
    DISCARDED = auto()


@dataclass(frozen=True, slots=True)
class Utterance:
    user_id: int
    audio: np.ndarray  # 16 kHz mono float32
    duration_ms: float
    ended_at: float


@dataclass(frozen=True, slots=True)
class TurnEvent:
    kind: TurnEventKind
    user_id: int
    utterance: Utterance | None = None
    reason: str = ""


@dataclass
class TurnBuffer:
    """Accumulates one user's audio and decides where their turn ends."""

    user_id: int
    vad: object
    endpoint_silence_ms: float = 700.0
    min_utterance_ms: float = 300.0
    max_utterance_ms: float = 30_000.0
    preroll_ms: float = 320.0

    _resampler: StreamResampler = field(init=False)
    _chunks: list[np.ndarray] = field(default_factory=list, init=False)
    _preroll: deque[np.ndarray] = field(init=False)
    _speaking: bool = field(default=False, init=False)
    _silence_ms: float = field(default=0.0, init=False)
    _speech_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._resampler = StreamResampler(DISCORD_RATE, ASR_RATE, channels=1)
        self._preroll = deque(maxlen=max(1, int(self.preroll_ms / WINDOW_MS)))

    # -- state ------------------------------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def buffered_ms(self) -> float:
        return self._speech_ms

    def reset(self) -> None:
        self._chunks.clear()
        self._preroll.clear()
        self._speaking = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        if hasattr(self.vad, "reset"):
            self.vad.reset()

    # -- ingest -----------------------------------------------------------------

    def push_pcm(self, pcm: bytes) -> list[TurnEvent]:
        """Feed one decoded Discord frame (48 kHz stereo s16le)."""
        mono = pcm_bytes_to_mono_float(pcm)
        return self.push_mono48(mono)

    def push_mono48(self, mono48: np.ndarray) -> list[TurnEvent]:
        resampled = self._resampler.feed(mono48)
        return self._advance(resampled)

    def _advance(self, audio16: np.ndarray) -> list[TurnEvent]:
        events: list[TurnEvent] = []
        probabilities = self.vad.feed(audio16)
        if not probabilities:
            return events

        # feed() consumes whole windows only, so probabilities map 1:1 onto the
        # leading WINDOW_SAMPLES*n samples it just accepted.
        consumed = len(probabilities) * WINDOW_SAMPLES
        windows = np.split(audio16[:consumed], len(probabilities)) if consumed else []

        for probability, window in zip(probabilities, windows, strict=False):
            is_speech = self.vad.is_speech(probability)

            if not self._speaking:
                self._preroll.append(window)
                if is_speech:
                    self._speaking = True
                    self._silence_ms = 0.0
                    self._speech_ms = 0.0
                    # Prepend the pre-roll so the word onset survives.
                    self._chunks.extend(self._preroll)
                    self._speech_ms += len(self._preroll) * WINDOW_MS
                    self._preroll.clear()
                    events.append(TurnEvent(TurnEventKind.SPEECH_START, self.user_id))
                continue

            self._chunks.append(window)
            self._speech_ms += WINDOW_MS
            self._silence_ms = 0.0 if is_speech else self._silence_ms + WINDOW_MS

            if self._silence_ms >= self.endpoint_silence_ms:
                events.append(self._finish("endpoint"))
            elif self._speech_ms >= self.max_utterance_ms:
                # A long monologue is cut rather than buffered without limit; the
                # remainder starts a fresh turn.
                events.append(self._finish("max_duration"))

        return events

    # -- endpoint ---------------------------------------------------------------

    def flush(self, reason: str = "flush") -> TurnEvent | None:
        """End the turn now, e.g. on a speaking-stop event or a disconnect."""
        if not self._speaking:
            return None
        return self._finish(reason)

    def _finish(self, reason: str) -> TurnEvent:
        audio = (
            np.concatenate(self._chunks)
            if self._chunks
            else np.zeros(0, dtype=np.float32)
        )
        spoken_ms = max(0.0, self._speech_ms - self._silence_ms)

        # Trim the silence that triggered the endpoint, keeping a short run-out.
        if self._silence_ms > TAIL_KEEP_MS:
            drop = int((self._silence_ms - TAIL_KEEP_MS) / 1000.0 * ASR_RATE)
            if 0 < drop < audio.size:
                audio = audio[: audio.size - drop]

        self._chunks = []
        self._preroll.clear()
        self._speaking = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        if hasattr(self.vad, "reset"):
            self.vad.reset()

        if spoken_ms < self.min_utterance_ms:
            return TurnEvent(
                TurnEventKind.DISCARDED,
                self.user_id,
                reason=f"{reason}: {spoken_ms:.0f}ms below {self.min_utterance_ms:.0f}ms floor",
            )

        return TurnEvent(
            TurnEventKind.UTTERANCE,
            self.user_id,
            utterance=Utterance(
                user_id=self.user_id,
                audio=audio,
                duration_ms=spoken_ms,
                ended_at=time.perf_counter(),
            ),
            reason=reason,
        )
