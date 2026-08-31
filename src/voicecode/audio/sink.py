"""The AudioSink: where Discord audio enters the bot, and where most of it is dropped.

Threading
---------
Nothing in this class runs on the asyncio event loop. `write()` is called from the
receive extension's reader thread; the speaking listeners are called from the router's
own queue-consumer thread. Every hop into bot logic therefore goes through
`run_coroutine_threadsafe`. Calling an async function directly from here, or awaiting
anything, deadlocks the reader.

Opus, not PCM
-------------
`wants_opus()` returns True so we receive the raw Opus payload. That is required, not
a preference: the payload is still DAVE-encrypted at this point, and the extension's
own PCM path would decode ciphertext into noise. See audio/dave.py.

The allowlist gate
------------------
The USER_ALLOWLIST check is the first statement of `write()`, above decryption and
decoding. A non-allowlisted speaker's audio is discarded while still encrypted: it
never reaches a decoder, a buffer, the VAD, or the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from discord.ext import voice_recv

from ..config import AllowlistSnapshot
from .dave import DaveDecryptor
from .opus_decode import OpusDecoderPool
from .turn import TurnBuffer, TurnEvent, TurnEventKind
from .vad import EnergyVad, SileroVad

log = logging.getLogger(__name__)


# A user whose buffer has seen no audio for this long is reclaimed, releasing
# their resampler and VAD state. They get a fresh buffer if they speak again.
IDLE_BUFFER_TTL_S = 120.0

# Frames to observe before the receive path is willing to call itself broken.
# 250 frames is 5 seconds of speech: long enough that a healthy path has
# certainly decoded something, short enough to fail during the first sentence
# somebody speaks rather than after a frustrating session.
DIAGNOSIS_MIN_FRAMES = 250


@dataclass(frozen=True, slots=True)
class ReceiveDiagnosis:
    """Whether audio is actually arriving, and if not, which stage lost it.

    This exists because the receive path fails silently. Decrypting with the
    wrong key does not raise: it produces bytes that are not Opus, the decoder
    rejects them at DEBUG level, and the channel simply stays quiet. The same
    silence is produced by a missing intent, a deafened bot, an empty
    USER_ALLOWLIST, and a VAD threshold set too high -- four unrelated
    problems with one symptom. Counting each stage separates them.
    """

    frames_in: int
    decrypted: int
    passthrough: int
    decrypt_dropped: int
    decoded: int
    decode_failed: int
    utterances: int
    dave: str

    @property
    def healthy(self) -> bool:
        return self.frames_in > 0 and self.decoded > 0

    @property
    def conclusive(self) -> bool:
        """True once enough frames have arrived for `verdict` to mean something."""
        return self.frames_in >= DIAGNOSIS_MIN_FRAMES

    def verdict(self) -> str:
        """One line naming the stage that lost the audio, and what causes it."""
        if self.frames_in == 0:
            return (
                "no audio received at all -- check that the bot is not "
                "server-deafened, that the voice_states intent is on, and that "
                "the speaker is in USER_ALLOWLIST (their audio is dropped at "
                "the sink before decryption)"
            )
        if not self.conclusive:
            return f"only {self.frames_in} frame(s) so far; too early to judge"
        if self.decoded == 0:
            if self.decrypt_dropped > self.passthrough:
                return (
                    f"DAVE decryption is failing ({self.decrypt_dropped} frame(s) "
                    "dropped) -- the session is active but rejecting our calls"
                )
            return (
                f"every Opus decode failed ({self.decode_failed} frame(s)) while "
                f"DAVE reported {self.dave!r}. Frames are arriving still "
                "encrypted: the decrypt step is being skipped, not performed. "
                "This is the failure audio/dave.py exists to prevent -- treat a "
                "passthrough-heavy tally here as the bug, not as normal"
            )
        if self.utterances == 0:
            return (
                f"{self.decoded} frame(s) decoded cleanly but nothing was "
                "endpointed -- audio is healthy and the VAD is the problem; "
                "lower VAD_THRESHOLD or MIN_UTTERANCE_MS"
            )
        return (
            f"healthy -- {self.decoded} frame(s) decoded, "
            f"{self.utterances} utterance(s) endpointed"
        )

    def describe(self) -> str:
        return (
            f"in={self.frames_in} decrypted={self.decrypted} "
            f"passthrough={self.passthrough} dropped={self.decrypt_dropped} "
            f"decoded={self.decoded} decode_failed={self.decode_failed} "
            f"utterances={self.utterances}\n{self.verdict()}"
        )


class TurnConsumer(Protocol):
    """What the sink hands finished turns to. Implemented by the voice session."""

    async def on_speech_start(self, user_id: int) -> None: ...
    async def on_utterance(self, event: TurnEvent) -> None: ...


class VoiceCodeSink(voice_recv.AudioSink):
    """Receives per-SSRC Opus, decrypts, decodes, endpoints, and forwards turns."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        snapshot_provider: Callable[[], AllowlistSnapshot],
        decryptor: DaveDecryptor,
        consumer: TurnConsumer,
        endpoint_silence_ms: float = 700.0,
        min_utterance_ms: float = 300.0,
        max_utterance_ms: float = 30_000.0,
        vad_threshold: float = 0.5,
        use_silero: bool = True,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._snapshot = snapshot_provider
        self._decryptor = decryptor
        self._consumer = consumer
        self._endpoint_silence_ms = endpoint_silence_ms
        self._min_utterance_ms = min_utterance_ms
        self._max_utterance_ms = max_utterance_ms
        self._vad_threshold = vad_threshold
        self._use_silero = use_silero

        self._decoders = OpusDecoderPool()
        self._buffers: dict[int, TurnBuffer] = {}
        self._last_audio_at: dict[int, float] = {}
        self._ssrc_for_user: dict[int, int] = {}
        self._muted = False
        self._accepting = True
        self._frames_in = 0
        self._utterances = 0
        self._warned_broken = False

    # -- required AudioSink surface --------------------------------------------

    def wants_opus(self) -> bool:
        return True

    def cleanup(self) -> None:
        for user_id, buffer in list(self._buffers.items()):
            event = buffer.flush("cleanup")
            if event and event.kind is TurnEventKind.UTTERANCE:
                self._dispatch(self._consumer.on_utterance(event))
            self._decryptor.forget(user_id)
        self._buffers.clear()
        self._last_audio_at.clear()
        self._decoders.clear()
        self._ssrc_for_user.clear()

    def write(self, user: Any | None, data: voice_recv.VoiceData) -> None:
        # --- SCOPING GATE #7. Keep this first. ---------------------------------
        if user is None:
            return
        user_id = getattr(user, "id", None)
        if not self._snapshot().user_allowed(user_id):
            return
        # -----------------------------------------------------------------------

        if self._muted or not self._accepting:
            return

        payload = data.opus
        if not payload:
            return

        packet = data.packet
        ssrc = getattr(packet, "ssrc", user_id)
        self._ssrc_for_user[user_id] = ssrc
        self._frames_in += 1

        plaintext = self._decryptor.decrypt(user_id, payload)
        if plaintext is None:
            self._check_receive_health()
            return

        pcm = self._decoders.decode(ssrc, plaintext)
        if not pcm:
            self._check_receive_health()
            return

        self._last_audio_at[user_id] = time.monotonic()
        self._reclaim_idle_buffers()

        try:
            events = self._buffer_for(user_id).push_pcm(pcm)
        except Exception:
            log.exception("turn buffering failed for user %s", user_id)
            return

        for event in events:
            if event.kind is TurnEventKind.UTTERANCE:
                self._utterances += 1
            self._emit(event)

    # -- diagnosis --------------------------------------------------------------

    @property
    def diagnosis(self) -> ReceiveDiagnosis:
        """A snapshot of what the receive path has actually managed to do."""
        return ReceiveDiagnosis(
            frames_in=self._frames_in,
            decrypted=self._decryptor.decrypted,
            passthrough=self._decryptor.passthrough,
            decrypt_dropped=self._decryptor.dropped,
            decoded=self._decoders.decoded,
            decode_failed=self._decoders.failed,
            utterances=self._utterances,
            dave=self._decryptor.status.describe(),
        )

    def _check_receive_health(self) -> None:
        """Say so, once, when frames are arriving but none of them survive.

        Called only on the failure branches, so the healthy path costs nothing.
        A single WARNING is the whole point: without it this failure has no
        symptom other than the bot never answering, which reads as a hung ASR
        or a bad microphone rather than as a decryption problem.
        """
        if self._warned_broken:
            return
        report = self.diagnosis
        if not report.conclusive or report.healthy:
            return
        self._warned_broken = True
        log.warning("receive path is not producing audio: %s", report.describe())

    # -- speaking-state listeners ----------------------------------------------
    # These are dispatched to sinks only -- a @bot.event handler for them never
    # fires -- and they arrive on the router thread. They are a coarse gate: the
    # extension derives them from packet activity with a 200 ms timeout, which is
    # far too slack to endpoint a turn. The VAD does that.

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member: Any) -> None:
        """Coarse gate: the earliest signal that someone has started talking.

        This is Discord's own speaking flag, sent by the speaker's client and
        delivered with their first packet -- ahead of anything the VAD can
        confirm, which needs a window or two plus resampler latency. That head
        start is worth having for barge-in specifically, where being late is
        heard as the bot talking over you. It does NOT start a turn: the VAD
        still decides what counts as speech and where it ends.
        """
        user_id = getattr(member, "id", None)
        if not self._snapshot().user_allowed(user_id):
            return
        if self._muted or not self._accepting:
            return
        log.debug("speaking start (coarse gate): %s", user_id)
        self._dispatch(self._consumer.on_speech_start(user_id))

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: Any) -> None:
        user_id = getattr(member, "id", None)
        if not self._snapshot().user_allowed(user_id):
            return
        buffer = self._buffers.get(user_id)
        if buffer is None or not buffer.speaking:
            return
        # Do not endpoint here. Discord's speaking flag drops during natural pauses
        # mid-sentence -- the extension derives it from packet activity with a
        # 200 ms timeout -- so ending the turn on it would truncate people
        # constantly. The VAD owns the endpoint.
        log.debug("speaking stop: %s (VAD still owns the endpoint)", user_id)

    @voice_recv.AudioSink.listener()
    def on_voice_member_disconnect(self, member: Any, ssrc: int | None = None) -> None:
        user_id = getattr(member, "id", None)
        if user_id is None:
            return
        buffer = self._buffers.pop(user_id, None)
        self._last_audio_at.pop(user_id, None)
        if buffer is not None:
            event = buffer.flush("disconnect")
            if event is not None:
                self._emit(event)
        known_ssrc = ssrc if ssrc is not None else self._ssrc_for_user.pop(user_id, None)
        if known_ssrc is not None:
            self._decoders.drop(known_ssrc)
        self._decryptor.forget(user_id)

    # -- control ----------------------------------------------------------------

    def set_muted(self, muted: bool) -> None:
        """/mute: stop consuming audio without leaving the channel."""
        self._muted = muted
        if muted:
            for buffer in self._buffers.values():
                buffer.reset()

    def stop_accepting(self) -> None:
        self._accepting = False

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def decryptor(self) -> DaveDecryptor:
        return self._decryptor

    # -- internals --------------------------------------------------------------

    def _reclaim_idle_buffers(self) -> None:
        """Drop per-user state for anyone who has been silent for a while.

        Each buffer holds a soxr resampler and a silero RNN state, so a busy
        channel would otherwise accumulate them for every user who ever spoke.
        A buffer mid-utterance is never reclaimed -- that would discard audio.
        """
        if len(self._buffers) < 2:
            return
        cutoff = time.monotonic() - IDLE_BUFFER_TTL_S
        for user_id, last in list(self._last_audio_at.items()):
            if last > cutoff:
                continue
            buffer = self._buffers.get(user_id)
            if buffer is not None and buffer.speaking:
                continue
            self._buffers.pop(user_id, None)
            self._last_audio_at.pop(user_id, None)
            ssrc = self._ssrc_for_user.pop(user_id, None)
            if ssrc is not None:
                self._decoders.drop(ssrc)
            log.debug("reclaimed idle turn buffer for user %s", user_id)

    def _buffer_for(self, user_id: int) -> TurnBuffer:
        buffer = self._buffers.get(user_id)
        if buffer is None:
            vad: object
            if self._use_silero:
                try:
                    SileroVad.load()
                    vad = SileroVad(self._vad_threshold)
                except Exception:
                    log.exception("silero-vad unavailable; falling back to energy VAD")
                    self._use_silero = False
                    vad = EnergyVad()
            else:
                vad = EnergyVad()
            buffer = TurnBuffer(
                user_id=user_id,
                vad=vad,
                endpoint_silence_ms=self._endpoint_silence_ms,
                min_utterance_ms=self._min_utterance_ms,
                max_utterance_ms=self._max_utterance_ms,
            )
            self._buffers[user_id] = buffer
        return buffer

    def _emit(self, event: TurnEvent) -> None:
        if event.kind is TurnEventKind.SPEECH_START:
            self._dispatch(self._consumer.on_speech_start(event.user_id))
        elif event.kind is TurnEventKind.UTTERANCE:
            self._dispatch(self._consumer.on_utterance(event))
        elif event.kind is TurnEventKind.DISCARDED:
            log.debug("discarded turn from %s (%s)", event.user_id, event.reason)

    def _dispatch(self, coro) -> None:
        """Hand a coroutine to the event loop from a library thread."""
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            coro.close()
            return

        def _log_failure(fut) -> None:
            try:
                fut.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("voice turn handler failed")

        future.add_done_callback(_log_failure)
