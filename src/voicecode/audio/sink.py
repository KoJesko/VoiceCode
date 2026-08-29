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
from collections.abc import Callable
from typing import Any, Protocol

from discord.ext import voice_recv

from ..config import AllowlistSnapshot
from .dave import DaveDecryptor
from .opus_decode import OpusDecoderPool
from .turn import TurnBuffer, TurnEvent, TurnEventKind
from .vad import EnergyVad, SileroVad

log = logging.getLogger(__name__)


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
        self._ssrc_for_user: dict[int, int] = {}
        self._muted = False
        self._accepting = True

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

        plaintext = self._decryptor.decrypt(user_id, payload)
        if plaintext is None:
            return

        pcm = self._decoders.decode(ssrc, plaintext)
        if not pcm:
            return

        try:
            events = self._buffer_for(user_id).push_pcm(pcm)
        except Exception:
            log.exception("turn buffering failed for user %s", user_id)
            return

        for event in events:
            self._emit(event)

    # -- speaking-state listeners ----------------------------------------------
    # These are dispatched to sinks only -- a @bot.event handler for them never
    # fires -- and they arrive on the router thread. They are a coarse gate: the
    # extension derives them from packet activity with a 200 ms timeout, which is
    # far too slack to endpoint a turn. The VAD does that.

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member: Any) -> None:
        if not self._snapshot().user_allowed(getattr(member, "id", None)):
            return
        log.debug("speaking start: %s", getattr(member, "display_name", member))

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: Any) -> None:
        user_id = getattr(member, "id", None)
        if not self._snapshot().user_allowed(user_id):
            return
        buffer = self._buffers.get(user_id)
        if buffer is None or not buffer.speaking:
            return
        # Do not endpoint here. Discord's speaking flag drops during natural pauses
        # mid-sentence; letting it end the turn truncates people constantly. The VAD
        # owns the endpoint; this event only marks the stream as idle.
        log.debug("speaking stop: %s (VAD still owns the endpoint)", user_id)

    @voice_recv.AudioSink.listener()
    def on_voice_member_disconnect(self, member: Any, ssrc: int | None = None) -> None:
        user_id = getattr(member, "id", None)
        if user_id is None:
            return
        buffer = self._buffers.pop(user_id, None)
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
