"""Playback: a queue-fed AudioSource, and the controller that owns barge-in.

discord.py pulls `read()` from a sender thread every 20 ms and treats an empty return
as end-of-stream. That is awkward for streamed TTS, where the next sentence may still
be synthesising when the previous one drains. So the source returns silence while the
producer is still working and only ends once the producer has said it is finished and
the queue is empty -- otherwise playback would stop at the first gap between sentences.

Barge-in is `flush()`: drop every queued frame and end the stream immediately. It has
to be synchronous and cheap, because it runs the moment an allowlisted user starts
speaking and any latency there is heard as the bot talking over them.
"""

from __future__ import annotations

import logging
import queue
import threading

import discord

from .resample import DISCORD_FRAME_BYTES

log = logging.getLogger(__name__)

SILENCE_FRAME = b"\x00" * DISCORD_FRAME_BYTES
# How long the source waits on an empty queue before giving up, when the producer has
# not finished. Guards against a wedged synthesiser holding the connection open.
_STALL_LIMIT_FRAMES = 250  # 5 seconds of 20 ms frames


class QueuedPCMSource(discord.AudioSource):
    """A 48 kHz stereo s16le source fed frame-by-frame from another thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._producer_done = threading.Event()
        self._cancelled = threading.Event()
        self._stall_frames = 0
        self._frames_played = 0
        self._first_frame_seen = threading.Event()

    # -- AudioSource ------------------------------------------------------------

    def is_opus(self) -> bool:
        return False

    def read(self) -> bytes:
        if self._cancelled.is_set():
            return b""
        try:
            frame = self._queue.get_nowait()
        except queue.Empty:
            if self._producer_done.is_set():
                return b""
            self._stall_frames += 1
            if self._stall_frames > _STALL_LIMIT_FRAMES:
                log.warning("playback stalled with no frames for 5s; ending stream")
                return b""
            return SILENCE_FRAME

        self._stall_frames = 0
        self._frames_played += 1
        if not self._first_frame_seen.is_set():
            self._first_frame_seen.set()
        return frame

    def cleanup(self) -> None:
        self._cancelled.set()

    # -- producer side ----------------------------------------------------------

    def feed(self, frames: list[bytes]) -> None:
        if self._cancelled.is_set():
            return
        for frame in frames:
            self._queue.put(frame)

    def finish(self) -> None:
        """No more frames are coming; end once the queue drains."""
        self._producer_done.set()

    def flush(self) -> int:
        """Barge-in. Drop everything queued and end the stream. Returns frames dropped."""
        self._cancelled.set()
        self._producer_done.set()
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def frames_played(self) -> int:
        return self._frames_played

    def wait_first_frame(self, timeout: float) -> bool:
        return self._first_frame_seen.wait(timeout)


class PlaybackController:
    """Owns the current source for one voice connection."""

    def __init__(self) -> None:
        self._source: QueuedPCMSource | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        source = self._source
        return source is not None and not source.cancelled

    def start(self, voice_client: discord.VoiceClient) -> QueuedPCMSource:
        """Stop whatever is playing and begin a fresh stream."""
        self.stop(voice_client)
        source = QueuedPCMSource()
        with self._lock:
            self._source = source

        def _after(error: Exception | None) -> None:
            if error is not None:
                log.error("playback ended with error: %s", error)

        try:
            voice_client.play(source, after=_after)
        except discord.ClientException as exc:
            # Already playing: the stop() above raced with the sender thread.
            log.warning("could not start playback: %s", exc)
        return source

    def stop(self, voice_client: discord.VoiceClient | None) -> int:
        """Stop playback and discard queued audio. Safe to call when idle."""
        with self._lock:
            source = self._source
            self._source = None

        dropped = source.flush() if source is not None else 0
        if voice_client is not None and voice_client.is_playing():
            # NOT voice_client.stop(). VoiceRecvClient overrides stop() to mean
            # "stop playing AND stop receiving" -- it calls stop_playing() then
            # stop_listening(), which tears down the packet router, the sink
            # event router and the speaking timer. Barge-in calls this on every
            # interruption, so stop() here left the bot connected, still able to
            # speak, and permanently deaf: audio piled up unread in the socket
            # while the receive pipeline was gone. Nothing logged it, because
            # this is a clean shutdown rather than a crash.
            stop_playing = getattr(voice_client, "stop_playing", None)
            if stop_playing is not None:
                stop_playing()
            else:
                voice_client.stop()
        if dropped:
            log.debug("barge-in dropped %d queued frame(s)", dropped)
        return dropped

    def finish(self) -> None:
        source = self._source
        if source is not None:
            source.finish()

    @property
    def source(self) -> QueuedPCMSource | None:
        return self._source
