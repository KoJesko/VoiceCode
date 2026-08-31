"""The voice session: one per connected voice channel, orchestrating the full loop.

    utterance -> ASR -> gate -> bridge -> sanitize -> Kokoro -> playback

Three behaviours here are safety-relevant rather than merely functional:

**Permission latch.** A PERMISSION event stops the session dead. Everything spoken
while latched is mirrored and dropped, with a spoken reminder. The only way out is an
explicit `/approve` or `/deny`. Nothing about the transcript can release it -- in
particular there is no matching of "yes", which is the whole point.

**Rate-limit breaker.** A RATE_LIMIT event opens a circuit for the remaining window.
Turns are refused without contacting Claude at all while it is open, so a rate limit
cannot become a retry storm against the shared subscription pool.

**Barge-in.** Speech start from an allowlisted user flushes playback synchronously.
This runs before the VAD has confirmed anything, because being interrupted late is
worse than a rare false trigger.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field

import discord

from ..asr.base import ASREngine
from ..audio.playback import PlaybackController
from ..audio.resample import discord_frames_from_float
from ..audio.turn import TurnEvent
from ..bridge.base import BridgeEvent, ClaudeBridge, EventKind, PermissionDecision
from ..config import ConfigStore, WakeMode
from ..logging_setup import TurnTimer
from ..speech.sanitize import sanitize_for_speech
from ..tts.kokoro_engine import KokoroTTS, TTSUnavailable
from ..tts.sentences import SentenceStreamer
from .mirror import Mirror
from .scoping import log_refusal, turn_gate

log = logging.getLogger(__name__)


@dataclass
class PendingPermission:
    prompt: str
    tool: str
    options: dict[str, str]
    heuristic: bool
    raised_at: float = field(default_factory=time.time)


@dataclass
class RateLimitState:
    until: float
    detail: str

    @property
    def open(self) -> bool:
        return time.time() < self.until

    def remaining_s(self) -> float:
        return max(0.0, self.until - time.time())


class VoiceSession:
    """Owns one voice connection and everything that happens on it."""

    def __init__(
        self,
        *,
        bot: discord.Client,
        voice_client: discord.VoiceClient,
        config: ConfigStore,
        asr: ASREngine,
        tts: KokoroTTS,
        bridge: ClaudeBridge,
        mirror: Mirror,
    ):
        self.bot = bot
        self.voice_client = voice_client
        self.config = config
        self.asr = asr
        self.tts = tts
        self.bridge = bridge
        self.mirror = mirror

        self.playback = PlaybackController()
        self.sink = None  # set by the bot once the sink is attached
        self._turn_lock = asyncio.Lock()
        self._pending_permission: PendingPermission | None = None
        self._rate_limit: RateLimitState | None = None
        self._ptt_open = False
        self._last_prompt: str | None = None
        self._speaking_task: asyncio.Task | None = None

    # -- identity ----------------------------------------------------------------

    @property
    def guild_id(self) -> int | None:
        return getattr(self.voice_client.guild, "id", None)

    @property
    def channel_id(self) -> int | None:
        return getattr(self.voice_client.channel, "id", None)

    @property
    def pending_permission(self) -> PendingPermission | None:
        return self._pending_permission

    @property
    def rate_limited(self) -> bool:
        return self._rate_limit is not None and self._rate_limit.open

    # -- TurnConsumer ------------------------------------------------------------

    async def on_speech_start(self, user_id: int) -> None:
        """Barge-in. Called the moment the VAD sees speech from an allowlisted user."""
        if self.playback.active:
            dropped = self.playback.stop(self.voice_client)
            log.info("barge-in from %s; dropped %d queued frame(s)", user_id, dropped)

    async def on_utterance(self, event: TurnEvent) -> None:
        utterance = event.utterance
        if utterance is None:
            return

        snapshot = self.config.snapshot
        decision = turn_gate(snapshot, self.guild_id, self.channel_id, utterance.user_id)
        if not decision:
            log_refusal("turn", decision, user=utterance.user_id, channel=self.channel_id)
            return

        timer = TurnTimer(label=f"user:{utterance.user_id}")
        timer.t0 = utterance.ended_at  # the clock starts at end of speech
        timer.mark("endpoint")

        try:
            transcript = await asyncio.to_thread(self.asr.transcribe, utterance.audio)
        except Exception:
            log.exception("transcription failed")
            return
        timer.mark("asr")

        text = transcript.text.strip()
        if not text:
            log.debug("empty transcript for a %.0fms utterance", utterance.duration_ms)
            return

        speaker = self._display_name(utterance.user_id)
        await self.mirror.transcript(self.guild_id, self.channel_id, speaker, text)

        gated = self._apply_wake_gate(text)
        if gated is None:
            log.debug("wake gate suppressed: %r", text)
            return

        await self._handle_prompt(gated, timer)

    # -- gating ------------------------------------------------------------------

    def _apply_wake_gate(self, text: str) -> str | None:
        """Apply the configured wake mode. Returns the prompt, or None to suppress."""
        mode = self.config.settings.wake_mode
        if mode is WakeMode.ALWAYS:
            return text
        if mode is WakeMode.PTT:
            return text if self._ptt_open else None
        if mode is WakeMode.WAKEWORD:
            wake = self.config.settings.wake_word.lower().strip()
            lowered = text.lower()
            if not wake or not lowered.startswith(wake):
                return None
            return text[len(wake) :].lstrip(" ,.:;-") or text
        return text

    def set_ptt(self, open_: bool) -> None:
        self._ptt_open = open_

    # -- the turn ----------------------------------------------------------------

    async def _handle_prompt(self, text: str, timer: TurnTimer) -> None:
        if self._pending_permission is not None:
            await self._remind_permission(text)
            return

        if self.rate_limited:
            assert self._rate_limit is not None
            await self.mirror.notice(
                self.guild_id,
                self.channel_id,
                f"Still rate limited for ~{self._rate_limit.remaining_s() / 60:.0f} min. "
                "Not sending this turn.",
            )
            await self._speak("Still at the usage limit. I did not send that.")
            return

        if self._turn_lock.locked():
            await self._speak("Still working on the last one.")
            return

        async with self._turn_lock:
            self._last_prompt = text
            await self._run_turn(text, timer)

    async def _run_turn(self, text: str, timer: TurnTimer) -> None:
        streamer = SentenceStreamer()
        full_output: list[str] = []
        source = None
        first_prose = True

        try:
            # aclosing matters here: this loop breaks on DONE and returns outright on
            # PERMISSION, RATE_LIMIT and ERROR. Without it the bridge's async generator
            # stays suspended inside its own turn lock until garbage collection gets to
            # it, and the next turn blocks.
            async with contextlib.aclosing(self.bridge.send(text)) as stream:
                async for event in stream:
                    if event.kind is EventKind.PROSE:
                        if first_prose:
                            timer.mark("bridge_first")
                            first_prose = False
                        full_output.append(event.text)
                        for sentence in streamer.feed(event.text + "\n"):
                            source = await self._speak_sentence(sentence, source, timer)

                    elif event.kind is EventKind.RAW:
                        full_output.append(event.text)

                    elif event.kind is EventKind.PERMISSION:
                        await self._raise_permission(event)
                        return

                    elif event.kind is EventKind.RATE_LIMIT:
                        await self._open_breaker(event)
                        return

                    elif event.kind is EventKind.ERROR:
                        log.error("bridge error: %s", event.text)
                        await self.mirror.notice(
                            self.guild_id,
                            self.channel_id,
                            f"Claude Code error: {event.text}",
                        )
                        await self._speak(
                            "Claude Code returned an error. Details are in the channel."
                        )
                        return

                    elif event.kind is EventKind.DONE:
                        turns = event.meta.get("turns")
                        if turns:
                            log.info("turn complete: ~%s Claude Code turn(s) used", turns)
                        break

            for sentence in streamer.flush():
                source = await self._speak_sentence(sentence, source, timer)

        finally:
            if source is not None:
                self.playback.finish()

        combined = "\n".join(part for part in full_output if part.strip())
        if combined.strip():
            await self.mirror.claude_output(self.guild_id, self.channel_id, combined)

        result = sanitize_for_speech(combined, self.config.settings.speak_char_limit)
        if result.truncated:
            await self.mirror.notice(
                self.guild_id,
                self.channel_id,
                "Spoken reply was capped; the full output is above.",
            )
        timer.emit()

    # -- speech ------------------------------------------------------------------

    async def _speak_sentence(self, sentence: str, source, timer: TurnTimer):
        """Sanitize, synthesize, and queue one sentence. Returns the live source."""
        result = sanitize_for_speech(sentence, self.config.settings.speak_char_limit)
        if not result.has_speech:
            return source
        if not self.config.settings.tts_enabled or not self.tts.enabled:
            return source

        try:
            speech = await asyncio.to_thread(self.tts.synthesize, result.spoken)
        except TTSUnavailable as exc:
            log.error("TTS unavailable, continuing text-only: %s", exc)
            await self.mirror.notice(
                self.guild_id, self.channel_id, f"Speech disabled: {exc}"
            )
            return source
        except Exception:
            log.exception("synthesis failed for one sentence; skipping it")
            return source

        if speech.audio.size == 0:
            return source
        timer.mark("tts_first")

        if source is None or source.cancelled:
            if not self._can_play():
                return source
            source = self.playback.start(self.voice_client)

        source.feed(discord_frames_from_float(speech.audio))
        timer.mark("first_frame")
        return source

    async def _speak(self, text: str) -> None:
        """Speak a short bot-generated line (not Claude output)."""
        if not self.config.settings.tts_enabled or not self.tts.enabled or not self._can_play():
            return
        try:
            speech = await asyncio.to_thread(self.tts.synthesize, text)
        except Exception:
            log.exception("could not synthesize a notice")
            return
        if speech.audio.size == 0:
            return
        source = self.playback.start(self.voice_client)
        source.feed(discord_frames_from_float(speech.audio))
        self.playback.finish()

    def _can_play(self) -> bool:
        """Gate 11: re-check occupancy before making sound."""
        from .scoping import occupancy_gate

        decision = occupancy_gate(self.config.snapshot, self.guild_id, self.channel_id)
        if not decision:
            log_refusal("playback", decision, channel=self.channel_id)
            return False
        return self.voice_client.is_connected()

    def stop_playback(self) -> int:
        return self.playback.stop(self.voice_client)

    # -- permission latch --------------------------------------------------------

    async def _raise_permission(self, event: BridgeEvent) -> None:
        self._pending_permission = PendingPermission(
            prompt=str(event.meta.get("prompt") or event.text),
            tool=str(event.meta.get("tool") or "a tool"),
            options=dict(event.meta.get("options") or {}),
            heuristic=event.heuristic,
        )
        detail = self._pending_permission.prompt
        note = (
            "\n\n_Detected from rendered terminal text, so treat it as advisory and "
            "check the session._"
            if event.heuristic
            else ""
        )
        await self.mirror.send(
            self.guild_id,
            self.channel_id,
            f"⛔ **Claude Code is waiting for permission** "
            f"(`{self._pending_permission.tool}`)\n```\n{detail[:1500]}\n```"
            f"Reply with `/approve` or `/deny`. I will not answer this myself.{note}",
        )
        await self._speak(
            f"Claude Code needs permission to use {self._pending_permission.tool}. "
            "I won't answer that myself. Approve or deny it with a slash command."
        )

    async def _remind_permission(self, heard: str) -> None:
        assert self._pending_permission is not None
        await self.mirror.notice(
            self.guild_id,
            self.channel_id,
            f"Heard “{heard}”, but a permission prompt is still open. "
            "Use `/approve` or `/deny` — spoken words are never treated as an answer.",
        )
        await self._speak("There's still a permission prompt open. Use approve or deny.")

    async def resolve_permission(self, decision: PermissionDecision) -> str:
        """Called only from the explicit slash command."""
        if self._pending_permission is None:
            return "No permission prompt is pending."

        pending = self._pending_permission
        self._pending_permission = None
        await self.bridge.respond_to_permission(decision)

        if decision is PermissionDecision.DENY:
            return f"Denied `{pending.tool}`."

        # Approval is one-shot and applies to a retry of the same prompt.
        if self._last_prompt:
            asyncio.create_task(self._retry_after_approval(self._last_prompt))
            return f"Approved `{pending.tool}` for one turn. Retrying your request."
        return f"Approved `{pending.tool}` for one turn."

    async def _retry_after_approval(self, text: str) -> None:
        async with self._turn_lock:
            await self._run_turn(text, TurnTimer(label="retry"))

    # -- rate limit breaker ------------------------------------------------------

    async def _open_breaker(self, event: BridgeEvent) -> None:
        delay_ms = event.meta.get("retry_delay_ms")
        resets_at = event.meta.get("resets_at")

        if isinstance(delay_ms, (int, float)) and delay_ms > 0:
            window = float(delay_ms) / 1000.0
        else:
            # No structured window (the tmux bridge never has one). Back off for a
            # fixed period rather than retrying and burning more of the pool.
            window = 15 * 60.0

        self._rate_limit = RateLimitState(time.time() + window, event.text)
        spoken = (
            f"Usage limit hit. Resets at {resets_at}."
            if resets_at
            else f"Usage limit hit. Backing off for {window / 60:.0f} minutes."
        )
        log.warning("rate limited: %s (backing off %.0fs)", event.text, window)
        await self.mirror.send(
            self.guild_id,
            self.channel_id,
            f"🚦 **Usage limit reached.** {event.text}\n"
            f"Not retrying for {window / 60:.0f} minutes.",
        )
        await self._speak(spoken)

    # -- misc --------------------------------------------------------------------

    def _display_name(self, user_id: int) -> str:
        guild = self.voice_client.guild
        member = guild.get_member(user_id) if guild else None
        return getattr(member, "display_name", None) or str(user_id)

    async def shutdown(self) -> None:
        self.playback.stop(self.voice_client)
        if self.voice_client.is_connected():
            await self.voice_client.disconnect(force=True)
