"""The Discord client: connection lifecycle, voice-state handling, command scoping."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import voice_recv

from ..asr.base import ASREngine
from ..audio.dave import DaveDecryptor
from ..audio.sink import VoiceCodeSink
from ..bridge.base import ClaudeBridge
from ..config import ConfigStore, describe_scope
from ..tts.kokoro_engine import KokoroTTS
from .mirror import Mirror
from .scoping import (
    auto_join_gate,
    connections_to_evict,
    guild_gate,
    join_gate,
    log_refusal,
    occupancy_gate,
)
from .session import VoiceSession

log = logging.getLogger(__name__)


def build_intents() -> discord.Intents:
    """The minimum this bot needs.

    `members` is privileged and must be enabled in the Discord developer portal: it is
    required to resolve a speaking SSRC to a Member, which every allowlist check
    depends on. `message_content` is deliberately not requested -- the bot reads no
    message text.
    """
    intents = discord.Intents.none()
    intents.guilds = True
    intents.voice_states = True
    intents.members = True
    return intents


class VoiceCodeBot(discord.Client):
    def __init__(
        self,
        *,
        config: ConfigStore,
        asr: ASREngine,
        tts: KokoroTTS,
        bridge: ClaudeBridge,
    ):
        super().__init__(intents=build_intents())
        self.config = config
        self.asr = asr
        self.tts = tts
        self.bridge = bridge
        self.tree = app_commands.CommandTree(self)
        self.mirror = Mirror(self, lambda: self.config.snapshot)
        self.sessions: dict[int, VoiceSession] = {}

    # -- lifecycle ---------------------------------------------------------------

    async def setup_hook(self) -> None:
        from .commands import register_commands

        register_commands(self)

        # Gate 2: commands are registered per-guild, never globally. A global sync
        # would publish them to every server the bot is in, including ones outside
        # the allowlist.
        snapshot = self.config.snapshot
        if not snapshot.guilds:
            log.error(
                "GUILD_ALLOWLIST is empty; no commands will be registered and every "
                "event will be ignored. This is the fail-closed default -- set "
                "GUILD_ALLOWLIST in .env."
            )
            return

        for guild_id in sorted(snapshot.guilds):
            guild = discord.Object(id=guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            log.info("registered slash commands for guild %s", guild_id)

    async def on_ready(self) -> None:
        log.info("connected as %s (%s)", self.user, getattr(self.user, "id", "?"))
        log.info("scope: %s", describe_scope(self.config.snapshot))

    async def close(self) -> None:
        for session in list(self.sessions.values()):
            await session.shutdown()
        self.sessions.clear()
        await self.bridge.close()
        await super().close()

    # -- voice connection --------------------------------------------------------

    async def join_channel(self, channel: discord.VoiceChannel) -> VoiceSession:
        """Connect and start listening. Assumes join_gate has already passed."""
        guild_id = channel.guild.id
        existing = self.sessions.pop(guild_id, None)
        if existing is not None:
            await existing.shutdown()

        voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
        session = VoiceSession(
            bot=self,
            voice_client=voice_client,
            config=self.config,
            asr=self.asr,
            tts=self.tts,
            bridge=self.bridge,
            mirror=self.mirror,
        )
        settings = self.config.settings
        sink = VoiceCodeSink(
            loop=self.loop,
            snapshot_provider=lambda: self.config.snapshot,
            decryptor=DaveDecryptor(voice_client),
            consumer=session,
            endpoint_silence_ms=settings.endpoint_silence_ms,
            min_utterance_ms=settings.min_utterance_ms,
            max_utterance_ms=settings.max_utterance_ms,
            vad_threshold=settings.vad_threshold,
        )
        voice_client.listen(sink)
        # Endpoints a turn when the packet flow stops rather than goes quiet, which
        # is what push-to-talk release looks like from here.
        sink.start_endpoint_watchdog()
        session.sink = sink
        session.start_idle_watch()
        self.sessions[guild_id] = session

        log.info("joined voice channel %s (%s)", channel.name, channel.id)
        await self.mirror.notice(
            guild_id, channel.id, f"Connected to **{channel.name}**. Listening."
        )
        return session

    async def leave_guild_voice(self, guild_id: int, reason: str = "") -> bool:
        session = self.sessions.pop(guild_id, None)
        if session is None:
            return False
        channel_id = session.channel_id
        await session.shutdown()
        log.info("left voice in guild %s%s", guild_id, f": {reason}" if reason else "")
        if reason:
            await self.mirror.notice(guild_id, channel_id, f"Disconnected: {reason}")
        return True

    def session_for(self, guild_id: int | None) -> VoiceSession | None:
        return self.sessions.get(guild_id) if guild_id is not None else None

    # -- events ------------------------------------------------------------------

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        guild = member.guild
        snapshot = self.config.snapshot

        # Gate 1: ignore everything outside the allowlisted guilds.
        if not guild_gate(snapshot, getattr(guild, "id", None)):
            return

        if self.user is not None and member.id == self.user.id:
            await self._handle_own_voice_state(member, after)
            return

        if snapshot.auto_join:
            await self._handle_auto_join(member, before, after)

    async def _handle_own_voice_state(
        self, member: discord.Member, after: discord.VoiceState
    ) -> None:
        """Gate 5: an admin dragging the bot somewhere it may not be.

        Discord lets a user with Move Members drag a bot into any voice channel. The
        bot must treat arriving in a non-allowlisted channel exactly like being asked
        to join one: refuse, and leave immediately.
        """
        guild_id = member.guild.id
        channel = after.channel
        if channel is None:
            if self.sessions.pop(guild_id, None) is not None:
                log.info("disconnected from voice in guild %s", guild_id)
            return

        decision = occupancy_gate(self.config.snapshot, guild_id, channel.id)
        if decision:
            return

        log.warning(
            "bot was moved into voice channel %s (%s) in guild %s, which it may not "
            "occupy: %s. Disconnecting.",
            channel.name,
            channel.id,
            guild_id,
            decision.reason,
        )
        log_refusal("occupancy", decision, guild=guild_id, channel=channel.id)

        session = self.sessions.pop(guild_id, None)
        if session is not None:
            await session.shutdown()
        else:
            voice_client = member.guild.voice_client
            if voice_client is not None:
                await voice_client.disconnect(force=True)

    async def _handle_auto_join(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Gate 6. Join when an allowlisted user enters; leave when the last one goes."""
        guild_id = member.guild.id

        if after.channel is not None and after.channel != before.channel:
            decision = auto_join_gate(
                self.config.snapshot, guild_id, after.channel.id, member.id
            )
            if decision and guild_id not in self.sessions:
                try:
                    await self.join_channel(after.channel)  # type: ignore[arg-type]
                except Exception:
                    log.exception("auto-join failed for channel %s", after.channel.id)
            return

        if before.channel is not None and after.channel != before.channel:
            session = self.sessions.get(guild_id)
            if session is None or session.channel_id != before.channel.id:
                return
            snapshot = self.config.snapshot
            remaining = [
                m
                for m in before.channel.members
                if not m.bot and snapshot.user_allowed(m.id)
            ]
            if not remaining:
                await self.leave_guild_voice(guild_id, "last allowlisted user left")

    # -- hot reload ---------------------------------------------------------------

    async def reload_config(self) -> str:
        """Reload allowlists and sweep live connections (gate 12).

        Without the sweep, hot-reload would be cosmetic: removing a channel from the
        allowlist would not remove the bot from it.
        """
        snapshot = self.config.reload()
        evicted = connections_to_evict(snapshot, list(self.voice_clients))

        for voice_client, reason in evicted:
            guild_id = getattr(getattr(voice_client, "guild", None), "id", None)
            log.warning("evicting voice connection in guild %s: %s", guild_id, reason)
            if guild_id is not None:
                await self.leave_guild_voice(guild_id, f"config reload: {reason}")
            else:  # pragma: no cover
                await voice_client.disconnect(force=True)

        for guild_id in sorted(snapshot.guilds):
            guild = discord.Object(id=guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)

        summary = describe_scope(snapshot)
        if evicted:
            summary += f" | evicted {len(evicted)} connection(s)"
        return summary

    async def join_if_allowed(
        self, guild_id: int | None, channel: discord.VoiceChannel | None
    ) -> tuple[bool, str]:
        """Gate 4, shared by /join and any other join path."""
        decision = join_gate(
            self.config.snapshot, guild_id, getattr(channel, "id", None)
        )
        if not decision:
            log_refusal("join", decision, guild=guild_id, channel=getattr(channel, "id", None))
            return False, decision.reason
        assert channel is not None
        await self.join_channel(channel)
        return True, f"Joined {channel.name}."
