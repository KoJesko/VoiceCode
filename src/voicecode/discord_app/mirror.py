"""Transcript mirroring to the bound text channel.

That channel is the source of truth: everything the bot hears and everything Claude
says lands there in full, untruncated and unsanitised. Voice is the summary. So when
speech is capped at 600 characters or a code block is collapsed to "12 lines of
Python", the real content is still one glance away.

The binding is resolved at write time from the current snapshot, never cached from
join time -- gate 10.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import discord

from ..config import AllowlistSnapshot
from .scoping import log_refusal, mirror_gate

log = logging.getLogger(__name__)

DISCORD_MESSAGE_LIMIT = 2000
# Leave room for the code-fence wrapper when chunking.
_CHUNK = DISCORD_MESSAGE_LIMIT - 12


class Mirror:
    """Writes to whichever text channel is currently bound to a voice channel."""

    def __init__(self, bot: Any, snapshot_provider):
        self._bot = bot
        self._snapshot: Callable[[], AllowlistSnapshot] = snapshot_provider

    def _resolve(self, guild_id: int | None, voice_channel_id: int | None):
        snapshot = self._snapshot()
        decision = mirror_gate(snapshot, guild_id, voice_channel_id)
        if not decision:
            log_refusal("mirror", decision, guild=guild_id, voice=voice_channel_id)
            return None

        text_channel_id = snapshot.bound_text_channel(voice_channel_id)
        channel = self._bot.get_channel(text_channel_id)
        if channel is None:
            log.error(
                "bound text channel %s is not visible to the bot (missing permissions, "
                "or the ID is wrong)",
                text_channel_id,
            )
            return None
        return channel

    async def send(
        self, guild_id: int | None, voice_channel_id: int | None, content: str
    ) -> bool:
        channel = self._resolve(guild_id, voice_channel_id)
        if channel is None or not content.strip():
            return False

        for chunk in _chunks(content):
            try:
                await channel.send(chunk)
            except discord.HTTPException as exc:
                log.error("failed to mirror to channel %s: %s", channel.id, exc)
                return False
        return True

    async def transcript(
        self, guild_id: int | None, voice_channel_id: int | None, speaker: str, text: str
    ) -> bool:
        return await self.send(guild_id, voice_channel_id, f"🎙️ **{speaker}:** {text}")

    async def claude_output(
        self, guild_id: int | None, voice_channel_id: int | None, text: str
    ) -> bool:
        return await self.send(guild_id, voice_channel_id, text)

    async def notice(
        self, guild_id: int | None, voice_channel_id: int | None, text: str
    ) -> bool:
        return await self.send(guild_id, voice_channel_id, f"⚙️ {text}")


def _chunks(content: str) -> list[str]:
    """Split on line boundaries where possible, so code blocks stay readable."""
    if len(content) <= DISCORD_MESSAGE_LIMIT:
        return [content]

    out: list[str] = []
    current = ""
    for line in content.split("\n"):
        while len(line) > _CHUNK:
            if current:
                out.append(current)
                current = ""
            out.append(line[:_CHUNK])
            line = line[_CHUNK:]
        if len(current) + len(line) + 1 > _CHUNK:
            out.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        out.append(current)
    return out
