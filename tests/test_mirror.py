"""The mirror posts untrusted content (tool results), so it must never ping anyone."""

from __future__ import annotations

import discord

from voicecode.discord_app.mirror import Mirror


class FakeChannel:
    id = 1

    def __init__(self):
        self.calls = []

    async def send(self, content, **kwargs):
        self.calls.append((content, kwargs))


async def test_mirror_suppresses_mentions(monkeypatch):
    channel = FakeChannel()
    mirror = Mirror(bot=None, snapshot_provider=lambda: None)
    monkeypatch.setattr(mirror, "_resolve", lambda g, v: channel)
    assert await mirror.claude_output(1, 2, "@everyone <@&123> look at this")
    (content, kwargs), = channel.calls
    allowed = kwargs["allowed_mentions"]
    assert isinstance(allowed, discord.AllowedMentions)
    assert not allowed.everyone and not allowed.users and not allowed.roles
