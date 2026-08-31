"""Channel, guild, and user scoping. Every gate in the bot routes through here.

Design rules, applied without exception:

* **Fail closed.** An empty allowlist denies everything. This inverts the usual
  convention on purpose: a typo or an unset variable should silence the bot, never
  widen it.
* **Decide against a snapshot, not live settings.** Callers pass an AllowlistSnapshot,
  which is immutable and swapped wholesale on reload, so a decision is always
  self-consistent.
* **Re-check, don't cache.** A decision made at join time is not evidence about now.
  Hot-reload is only meaningful if tightening an allowlist evicts existing connections,
  which is what `connections_to_evict` is for.

Gate map (docs/DESIGN.md 4):

  1  gateway events            guild_gate
  2  command registration      guild-scoped sync only, in bot.py
  3  slash command dispatch    interaction_gate
  4  /join                     join_gate
  5  bot's own voice state     occupancy_gate      <- drag-in eviction
  6  other members' state      auto_join_gate
  7  AudioSink.write           snapshot.user_allowed, first statement in sink.py
  8  speaking listeners        snapshot.user_allowed, in sink.py
  9  turn dispatch             turn_gate
  10 mirror write              mirror_gate
  11 playback start            occupancy_gate
  12 post-reload sweep         connections_to_evict
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..config import AllowlistSnapshot

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = ScopeDecision(True)


def _deny(reason: str) -> ScopeDecision:
    return ScopeDecision(False, reason)


# -- 1. Gateway events ----------------------------------------------------------

def guild_gate(snapshot: AllowlistSnapshot, guild_id: int | None) -> ScopeDecision:
    """Gate 1. Every event carrying a guild passes through this first."""
    if guild_id is None:
        return _deny("not in a guild")
    if not snapshot.guilds:
        return _deny("GUILD_ALLOWLIST is empty, so every guild is denied")
    if guild_id not in snapshot.guilds:
        return _deny(f"guild {guild_id} is not in GUILD_ALLOWLIST")
    return ALLOWED


# -- 3. Slash command dispatch --------------------------------------------------

def interaction_gate(
    snapshot: AllowlistSnapshot, guild_id: int | None, user_id: int | None
) -> ScopeDecision:
    """Gate 3. Commands are also registered guild-scoped, but never rely on that alone.

    Registration scope is a Discord-side convenience; this is the check that actually
    holds, and it covers the invoking user as well as the guild.
    """
    decision = guild_gate(snapshot, guild_id)
    if not decision:
        return decision
    if not snapshot.users:
        return _deny("USER_ALLOWLIST is empty, so every user is denied")
    if user_id not in snapshot.users:
        return _deny("you are not in USER_ALLOWLIST")
    return ALLOWED


# -- 4. /join -------------------------------------------------------------------

def join_gate(
    snapshot: AllowlistSnapshot, guild_id: int | None, channel_id: int | None
) -> ScopeDecision:
    """Gate 4. Names which of the two requirements failed, so refusals are actionable."""
    decision = guild_gate(snapshot, guild_id)
    if not decision:
        return decision
    if channel_id is None:
        return _deny("no voice channel given, and you are not in one")
    if not snapshot.voice_channels:
        return _deny("VOICE_CHANNEL_ALLOWLIST is empty, so every channel is denied")
    if channel_id not in snapshot.voice_channels:
        return _deny(f"channel {channel_id} is not in VOICE_CHANNEL_ALLOWLIST")
    if snapshot.bound_text_channel(channel_id) is None:
        return _deny(
            f"channel {channel_id} has no TEXT_CHANNEL_BINDING entry. The bound text "
            "channel is the source of truth for the conversation, so joining without "
            "one is refused."
        )
    return ALLOWED


# -- 5, 11. Occupancy -----------------------------------------------------------

def occupancy_gate(
    snapshot: AllowlistSnapshot, guild_id: int | None, channel_id: int | None
) -> ScopeDecision:
    """Gates 5 and 11: may the bot legitimately be sitting in this channel right now?

    Checked on every voice-state change for the bot's own member, so an admin dragging
    the bot into a non-allowlisted channel is disconnected immediately, and before each
    playback start.
    """
    decision = guild_gate(snapshot, guild_id)
    if not decision:
        return decision
    if not snapshot.occupiable(channel_id):
        if not snapshot.voice_channel_allowed(channel_id):
            return _deny(f"channel {channel_id} is not in VOICE_CHANNEL_ALLOWLIST")
        return _deny(f"channel {channel_id} has no bound text channel")
    return ALLOWED


# -- 6. Auto-join ---------------------------------------------------------------

def auto_join_gate(
    snapshot: AllowlistSnapshot,
    guild_id: int | None,
    channel_id: int | None,
    user_id: int | None,
) -> ScopeDecision:
    """Gate 6. Auto-join needs the channel occupiable AND the user allowlisted."""
    if not snapshot.auto_join:
        return _deny("AUTO_JOIN is off")
    if not snapshot.user_allowed(user_id):
        return _deny("user is not in USER_ALLOWLIST")
    return occupancy_gate(snapshot, guild_id, channel_id)


# -- 9. Turn dispatch -----------------------------------------------------------

def turn_gate(
    snapshot: AllowlistSnapshot,
    guild_id: int | None,
    channel_id: int | None,
    user_id: int | None,
) -> ScopeDecision:
    """Gate 9. Re-validates a finished utterance against the *current* snapshot.

    The sink already gated this user when the audio arrived, but a reload may have
    landed in between -- an utterance that started while someone was allowlisted must
    not be transcribed after they were removed.
    """
    if not snapshot.user_allowed(user_id):
        return _deny("user is no longer in USER_ALLOWLIST")
    return occupancy_gate(snapshot, guild_id, channel_id)


# -- 10. Mirror -----------------------------------------------------------------

def mirror_gate(
    snapshot: AllowlistSnapshot, guild_id: int | None, voice_channel_id: int | None
) -> ScopeDecision:
    """Gate 10. The target is resolved at write time, never cached from join time."""
    decision = guild_gate(snapshot, guild_id)
    if not decision:
        return decision
    if snapshot.bound_text_channel(voice_channel_id) is None:
        return _deny(f"no TEXT_CHANNEL_BINDING for voice channel {voice_channel_id}")
    return ALLOWED


# -- 12. Post-reload sweep ------------------------------------------------------

def connections_to_evict(
    snapshot: AllowlistSnapshot, voice_clients: Iterable[Any]
) -> list[tuple[Any, str]]:
    """Gate 12. Which live connections the new snapshot no longer permits.

    Without this, hot-reload would be cosmetic: removing a channel from the allowlist
    would not remove the bot from it.
    """
    evictions: list[tuple[Any, str]] = []
    for client in voice_clients:
        guild = getattr(client, "guild", None)
        channel = getattr(client, "channel", None)
        decision = occupancy_gate(
            snapshot,
            getattr(guild, "id", None),
            getattr(channel, "id", None),
        )
        if not decision:
            evictions.append((client, decision.reason))
    return evictions


def log_refusal(gate: str, decision: ScopeDecision, **context: Any) -> None:
    """Uniform refusal logging, so scope denials are greppable."""
    details = " ".join(
        f"{key}={value}" for key, value in context.items() if value is not None
    )
    suffix = f" ({details})" if details else ""
    log.warning("scope refusal [%s]: %s%s", gate, decision.reason, suffix)
