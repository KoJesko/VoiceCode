"""Scoping must fail closed. These are the tests that matter most."""

from __future__ import annotations

import pytest

from voicecode.config import AllowlistSnapshot
from voicecode.discord_app import scoping


def snap(**kw) -> AllowlistSnapshot:
    return AllowlistSnapshot(
        guilds=frozenset(kw.get("guilds", ())),
        voice_channels=frozenset(kw.get("voice_channels", ())),
        users=frozenset(kw.get("users", ())),
        bindings=dict(kw.get("bindings", {})),
        auto_join=kw.get("auto_join", False),
        revision=kw.get("revision", 0),
    )


EMPTY = snap()
FULL = snap(guilds={1}, voice_channels={2, 9}, users={3}, bindings={2: 5}, auto_join=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: scoping.guild_gate(s, 1),
        lambda s: scoping.join_gate(s, 1, 2),
        lambda s: scoping.interaction_gate(s, 1, 3),
        lambda s: scoping.occupancy_gate(s, 1, 2),
        lambda s: scoping.turn_gate(s, 1, 2, 3),
        lambda s: scoping.mirror_gate(s, 1, 2),
        lambda s: scoping.auto_join_gate(s, 1, 2, 3),
    ],
)
def test_empty_allowlists_deny_everything(call):
    """Empty means deny, not allow-all. This inverts the usual convention."""
    assert not call(EMPTY)


def test_guild_gate():
    assert scoping.guild_gate(FULL, 1)
    assert not scoping.guild_gate(FULL, 99)
    assert not scoping.guild_gate(FULL, None)


def test_join_requires_both_allowlist_and_binding():
    assert scoping.join_gate(FULL, 1, 2)
    # Channel 9 is allowlisted but has no bound text channel.
    decision = scoping.join_gate(FULL, 1, 9)
    assert not decision
    assert "TEXT_CHANNEL_BINDING" in decision.reason


def test_join_names_the_failing_requirement():
    assert "GUILD_ALLOWLIST" in scoping.join_gate(FULL, 99, 2).reason
    assert "VOICE_CHANNEL_ALLOWLIST" in scoping.join_gate(FULL, 1, 77).reason


def test_interaction_gate_checks_user_not_just_guild():
    assert scoping.interaction_gate(FULL, 1, 3)
    assert not scoping.interaction_gate(FULL, 1, 4)


def test_auto_join_requires_allowlisted_user():
    assert scoping.auto_join_gate(FULL, 1, 2, 3)
    assert not scoping.auto_join_gate(FULL, 1, 2, 4)
    off = snap(guilds={1}, voice_channels={2}, users={3}, bindings={2: 5}, auto_join=False)
    assert not scoping.auto_join_gate(off, 1, 2, 3)


def test_turn_gate_rechecks_user_against_current_snapshot():
    """An utterance that began while allowlisted must not be transcribed after removal."""
    assert scoping.turn_gate(FULL, 1, 2, 3)
    revoked = snap(guilds={1}, voice_channels={2}, users=set(), bindings={2: 5})
    assert not scoping.turn_gate(revoked, 1, 2, 3)


class FakeVoiceClient:
    def __init__(self, guild_id, channel_id):
        self.guild = type("G", (), {"id": guild_id})()
        self.channel = type("C", (), {"id": channel_id})()


def test_reload_evicts_connections_the_new_policy_forbids():
    """Without this, hot-reload would be cosmetic."""
    tightened = snap(guilds={1}, voice_channels={9}, users={3}, bindings={9: 5})
    evictions = scoping.connections_to_evict(
        tightened, [FakeVoiceClient(1, 2), FakeVoiceClient(1, 9)]
    )
    assert len(evictions) == 1
    assert "VOICE_CHANNEL_ALLOWLIST" in evictions[0][1]


def test_reload_evicts_when_binding_is_removed():
    unbound = snap(guilds={1}, voice_channels={2}, users={3}, bindings={})
    evictions = scoping.connections_to_evict(unbound, [FakeVoiceClient(1, 2)])
    assert len(evictions) == 1
    assert "bound text channel" in evictions[0][1]
