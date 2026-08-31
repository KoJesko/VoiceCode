from __future__ import annotations

import pytest

from voicecode.config import AllowlistSnapshot, ConfigStore, Settings


def test_id_sets_parse_from_comma_and_space_forms():
    s = Settings(guild_allowlist="1,2 3", user_allowlist="[4, 5]")
    assert s.guild_allowlist == {1, 2, 3}
    assert s.user_allowlist == {4, 5}


def test_blank_allowlist_is_empty_not_unset():
    assert Settings(guild_allowlist="").guild_allowlist == set()
    assert Settings().voice_channel_allowlist == set()


def test_binding_parses_pairs():
    s = Settings(text_channel_binding="10:20,30:40")
    assert s.text_channel_binding == {10: 20, 30: 40}


def test_binding_rejects_malformed_entry():
    with pytest.raises(ValueError):
        Settings(text_channel_binding="10-20")


def test_binding_rejects_self_binding():
    with pytest.raises(ValueError):
        Settings(text_channel_binding="10:10")


def test_occupiable_requires_allowlist_and_binding():
    s = AllowlistSnapshot(frozenset({1}), frozenset({2, 3}), frozenset(), {2: 9}, False, 0)
    assert s.occupiable(2)
    assert not s.occupiable(3)   # allowlisted but unbound
    assert not s.occupiable(4)   # not allowlisted


def test_reload_swaps_the_snapshot_atomically(tmp_path):
    env = tmp_path / ".env"
    env.write_text("GUILD_ALLOWLIST=1\nVOICE_CHANNEL_ALLOWLIST=2\nTEXT_CHANNEL_BINDING=2:5\n")
    store = ConfigStore(env_file=env)
    first = store.snapshot
    assert first.guilds == {1}

    env.write_text("GUILD_ALLOWLIST=7\nVOICE_CHANNEL_ALLOWLIST=8\nTEXT_CHANNEL_BINDING=8:9\n")
    second = store.reload()

    assert second.guilds == {7}
    assert second.revision == first.revision + 1
    # The old snapshot is untouched: a decision in flight stays self-consistent.
    assert first.guilds == {1}
