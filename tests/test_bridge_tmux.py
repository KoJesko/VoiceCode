"""The tmux bridge reads rendered text, so its job is mostly knowing what to ignore."""

from __future__ import annotations

import pytest

from voicecode.bridge.base import BridgeError, EventKind, PermissionDecision
from voicecode.bridge.tmux import TmuxBridge, _is_furniture


@pytest.fixture
def bridge():
    return TmuxBridge(session="dev")


def test_session_name_is_required():
    with pytest.raises(BridgeError, match="TMUX_SESSION"):
        TmuxBridge(session="")


@pytest.mark.parametrize(
    "line",
    [
        "╭──────────────╮",
        "│ >             │",
        "⠋ Thinking…",
        "  ? for shortcuts · esc to interrupt",
        "  (2,481 tokens)",
    ],
)
def test_furniture_is_recognised(line):
    assert _is_furniture(line)


@pytest.mark.parametrize("line", ["I fixed the bug.", "The test passes now."])
def test_prose_is_not_furniture(line):
    assert not _is_furniture(line)


def test_prose_lines_are_merged_into_one_event(bridge):
    events = bridge._classify(["I found the bug.", "It was in the resampler."])
    assert [e.kind for e in events] == [EventKind.PROSE]
    assert "resampler" in events[0].text


def test_permission_prompt_is_detected_and_marked_heuristic(bridge):
    events = bridge._classify([
        "Do you want to run this command?",
        "  1) Yes, allow this once",
        "  2) No, tell Claude what to do differently",
    ])
    assert [e.kind for e in events] == [EventKind.PERMISSION]
    assert events[0].heuristic
    assert events[0].meta["options"] == {
        "1": "Yes, allow this once",
        "2": "No, tell Claude what to do differently",
    }


def test_permission_prompt_is_not_also_spoken_as_prose(bridge):
    """The prompt travels in meta; emitting it twice would read the question aloud twice."""
    events = bridge._classify([
        "Do you want to run this command?",
        "  1) Yes, allow this once",
        "  2) No, cancel",
    ])
    assert EventKind.PROSE not in [e.kind for e in events]


def test_numbered_list_without_a_prompt_stays_prose(bridge):
    events = bridge._classify(["I did three things:", "  1) read the file", "  2) fixed it"])
    assert [e.kind for e in events] == [EventKind.PROSE]


def test_rate_limit_is_detected_with_a_reset_time(bridge):
    events = bridge._classify(["You've reached your usage limit. Limit resets at 3:00pm."])
    assert [e.kind for e in events] == [EventKind.RATE_LIMIT]
    assert events[0].meta["resets_at"] == "3:00pm"
    assert events[0].heuristic


async def test_permission_response_matches_the_printed_options(bridge):
    bridge._pending_options = {"1": "Yes, allow this once", "2": "No, cancel"}
    sent = []
    async def fake_run(*argv):
        sent.append(argv)
        return 0, "", ""
    bridge._run = fake_run
    await bridge.respond_to_permission(PermissionDecision.APPROVE)
    assert any("1" in argv for argv in sent)


async def test_permission_response_refuses_to_guess(bridge):
    """If the intended option is not on screen, refuse rather than press a key."""
    bridge._pending_options = {"1": "Something unrelated"}
    with pytest.raises(BridgeError, match="could not match"):
        await bridge.respond_to_permission(PermissionDecision.APPROVE)


async def test_permission_response_without_a_prompt_raises(bridge):
    with pytest.raises(BridgeError):
        await bridge.respond_to_permission(PermissionDecision.APPROVE)


async def test_health_reports_degraded_detection(bridge):
    async def fake_run(*argv):
        return 0, "", ""
    bridge._run = fake_run
    health = await bridge.health()
    assert not health.structured_permissions
    assert not health.structured_rate_limits
    assert "heuristic" in health.describe()
