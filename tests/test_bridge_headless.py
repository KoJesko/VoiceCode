"""Parsing the stream-json event stream."""

from __future__ import annotations

import json

import pytest

from voicecode.bridge.base import BridgeError, EventKind, PermissionDecision
from voicecode.bridge.headless import HeadlessBridge


def parse(bridge, obj, names=None):
    return bridge._parse_line((json.dumps(obj) + "\n").encode(), names if names is not None else {})


def test_init_captures_the_session_id():
    bridge = HeadlessBridge()
    parse(bridge, {"type": "system", "subtype": "init", "session_id": "s1"})
    assert bridge._session_id == "s1"
    assert "--resume" in bridge._build_argv("hello")


def test_assistant_text_is_prose():
    events = parse(
        HeadlessBridge(),
        {"type": "assistant", "parent_tool_use_id": None,
         "message": {"content": [{"type": "text", "text": "Done."}]}},
    )
    assert [e.kind for e in events] == [EventKind.PROSE]


def test_tool_use_is_raw_not_prose():
    """Tool calls are mirrored but never spoken."""
    events = parse(
        HeadlessBridge(),
        {"type": "assistant", "parent_tool_use_id": None,
         "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                  "input": {"command": "ls"}}]}},
    )
    assert [e.kind for e in events] == [EventKind.RAW]
    assert events[0].meta["tool"] == "Bash"


def test_subagent_text_is_not_spoken():
    """Otherwise the bot narrates its own internal fan-out."""
    events = parse(
        HeadlessBridge(),
        {"type": "assistant", "parent_tool_use_id": "t9",
         "message": {"content": [{"type": "text", "text": "chatter"}]}},
    )
    assert [e.kind for e in events] == [EventKind.RAW]


def test_denied_tool_becomes_a_permission_event():
    bridge = HeadlessBridge()
    names = {}
    parse(bridge, {"type": "assistant", "parent_tool_use_id": None,
                   "message": {"content": [{"type": "tool_use", "id": "t1",
                                            "name": "Bash", "input": {}}]}}, names)
    events = parse(bridge, {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
         "content": "Claude requested permissions to use Bash, but you haven't granted it."}
    ]}}, names)
    assert [e.kind for e in events] == [EventKind.PERMISSION]
    assert events[0].meta["tool"] == "Bash"
    assert bridge.pending_tool == "Bash"


def test_ordinary_tool_error_is_not_a_permission_event():
    bridge = HeadlessBridge()
    events = parse(bridge, {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
         "content": "ENOENT: no such file"}
    ]}})
    assert [e.kind for e in events] == [EventKind.RAW]
    assert bridge.pending_tool is None


def test_api_retry_rate_limit_is_structured():
    events = parse(HeadlessBridge(), {
        "type": "system", "subtype": "api_retry", "error": "rate_limit",
        "attempt": 1, "max_retries": 3, "retry_delay_ms": 42000,
    })
    assert [e.kind for e in events] == [EventKind.RATE_LIMIT]
    assert events[0].meta["retry_delay_ms"] == 42000
    assert events[0].meta["structured"] is True
    assert not events[0].heuristic


def test_non_rate_limit_retry_is_not_a_rate_limit():
    events = parse(HeadlessBridge(), {
        "type": "system", "subtype": "api_retry", "error": "overloaded", "attempt": 1,
    })
    assert [e.kind for e in events] == [EventKind.RAW]


def test_result_is_done_with_metadata():
    events = parse(HeadlessBridge(), {
        "type": "result", "subtype": "success", "result": "ok",
        "session_id": "s1", "num_turns": 4,
    })
    assert [e.kind for e in events] == [EventKind.DONE]
    assert events[0].meta["turns"] == 4


def test_error_result_is_an_error():
    events = parse(HeadlessBridge(), {
        "type": "result", "subtype": "error_during_execution", "is_error": True,
        "result": "it broke",
    })
    assert [e.kind for e in events] == [EventKind.ERROR]


def test_malformed_line_is_ignored():
    assert HeadlessBridge()._parse_line(b"not json\n", {}) == []
    assert HeadlessBridge()._parse_line(b"\n", {}) == []


def test_argv_never_contains_bare():
    """--bare would break subscription auth; this is the guard."""
    assert "--bare" not in HeadlessBridge()._build_argv("hello")


def test_argv_has_the_flags_stream_json_requires():
    argv = HeadlessBridge()._build_argv("hello")
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv  # stream-json requires it


async def test_approval_is_one_shot():
    bridge = HeadlessBridge()
    bridge._pending_tool = "Bash"
    await bridge.respond_to_permission(PermissionDecision.APPROVE)
    assert "--allowedTools" in bridge._build_argv("go")
    bridge._one_shot_tools.clear()  # send() does this after building argv
    assert "--allowedTools" not in bridge._build_argv("next")


async def test_denial_clears_without_granting():
    bridge = HeadlessBridge()
    bridge._pending_tool = "Bash"
    await bridge.respond_to_permission(PermissionDecision.DENY)
    assert bridge.pending_tool is None
    assert "--allowedTools" not in bridge._build_argv("go")


async def test_approving_nothing_raises():
    with pytest.raises(BridgeError):
        await HeadlessBridge().respond_to_permission(PermissionDecision.APPROVE)
