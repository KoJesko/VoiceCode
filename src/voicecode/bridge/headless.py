"""Headless bridge: `claude -p ... --output-format stream-json`.

Default bridge. Unlike the tmux bridge it reads structured JSON rather than rendered
terminal text, which is what makes permission handling and rate-limit detection
trustworthy rather than guessed.

Flags, verified against the current CLI reference:
  -p / --print                one-shot, non-interactive
  --output-format stream-json newline-delimited JSON events
  --verbose                   required alongside stream-json
  --resume <session_id>       continue the same conversation across turns
  --permission-mode <mode>    -p starts in Manual on every plan

Deliberately NOT passed: --bare (breaks subscription auth -- see bridge/auth.py) and
--dangerously-skip-permissions (the whole point of the permission handling below).
--include-partial-messages is also omitted: token-level deltas are worse for us than
whole assistant messages, because synthesis wants clauses, not tokens.

Permissions in -p mode
----------------------
There is no interactive prompt to answer: the CLI denies the tool and Claude sees the
denial. So a "permission prompt" surfaces here as a denied tool_result. The tool's
identity and input come from the matching tool_use block, which is structured JSON --
only the is-this-a-denial classification reads text. Approval is one-shot: it allows
that single tool for one retry of the same prompt, and is cleared afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
from collections.abc import AsyncIterator
from typing import Any

from .auth import AuthInfo, check_forbidden_flags, build_subprocess_env, detect_auth
from .base import (
    AuthError,
    BridgeError,
    BridgeEvent,
    BridgeHealth,
    EventKind,
    PermissionDecision,
)

log = logging.getLogger(__name__)

# Substrings that mark a tool_result as a permission refusal rather than a tool failure.
_DENIAL_MARKERS = (
    "permission to use",
    "requested permissions",
    "permission denied",
    "user has not granted",
    "not allowed to use",
    "requires approval",
    "haven't granted you permission",
)
_LIMIT_MARKERS = (
    "usage limit reached",
    "rate limit",
    "you've reached your usage limit",
    "resets at",
)

_MAX_LINE_BYTES = 8 * 1024 * 1024


class HeadlessBridge:
    kind = "headless"

    def __init__(
        self,
        claude_binary: str = "claude",
        cwd: str | None = None,
        permission_mode: str = "default",
    ):
        self._binary = claude_binary
        self._cwd = cwd or None
        self._permission_mode = permission_mode
        self._session_id: str | None = None
        self._turns = 0
        self._auth: AuthInfo | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._one_shot_tools: set[str] = set()
        self._pending_tool: str | None = None
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        self._auth = await detect_auth(self._binary)

    async def close(self) -> None:
        await self._terminate()

    async def health(self) -> BridgeHealth:
        return BridgeHealth(
            kind=self.kind,
            auth_method=self._auth.describe() if self._auth else "not verified",
            alive=True,
            session_id=self._session_id,
            turns=self._turns,
            structured_permissions=True,
            structured_rate_limits=True,
        )

    # -- turn --------------------------------------------------------------------

    def _build_argv(self, text: str) -> list[str]:
        argv = [
            self._binary,
            "-p",
            text,
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if self._session_id:
            argv += ["--resume", self._session_id]
        if self._permission_mode and self._permission_mode != "default":
            argv += ["--permission-mode", self._permission_mode]
        if self._one_shot_tools:
            argv += ["--allowedTools", ",".join(sorted(self._one_shot_tools))]
        check_forbidden_flags(argv)
        return argv

    async def send(self, text: str) -> AsyncIterator[BridgeEvent]:
        async with self._lock:
            argv = self._build_argv(text)
            # One-shot approvals apply to exactly one invocation.
            self._one_shot_tools.clear()
            self._pending_tool = None

            log.debug("headless turn: %s", " ".join(argv[:2] + ["<text>"] + argv[3:]))
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._cwd,
                    env=build_subprocess_env(),
                    limit=_MAX_LINE_BYTES,
                )
            except OSError as exc:
                yield BridgeEvent(EventKind.ERROR, f"could not start {self._binary}: {exc}")
                return

            self._process = process
            self._turns += 1
            tool_names: dict[str, str] = {}

            try:
                assert process.stdout is not None
                async for line in process.stdout:
                    for event in self._parse_line(line, tool_names):
                        yield event
            except asyncio.CancelledError:
                await self._terminate()
                raise
            finally:
                stderr = b""
                if process.stderr is not None:
                    with contextlib.suppress(Exception):
                        stderr = await process.stderr.read()
                code = await process.wait()
                self._process = None

            if code not in (0, None):
                detail = stderr.decode("utf-8", "replace").strip()
                yield BridgeEvent(
                    EventKind.ERROR,
                    detail or f"claude exited with code {code}",
                    {"exit_code": code},
                )

    # -- stream parsing ----------------------------------------------------------

    def _parse_line(self, raw: bytes, tool_names: dict[str, str]) -> list[BridgeEvent]:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            return []
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log.debug("non-JSON line on the stream: %.200s", line)
            return []
        if not isinstance(payload, dict):
            return []

        kind = payload.get("type")
        if kind == "system":
            return self._parse_system(payload)
        if kind == "assistant":
            return self._parse_assistant(payload, tool_names)
        if kind == "user":
            return self._parse_user(payload, tool_names)
        if kind == "result":
            return self._parse_result(payload)
        return []

    def _parse_system(self, payload: dict[str, Any]) -> list[BridgeEvent]:
        subtype = payload.get("subtype")
        if subtype == "init":
            session_id = payload.get("session_id")
            if session_id:
                self._session_id = str(session_id)
            return []
        if subtype == "api_retry":
            error = str(payload.get("error", ""))
            if error == "rate_limit":
                delay_ms = payload.get("retry_delay_ms")
                return [
                    BridgeEvent(
                        EventKind.RATE_LIMIT,
                        "Usage limit reached on your Claude subscription.",
                        {
                            "retry_delay_ms": delay_ms,
                            "attempt": payload.get("attempt"),
                            "max_retries": payload.get("max_retries"),
                            "structured": True,
                        },
                    )
                ]
            return [
                BridgeEvent(
                    EventKind.RAW,
                    f"[retrying after {error}]",
                    {"subtype": "api_retry", "error": error},
                )
            ]
        return []

    def _parse_assistant(
        self, payload: dict[str, Any], tool_names: dict[str, str]
    ) -> list[BridgeEvent]:
        # Subagent output carries a parent_tool_use_id; only the main conversation
        # is spoken, or the bot narrates its own internal fan-out.
        from_subagent = payload.get("parent_tool_use_id") is not None
        message = payload.get("message") or {}
        events: list[BridgeEvent] = []

        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = (block.get("text") or "").strip()
                if not text:
                    continue
                events.append(
                    BridgeEvent(
                        EventKind.RAW if from_subagent else EventKind.PROSE, text
                    )
                )
            elif block_type == "tool_use":
                name = str(block.get("name", "tool"))
                tool_id = block.get("id")
                if tool_id:
                    tool_names[str(tool_id)] = name
                events.append(
                    BridgeEvent(
                        EventKind.RAW,
                        f"[{name}]",
                        {"tool": name, "input": block.get("input")},
                    )
                )
        return events

    def _parse_user(
        self, payload: dict[str, Any], tool_names: dict[str, str]
    ) -> list[BridgeEvent]:
        message = payload.get("message") or {}
        events: list[BridgeEvent] = []

        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _flatten_content(block.get("content"))
            tool = tool_names.get(str(block.get("tool_use_id")), "a tool")
            lowered = text.lower()

            if block.get("is_error") and any(m in lowered for m in _DENIAL_MARKERS):
                self._pending_tool = tool
                events.append(
                    BridgeEvent(
                        EventKind.PERMISSION,
                        f"Claude Code needs permission to use {tool}.",
                        {
                            "tool": tool,
                            "prompt": text,
                            # The tool identity is structural; only the is-this-a-denial
                            # classification reads the result text.
                            "detection": "tool_result",
                        },
                    )
                )
                continue

            if any(m in lowered for m in _LIMIT_MARKERS):
                events.append(
                    BridgeEvent(EventKind.RATE_LIMIT, text, {"structured": False})
                )
                continue

            events.append(BridgeEvent(EventKind.RAW, text, {"tool": tool}))
        return events

    def _parse_result(self, payload: dict[str, Any]) -> list[BridgeEvent]:
        session_id = payload.get("session_id")
        if session_id:
            self._session_id = str(session_id)

        text = payload.get("result")
        result_text = text if isinstance(text, str) else ""
        lowered = result_text.lower()

        if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
            if any(m in lowered for m in _LIMIT_MARKERS):
                return [
                    BridgeEvent(
                        EventKind.RATE_LIMIT, result_text, {"structured": False}
                    )
                ]
            return [
                BridgeEvent(
                    EventKind.ERROR,
                    result_text or str(payload.get("subtype", "run failed")),
                    {"subtype": payload.get("subtype")},
                )
            ]

        return [
            BridgeEvent(
                EventKind.DONE,
                "",
                {
                    "session_id": self._session_id,
                    "turns": payload.get("num_turns"),
                    "duration_ms": payload.get("duration_ms"),
                },
            )
        ]

    # -- control -----------------------------------------------------------------

    async def interrupt(self) -> None:
        """SIGINT ends the current turn but leaves the session resumable."""
        process = self._process
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGINT)
            log.info("sent SIGINT to the in-flight claude turn")

    async def respond_to_permission(self, decision: PermissionDecision) -> None:
        if decision is PermissionDecision.DENY:
            self._pending_tool = None
            return
        if self._pending_tool is None:
            raise BridgeError("no permission request is pending")
        # One-shot: allowed for the next invocation only, then cleared in send().
        self._one_shot_tools.add(self._pending_tool)
        log.warning(
            "one-shot approval granted for tool %r (next turn only)", self._pending_tool
        )
        self._pending_tool = None

    @property
    def pending_tool(self) -> str | None:
        return self._pending_tool

    async def _terminate(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (TimeoutError, asyncio.TimeoutError):
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        self._process = None


def _flatten_content(content: Any) -> str:
    """tool_result content is a string on some versions and a block list on others."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content).strip()


def ensure_subscription_auth(info: AuthInfo) -> None:
    """Re-assert the auth contract at call sites that did not run detect_auth."""
    if not info.method.acceptable:
        raise AuthError(f"unacceptable auth method: {info.method.value}")
