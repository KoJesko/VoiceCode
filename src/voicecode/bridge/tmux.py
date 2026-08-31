"""tmux bridge: inject into an existing interactive Claude Code session.

Selected with CLAUDE_BRIDGE=tmux. Its appeal is that it drives the session you are
already looking at; its cost is that it only ever sees rendered terminal text, so
permission prompts and rate limits are recognised heuristically rather than
structurally. Those events carry meta["heuristic"] = True and /status reports the
bridge as degraded on both.

Two details that bite:

* `send-keys` without `-l` interprets its argument as key names. A transcript
  containing "C-c" or "Escape" would then be delivered as a control key rather than
  text. Every text injection here uses `-l`, and Enter is sent as a separate call.

* The tmux server does not inherit our scrubbed environment -- it was started long
  before this bot, with whatever the user's shell had. So scrubbing our own subprocess
  env does nothing for the `claude` process running inside it. `start()` therefore
  inspects the session's environment directly and refuses if an API credential is
  present there.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator

from .auth import SCRUBBED_VARS, build_subprocess_env, detect_auth
from .base import (
    AuthError,
    BridgeError,
    BridgeEvent,
    BridgeHealth,
    EventKind,
    PermissionDecision,
)

log = logging.getLogger(__name__)

# Terminal furniture that should never be treated as Claude's answer.
_PROMPT_LINE = re.compile(r"^\s*[│|]?\s*[>❯]\s*$|^\s*[│|]\s*>\s")
_BOX_LINE = re.compile(r"^\s*[─-╿]")
_SPINNER_LINE = re.compile(r"^\s*[⠀-⣿●○*+x]\s")
_HINT_LINE = re.compile(
    r"^\s*(?:\?\s*for shortcuts|esc to interrupt|ctrl\+\w|\d+\s*tokens?|"
    r"press\s+\w+\s+to|shift\+tab)", re.IGNORECASE
)
_TOKEN_COUNTER = re.compile(r"^\s*[─-╿\s]*\(?\s*\d[\d,.]*\s*(?:tokens?|s)\b")

# Heuristic markers. These read rendered text -- that is the limitation of this bridge.
_PERMISSION_MARKERS = (
    "do you want to",
    "would you like to",
    "requesting permission",
    "needs your permission",
    "allow this",
    "approve this",
)
_OPTION_LINE = re.compile(r"^\s*[│|]?\s*(?:[❯>]\s*)?(\d)[.)]\s+(\S.*?)\s*$")
_LIMIT_MARKERS = (
    "usage limit reached",
    "you've reached your usage limit",
    "approaching your usage limit",
    "rate limit",
    "limit resets at",
    "limit will reset",
)
_RESET_AT = re.compile(r"reset[s]?\s+at\s+([0-9]{1,2}[:.][0-9]{2}\s*(?:am|pm)?[^\n.]*)", re.I)


class TmuxBridge:
    kind = "tmux"

    def __init__(
        self,
        session: str,
        claude_binary: str = "claude",
        poll_interval_ms: int = 250,
        idle_settle_ms: int = 1200,
        turn_timeout_s: float = 900.0,
    ):
        if not session:
            raise BridgeError(
                "CLAUDE_BRIDGE=tmux requires TMUX_SESSION. Run `tmux ls` to find the "
                "session running Claude Code, or set CLAUDE_BRIDGE=headless."
            )
        self._session = session
        self._binary = claude_binary
        self._poll_s = poll_interval_ms / 1000.0
        self._idle_settle_s = idle_settle_ms / 1000.0
        self._turn_timeout_s = turn_timeout_s
        self._committed = 0
        self._auth_detail = "not verified"
        self._turns = 0
        self._pending_options: dict[str, str] = {}
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        code, out, err = await self._run("tmux", "has-session", "-t", self._session)
        if code != 0:
            detail = err.strip() or "no such session"
            raise BridgeError(
                f"tmux session {self._session!r} does not exist ({detail}). "
                "Start Claude Code in a tmux session first, e.g. "
                f"`tmux new -s {self._session} claude`."
            )

        await self._check_session_env()
        info = await detect_auth(self._binary)
        self._auth_detail = info.describe()

        # Establish the baseline so the first turn does not replay the scrollback.
        self._committed = len(await self._capture())
        log.info(
            "attached to tmux session %r (%d lines of history)",
            self._session,
            self._committed,
        )

    async def _check_session_env(self) -> None:
        """The tmux server has its own environment; ours cannot reach it."""
        code, out, _ = await self._run(
            "tmux", "show-environment", "-t", self._session
        )
        if code != 0:
            log.warning(
                "could not read the tmux session environment; cannot confirm that "
                "%s is absent from the session running Claude Code",
                "ANTHROPIC_API_KEY",
            )
            return

        present = [
            name
            for name in SCRUBBED_VARS
            if re.search(rf"^{re.escape(name)}=", out, re.MULTILINE)
        ]
        if present:
            raise AuthError(
                f"tmux session {self._session!r} has {', '.join(present)} in its "
                "environment. The Claude Code process inside it inherited that, and "
                "those credentials outrank your subscription login -- turns would be "
                "billed per token. Unset them and restart the session:\n"
                f"  tmux kill-session -t {self._session}\n"
                f"  env -u {' -u '.join(present)} tmux new -s {self._session} claude"
            )

    async def close(self) -> None:
        return None

    async def health(self) -> BridgeHealth:
        code, _, _ = await self._run("tmux", "has-session", "-t", self._session)
        return BridgeHealth(
            kind=self.kind,
            auth_method=self._auth_detail,
            alive=code == 0,
            session_id=self._session,
            turns=self._turns,
            structured_permissions=False,
            structured_rate_limits=False,
            detail="reads rendered text; permission and limit detection are heuristic",
        )

    # -- turn --------------------------------------------------------------------

    async def send(self, text: str) -> AsyncIterator[BridgeEvent]:
        async with self._lock:
            self._pending_options.clear()
            # -l is essential: without it, tmux reads the transcript as key names.
            code, _, err = await self._run(
                "tmux", "send-keys", "-t", self._session, "-l", text
            )
            if code != 0:
                yield BridgeEvent(EventKind.ERROR, f"tmux send-keys failed: {err.strip()}")
                return
            await self._run("tmux", "send-keys", "-t", self._session, "Enter")
            self._turns += 1

            started = time.perf_counter()
            last_change = time.perf_counter()
            previous: list[str] = []
            saw_output = False

            while True:
                await asyncio.sleep(self._poll_s)
                lines = await self._capture()

                if lines != previous:
                    last_change = time.perf_counter()
                    previous = lines

                # Hold the final line back: it may still be mid-render.
                stable_upto = max(self._committed, len(lines) - 1)
                if stable_upto > self._committed:
                    for event in self._classify(lines[self._committed : stable_upto]):
                        saw_output = True
                        yield event
                    self._committed = stable_upto

                idle_for = time.perf_counter() - last_change
                if idle_for >= self._idle_settle_s and saw_output:
                    break
                if time.perf_counter() - started > self._turn_timeout_s:
                    yield BridgeEvent(
                        EventKind.ERROR,
                        f"no settled output from tmux after {self._turn_timeout_s:.0f}s",
                    )
                    return

            # The pane has settled; commit the final line too.
            lines = await self._capture()
            if len(lines) > self._committed:
                for event in self._classify(lines[self._committed :]):
                    yield event
                self._committed = len(lines)

            yield BridgeEvent(
                EventKind.DONE, "", {"session_id": self._session, "turns": self._turns}
            )

    def _classify(self, lines: list[str]) -> list[BridgeEvent]:
        events: list[BridgeEvent] = []
        prose: list[str] = []
        window = "\n".join(lines).lower()

        for line in lines:
            match = _OPTION_LINE.match(line)
            if match:
                self._pending_options[match.group(1)] = match.group(2)

        def flush_prose() -> None:
            if prose:
                text = "\n".join(prose).strip()
                if text:
                    events.append(BridgeEvent(EventKind.PROSE, text))
                prose.clear()

        for line in lines:
            if _is_furniture(line):
                continue
            stripped = line.strip()
            if not stripped:
                prose.append("")
                continue
            lowered = stripped.lower()

            if any(marker in lowered for marker in _LIMIT_MARKERS):
                flush_prose()
                reset = _RESET_AT.search(stripped)
                events.append(
                    BridgeEvent(
                        EventKind.RATE_LIMIT,
                        stripped,
                        {
                            "heuristic": True,
                            "resets_at": reset.group(1).strip() if reset else None,
                        },
                    )
                )
                continue
            prose.append(stripped)

        if any(marker in window for marker in _PERMISSION_MARKERS) and self._pending_options:
            # The accumulated prose *is* the prompt; it travels in meta["prompt"] so
            # emitting it as PROSE too would have the bot read the question twice.
            prose.clear()
            events.append(
                BridgeEvent(
                    EventKind.PERMISSION,
                    "Claude Code is asking for permission.",
                    {
                        "heuristic": True,
                        "options": dict(self._pending_options),
                        "prompt": "\n".join(
                            line.strip()
                            for line in lines
                            if not _is_furniture(line) and line.strip()
                        ),
                    },
                )
            )
            return events

        flush_prose()
        return events

    # -- control -----------------------------------------------------------------

    async def interrupt(self) -> None:
        """/interrupt sends ESC, which is how Claude Code cancels an in-flight turn."""
        await self._run("tmux", "send-keys", "-t", self._session, "Escape")
        log.info("sent ESC to tmux session %r", self._session)

    async def respond_to_permission(self, decision: PermissionDecision) -> None:
        if not self._pending_options:
            raise BridgeError(
                "no permission prompt is pending, or its options could not be read. "
                "Answer it directly in the tmux session."
            )

        wanted = ("yes", "allow", "approve") if decision is PermissionDecision.APPROVE else (
            "no", "deny", "reject", "cancel"
        )
        # Match against the options the prompt actually printed. If the intended
        # option is not there, refuse rather than guessing a keystroke.
        choice = next(
            (key for key, label in sorted(self._pending_options.items())
             if any(word in label.lower() for word in wanted)),
            None,
        )
        if choice is None:
            raise BridgeError(
                "could not match that decision to the options on screen "
                f"({self._pending_options}). Answer it in the tmux session instead."
            )

        log.warning(
            "sending permission response %r (option %s: %r) to tmux session %r",
            decision.value, choice, self._pending_options[choice], self._session,
        )
        await self._run("tmux", "send-keys", "-t", self._session, "-l", choice)
        await self._run("tmux", "send-keys", "-t", self._session, "Enter")
        self._pending_options.clear()

    @property
    def pending_options(self) -> dict[str, str]:
        return dict(self._pending_options)

    # -- tmux plumbing -----------------------------------------------------------

    async def _capture(self) -> list[str]:
        code, out, _ = await self._run(
            "tmux", "capture-pane", "-p", "-t", self._session, "-S", "-"
        )
        if code != 0:
            return []
        lines = [line.rstrip() for line in out.split("\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        return lines

    async def _run(self, *argv: str) -> tuple[int, str, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=build_subprocess_env(),
            )
            stdout, stderr = await process.communicate()
        except OSError as exc:
            return 127, "", str(exc)
        return (
            process.returncode or 0,
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
        )


def _is_furniture(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        _PROMPT_LINE.match(line)
        or _BOX_LINE.match(line)
        or _SPINNER_LINE.match(line)
        or _HINT_LINE.match(stripped)
        or _TOKEN_COUNTER.match(line)
    )
