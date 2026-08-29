"""The ClaudeBridge interface.

`send()` yields typed events rather than bare strings. An opaque string stream cannot
carry the three things this bot has to act on differently:

  * prose, which is spoken and mirrored;
  * tool-call output, which is mirrored but never spoken;
  * a permission prompt, which must halt the turn and require explicit confirmation.

Detecting that last one by pattern-matching rendered text is exactly the failure mode
worth avoiding -- it is how a bot ends up answering "yes" to a prompt nobody read. The
headless bridge gets this structurally from the JSON stream. The tmux bridge cannot,
and marks its guesses as heuristic so /status can report the difference.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class EventKind(StrEnum):
    PROSE = "prose"          # speakable; goes to TTS and the mirror
    RAW = "raw"              # mirror only, never spoken
    PERMISSION = "permission"  # blocked awaiting explicit confirmation
    RATE_LIMIT = "rate_limit"  # subscription usage limit reached
    ERROR = "error"
    DONE = "done"


class PermissionDecision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class BridgeEvent:
    kind: EventKind
    text: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def heuristic(self) -> bool:
        """True when the classification was inferred from rendered text, not structure."""
        return bool(self.meta.get("heuristic"))


@dataclass(frozen=True, slots=True)
class BridgeHealth:
    kind: str
    auth_method: str
    alive: bool
    session_id: str | None = None
    turns: int = 0
    detail: str = ""
    structured_permissions: bool = True
    structured_rate_limits: bool = True

    def describe(self) -> str:
        lines = [
            f"bridge: {self.kind} ({'alive' if self.alive else 'not connected'})",
            f"auth: {self.auth_method}",
            f"turns this session: ~{self.turns}",
        ]
        if self.session_id:
            lines.append(f"session: {self.session_id}")
        degraded = [
            name
            for name, ok in (
                ("permission prompts", self.structured_permissions),
                ("rate limits", self.structured_rate_limits),
            )
            if not ok
        ]
        if degraded:
            lines.append(f"degraded (heuristic only): {', '.join(degraded)}")
        if self.detail:
            lines.append(self.detail)
        return "\n".join(lines)


class BridgeError(RuntimeError):
    pass


class AuthError(BridgeError):
    """Raised at startup when the active auth is not a subscription credential."""


@runtime_checkable
class ClaudeBridge(Protocol):
    kind: str

    async def start(self) -> None:
        """Verify auth and attach to the session. Raises AuthError to abort startup."""

    def send(self, text: str) -> AsyncIterator[BridgeEvent]:
        """Inject one user turn. Yields until DONE, ERROR, or RATE_LIMIT."""

    async def interrupt(self) -> None:
        """Interrupt the in-flight turn without killing the session."""

    async def respond_to_permission(self, decision: PermissionDecision) -> None:
        """Answer a pending permission prompt.

        Only ever called from an explicit human confirmation -- a slash command, or a
        spoken phrase matched against the options the prompt actually offered. Never
        from a bare affirmative in the transcript.
        """

    async def health(self) -> BridgeHealth: ...

    async def close(self) -> None: ...
