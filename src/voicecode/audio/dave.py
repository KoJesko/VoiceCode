"""DAVE (Discord Audio & Video E2EE) decryption for the receive path.

Why this module exists
----------------------
Discord enforced the DAVE protocol for non-Stage voice calls on 2026-03-02. Clients
that advertise `max_dave_protocol_version: 0` are rejected with close code 4017.

discord.py 2.7 implements DAVE via the `davey` native module, so the bot connects and
*sends* fine. But it only ever encrypts -- there is no decrypt call anywhere in
discord.py's voice code, because the library has no receive path. And
discord-ext-voice-recv, whose last release predates the enforcement by nine months,
knows nothing about DAVE: its PacketDecryptor undoes the transport layer
(aead_xchacha20_poly1305_rtpsize and the legacy xsalsa modes) and stops there.

The result, without this module, is that everything looks connected and nothing is
heard: transport-decrypted frames are still MLS-encrypted, so Opus decoding produces
noise and speaking detection never fires.

So we take the opus payload from the sink (`wants_opus() -> True`) and run the missing
step ourselves. discord.py has already done the hard part -- it negotiates the MLS
group, processes welcomes and commits, and keeps the session current -- so all we need
is the read side of the session it maintains.

The private attribute
---------------------
`voice_client._connection.dave_session` is not public API. discord.py uses exactly this
path internally for its own `voice_privacy_code` property, so it is unlikely to move
without a major version, but it is still a private attribute and we treat it as one:
`DaveDecryptor.self_check()` runs at startup and fails loudly if the shape has changed.
Failing to start is much better than the alternative, which is a bot that joins,
appears healthy, and silently transcribes noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

try:  # pragma: no cover - availability depends on the install
    import davey  # type: ignore

    HAVE_DAVEY = True
except ImportError:  # pragma: no cover
    davey = None  # type: ignore
    HAVE_DAVEY = False


class DaveUnavailable(RuntimeError):
    """Raised at startup when the DAVE receive path cannot be wired up."""


@dataclass(frozen=True, slots=True)
class DaveStatus:
    available: bool
    session_active: bool
    protocol_version: int
    detail: str

    def describe(self) -> str:
        if not self.available:
            return f"unavailable ({self.detail})"
        if not self.session_active:
            return "no active session (passthrough)"
        return f"active (protocol v{self.protocol_version})"


def preflight() -> None:
    """Verify the DAVE receive path can work at all. Call once at startup.

    Checks the `davey` module exposes the calls we depend on, and that discord.py still
    keeps its session where we expect. Raises DaveUnavailable with a specific reason.
    """
    if not HAVE_DAVEY:
        raise DaveUnavailable(
            "the 'davey' module is not installed. discord.py needs it for voice at all "
            "since Discord's DAVE enforcement (2026-03-02); install discord.py[voice]."
        )

    for name in ("DaveSession", "MediaType"):
        if not hasattr(davey, name):
            raise DaveUnavailable(f"davey has no {name}; expected davey >= 0.1.6")
    for method in ("decrypt", "can_passthrough"):
        if not hasattr(davey.DaveSession, method):
            raise DaveUnavailable(
                f"davey.DaveSession has no {method}(); this build cannot decrypt "
                "received audio"
            )
    if not hasattr(davey.MediaType, "audio"):
        raise DaveUnavailable("davey.MediaType has no 'audio' member")

    try:
        from discord import VoiceClient
        from discord.voice_state import VoiceConnectionState
    except ImportError as exc:  # pragma: no cover
        raise DaveUnavailable(f"cannot import discord voice internals: {exc}") from exc

    # We read voice_client._connection.dave_session. Confirm both hops still exist
    # before we depend on them in the hot path.
    if "_connection" not in getattr(VoiceClient, "__annotations__", {}) and not hasattr(
        VoiceClient, "voice_privacy_code"
    ):
        raise DaveUnavailable(
            "discord.VoiceClient no longer looks like it carries a _connection state; "
            "voicecode/audio/dave.py needs updating for this discord.py version"
        )
    if not hasattr(VoiceConnectionState, "__init__"):  # pragma: no cover
        raise DaveUnavailable("discord.voice_state.VoiceConnectionState is not a class")

    log.info("DAVE receive path available (davey protocol v%s)", davey.DAVE_PROTOCOL_VERSION)


class DaveDecryptor:
    """Decrypts received Opus frames for one voice connection.

    Frames pass through untouched when there is no active session or when the sender is
    marked passthrough. That keeps a single code path working for Stage channels (which
    are exempt from the enforcement) and for the window during a protocol transition,
    when Discord explicitly puts the session into passthrough.
    """

    __slots__ = ("_voice_client", "_warned_missing", "_fail_counts")

    def __init__(self, voice_client: Any):
        self._voice_client = voice_client
        self._warned_missing = False
        self._fail_counts: dict[int, int] = {}

    def _session(self) -> Any | None:
        connection = getattr(self._voice_client, "_connection", None)
        if connection is None:
            if not self._warned_missing:
                log.error(
                    "voice client has no _connection; DAVE decryption disabled. "
                    "Received audio will be unusable if the channel has E2EE active."
                )
                self._warned_missing = True
            return None
        return getattr(connection, "dave_session", None)

    @property
    def status(self) -> DaveStatus:
        if not HAVE_DAVEY:
            return DaveStatus(False, False, 0, "davey not installed")
        session = self._session()
        if session is None:
            return DaveStatus(True, False, 0, "no session")
        try:
            active = bool(session.ready)
            version = int(session.protocol_version)
        except Exception as exc:  # pragma: no cover - defensive
            return DaveStatus(True, False, 0, f"session unreadable: {exc}")
        return DaveStatus(True, active, version, "ok")

    def decrypt(self, user_id: int, payload: bytes) -> bytes | None:
        """Decrypt one Opus frame. Returns None if the frame must be dropped.

        A dropped frame is normal early in a session -- a sender's key can arrive after
        their first packets do -- so failures are counted per user and logged on a
        ramp rather than per frame.
        """
        if not payload:
            return None

        session = self._session()
        if session is None:
            return payload  # no DAVE in play; transport decryption was enough

        try:
            if not session.ready:
                return payload
            if session.can_passthrough(user_id):
                return payload
            plaintext = session.decrypt(user_id, davey.MediaType.audio, payload)
        except Exception as exc:
            self._note_failure(user_id, str(exc))
            return None

        if not plaintext:
            self._note_failure(user_id, "empty plaintext")
            return None

        if self._fail_counts.pop(user_id, 0):
            log.info("DAVE decryption recovered for user %s", user_id)
        return bytes(plaintext)

    def _note_failure(self, user_id: int, reason: str) -> None:
        count = self._fail_counts.get(user_id, 0) + 1
        self._fail_counts[user_id] = count
        # 1st, 10th, 100th, ... -- enough to notice a persistent failure without
        # writing a line for every 20 ms frame.
        if count == 1 or (count % 100 == 0):
            log.warning(
                "DAVE decryption failed for user %s (%d frame(s)): %s",
                user_id,
                count,
                reason,
            )

    def forget(self, user_id: int) -> None:
        self._fail_counts.pop(user_id, None)
