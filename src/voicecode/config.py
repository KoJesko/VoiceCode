"""Configuration, loaded from .env via pydantic-settings.

Two things here are load-bearing beyond ordinary settings handling:

1. Allowlists fail closed. An unset or empty allowlist denies everything. This is the
   opposite of the usual "empty means unrestricted" convention and is deliberate --
   a typo in .env should silence the bot, never open it up.

2. Allowlists are hot-reloadable. Callers never read the live Settings object; they
   read an immutable AllowlistSnapshot from the ConfigStore. Reload builds a whole new
   snapshot and swaps it in one assignment, so a caller can never observe a
   half-applied policy.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator, Field, SecretStr
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

log = logging.getLogger(__name__)


class WakeMode(StrEnum):
    ALWAYS = "always"
    WAKEWORD = "wakeword"
    PTT = "ptt"


class BridgeKind(StrEnum):
    HEADLESS = "headless"
    TMUX = "tmux"


class AsrBackend(StrEnum):
    NEMO_UNIFIED = "nemo_unified"
    HF_TDT = "hf_tdt"


def _parse_id_set(value: Any) -> Any:
    """Accept "1,2,3" or "1 2 3" or a JSON list. Blank means the empty set (deny all)."""
    if value is None or isinstance(value, (set, frozenset, list, tuple)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("["):
            # NoDecode means nothing else will parse this, so do it here.
            return {int(item) for item in json.loads(text)}
        return {int(part) for part in text.replace(",", " ").split() if part}
    return value


def _parse_binding(value: Any) -> Any:
    """Accept "voice_id:text_id,voice_id:text_id" or a JSON object."""
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{"):
            return {int(k): int(v) for k, v in json.loads(text).items()}
        out: dict[int, int] = {}
        for pair in text.replace(",", " ").split():
            if ":" not in pair:
                raise ValueError(
                    f"TEXT_CHANNEL_BINDING entry {pair!r} is not voice_id:text_id"
                )
            voice, chan = pair.split(":", 1)
            out[int(voice)] = int(chan)
        return out
    return value


# NoDecode stops pydantic-settings from JSON-decoding these before our validators
# run. Without it, GUILD_ALLOWLIST=1,2,3 raises a SettingsError from the dotenv source
# rather than reaching _parse_id_set, because set[int] is a "complex" field.
IdSet = Annotated[set[int], NoDecode, BeforeValidator(_parse_id_set)]
Binding = Annotated[dict[int, int], NoDecode, BeforeValidator(_parse_binding)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- Discord ---------------------------------------------------------------
    discord_token: SecretStr = Field(default=SecretStr(""))

    # --- Scoping. All of these deny when empty. --------------------------------
    guild_allowlist: IdSet = Field(default_factory=set)
    voice_channel_allowlist: IdSet = Field(default_factory=set)
    text_channel_binding: Binding = Field(default_factory=dict)
    user_allowlist: IdSet = Field(default_factory=set)
    auto_join: bool = False

    # --- Turn logic ------------------------------------------------------------
    wake_mode: WakeMode = WakeMode.ALWAYS
    wake_word: str = "claude"
    endpoint_silence_ms: int = Field(default=700, ge=100, le=5000)
    min_utterance_ms: int = Field(default=300, ge=0, le=5000)
    max_utterance_ms: int = Field(default=30_000, ge=1000)
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # --- ASR -------------------------------------------------------------------
    asr_backend: AsrBackend = AsrBackend.NEMO_UNIFIED
    asr_model_id: str = "nvidia/parakeet-unified-en-0.6b"
    asr_device: str = "cuda"

    # --- TTS -------------------------------------------------------------------
    tts_voice: str = "af_heart"
    tts_lang_code: str = "a"
    tts_speed: float = Field(default=1.0, gt=0.0, le=3.0)
    tts_enabled: bool = True

    # --- Speech shaping --------------------------------------------------------
    speak_char_limit: int = Field(default=600, ge=50)

    # --- Bridge ----------------------------------------------------------------
    claude_bridge: BridgeKind = BridgeKind.HEADLESS
    tmux_session: str = ""
    claude_binary: str = "claude"
    claude_cwd: str = ""
    claude_permission_mode: str = "default"
    tmux_poll_interval_ms: int = Field(default=250, ge=50)
    tmux_idle_settle_ms: int = Field(default=1200, ge=200)

    # --- Ops -------------------------------------------------------------------
    log_level: str = "INFO"

    # There is deliberately no "a channel may not be bound to itself" check.
    # Discord voice channels carry a built-in text chat whose channel ID *is* the
    # voice channel's ID (Text-in-Voice), and discord.py models that by making
    # VoiceChannel a Messageable -- so `1234:1234` is a valid, useful binding
    # meaning "mirror into this voice channel's own chat", and mirror.py sends to
    # it without caring which kind of channel it is. Rejecting it caught a
    # plausible copy-paste slip, but at the cost of refusing a real configuration,
    # and a wrong-but-distinct ID slipped through the same check anyway.


@dataclass(frozen=True, slots=True)
class AllowlistSnapshot:
    """An immutable view of the scoping policy.

    Every scoping decision in the bot reads one of these. Snapshots are replaced
    wholesale on reload, never mutated, so a decision made against a snapshot is
    always self-consistent even if a reload lands mid-decision.
    """

    guilds: frozenset[int]
    voice_channels: frozenset[int]
    users: frozenset[int]
    bindings: dict[int, int]
    auto_join: bool
    revision: int

    def guild_allowed(self, guild_id: int | None) -> bool:
        return guild_id is not None and guild_id in self.guilds

    def user_allowed(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.users

    def voice_channel_allowed(self, channel_id: int | None) -> bool:
        return channel_id is not None and channel_id in self.voice_channels

    def bound_text_channel(self, voice_channel_id: int | None) -> int | None:
        if voice_channel_id is None:
            return None
        return self.bindings.get(voice_channel_id)

    def occupiable(self, voice_channel_id: int | None) -> bool:
        """A channel the bot may sit in: allowlisted AND with a bound text channel.

        Refusing to occupy an unbound channel is a spec requirement -- without a
        mirror target there is no source of truth for the conversation.
        """
        return self.voice_channel_allowed(voice_channel_id) and (
            self.bound_text_channel(voice_channel_id) is not None
        )


def _snapshot_from(settings: Settings, revision: int) -> AllowlistSnapshot:
    return AllowlistSnapshot(
        guilds=frozenset(settings.guild_allowlist),
        voice_channels=frozenset(settings.voice_channel_allowlist),
        users=frozenset(settings.user_allowlist),
        bindings=dict(settings.text_channel_binding),
        auto_join=settings.auto_join,
        revision=revision,
    )


class ConfigStore:
    """Holds live settings plus the current allowlist snapshot.

    `reload()` re-reads the .env file and rebuilds the snapshot. Non-scoping settings
    that are consumed at startup (the Discord token, the ASR model) are re-read too but
    take effect only on restart; that is noted rather than enforced, since reloading
    them would mean tearing down the model or the gateway connection.
    """

    def __init__(self, settings: Settings | None = None, env_file: str | Path = ".env"):
        self._lock = threading.Lock()
        self._env_file = Path(env_file)
        self._settings = settings or Settings(_env_file=self._env_file)  # type: ignore[call-arg]
        self._revision = 0
        self._snapshot = _snapshot_from(self._settings, self._revision)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def snapshot(self) -> AllowlistSnapshot:
        # Plain attribute read. Assignment is atomic under the GIL, so readers never
        # need the lock and never block on a reload.
        return self._snapshot

    def reload(self) -> AllowlistSnapshot:
        """Re-read config from disk and atomically swap in a new snapshot."""
        with self._lock:
            fresh = Settings(_env_file=self._env_file)  # type: ignore[call-arg]
            self._revision += 1
            new_snapshot = _snapshot_from(fresh, self._revision)
            self._settings = fresh
            self._snapshot = new_snapshot
        log.info(
            "config reloaded (revision %d): %d guild(s), %d voice channel(s), "
            "%d user(s), %d binding(s), auto_join=%s",
            new_snapshot.revision,
            len(new_snapshot.guilds),
            len(new_snapshot.voice_channels),
            len(new_snapshot.users),
            len(new_snapshot.bindings),
            new_snapshot.auto_join,
        )
        return new_snapshot

    def replace_snapshot(self, snapshot: AllowlistSnapshot) -> None:
        """Test hook. Not used in production paths."""
        with self._lock:
            self._snapshot = snapshot


def describe_scope(snapshot: AllowlistSnapshot) -> str:
    """Human-readable scope summary for /status and startup logging."""
    unbound = sorted(c for c in snapshot.voice_channels if c not in snapshot.bindings)
    lines = [
        f"guilds: {len(snapshot.guilds)}",
        f"voice channels: {len(snapshot.voice_channels)}",
        f"transcribed users: {len(snapshot.users)}",
        f"bindings: {len(snapshot.bindings)}",
        f"auto-join: {'on' if snapshot.auto_join else 'off'}",
    ]
    if unbound:
        lines.append(f"UNBOUND (will refuse to join): {unbound}")
    return " | ".join(lines)
