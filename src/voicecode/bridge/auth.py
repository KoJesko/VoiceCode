"""Auth verification and environment scrubbing.

The hard constraint: this bot drives Claude Code on a Claude subscription and must
never move the user onto per-token API billing. That is not the default -- it is the
opposite of the default. Claude Code's documented precedence puts four credential
sources *above* the subscription login:

  1. Cloud provider  (CLAUDE_CODE_USE_BEDROCK / _VERTEX / _FOUNDRY)
  2. ANTHROPIC_AUTH_TOKEN
  3. ANTHROPIC_API_KEY      <- "In non-interactive mode (-p), the key is always used
                               when present." No approval prompt to catch it.
  4. apiKeyHelper           (a settings key, not an env var)
  5. CLAUDE_CODE_OAUTH_TOKEN
  6. Anthropic profile / federation credentials
  7. Subscription OAuth from /login                     <- what we want

So scrubbing the two variables named in the brief is necessary but not sufficient; all
of 1, 2, 3 and 6 are stripped from the child environment, and 4 is checked at startup
because it cannot be stripped.

CLAUDE_CODE_OAUTH_TOKEN is deliberately *not* stripped: it is a supported subscription
credential (`claude setup-token`, one year, Pro/Max/Team/Enterprise, inference-scoped)
and is the path for running this bot as a service.

One further trap: `--bare` is the flag the docs otherwise recommend for scripted use,
and is slated to become the default for `-p`. It never reads OAuth credentials or the
system keychain, and does not read CLAUDE_CODE_OAUTH_TOKEN either -- so it is
incompatible with *both* supported auth paths. The headless bridge must never pass it.
There is a test asserting it does not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .base import AuthError

log = logging.getLogger(__name__)

# Stripped from every subprocess environment. Each of these outranks the subscription
# login and would silently move billing to metered API usage.
SCRUBBED_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)

# Never pass these to the CLI. See the module docstring.
FORBIDDEN_FLAGS = ("--bare",)

_SETTINGS_PATHS = (
    Path.home() / ".claude" / "settings.json",
    Path("/etc/claude-code/managed-settings.json"),
)


class AuthMethod(StrEnum):
    SUBSCRIPTION = "subscription"       # /login credentials -- the default, preferred
    OAUTH_TOKEN = "oauth_token"         # CLAUDE_CODE_OAUTH_TOKEN from setup-token
    API_KEY = "api_key"                 # refuse
    CLOUD_PROVIDER = "cloud_provider"   # refuse
    API_KEY_HELPER = "api_key_helper"   # refuse
    UNKNOWN = "unknown"

    @property
    def acceptable(self) -> bool:
        return self in (AuthMethod.SUBSCRIPTION, AuthMethod.OAUTH_TOKEN)


@dataclass(frozen=True, slots=True)
class AuthInfo:
    method: AuthMethod
    detail: str
    scrubbed: tuple[str, ...]

    def describe(self) -> str:
        text = f"{self.method.value}: {self.detail}"
        if self.scrubbed:
            text += f" (stripped from subprocess env: {', '.join(self.scrubbed)})"
        return text


def build_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of os.environ with every billing-flipping credential removed."""
    env = dict(os.environ)
    for name in SCRUBBED_VARS:
        env.pop(name, None)
    if extra:
        env.update(extra)
    return env


def scrubbed_names() -> tuple[str, ...]:
    """Which scrubbed variables were actually present. Used for logging."""
    return tuple(name for name in SCRUBBED_VARS if name in os.environ)


def check_forbidden_flags(argv: list[str]) -> None:
    """Guard against a flag that would silently break subscription auth."""
    for flag in FORBIDDEN_FLAGS:
        if flag in argv:
            raise AuthError(
                f"refusing to run `claude` with {flag}: it never reads OAuth "
                "credentials or CLAUDE_CODE_OAUTH_TOKEN, so it cannot authenticate "
                "with a Claude subscription."
            )


def _api_key_helper_configured() -> str | None:
    """apiKeyHelper is a settings key, so it cannot be stripped -- only detected."""
    for path in _SETTINGS_PATHS:
        try:
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("apiKeyHelper"):
            return str(path)
    return None


async def detect_auth(claude_binary: str = "claude", timeout: float = 20.0) -> AuthInfo:
    """Determine which credential `claude` will actually use, under our scrubbed env.

    Runs `claude auth status` with the same environment the bridge will use, so the
    answer reflects the child process rather than this one.
    """
    scrubbed = scrubbed_names()

    if shutil.which(claude_binary) is None:
        raise AuthError(
            f"`{claude_binary}` is not on PATH. Install Claude Code and run `claude` "
            "once to log in, or set CLAUDE_BINARY to its full path."
        )

    helper_path = _api_key_helper_configured()
    if helper_path:
        raise AuthError(
            f"apiKeyHelper is configured in {helper_path}. It outranks your "
            "subscription login and cannot be stripped from the environment, so "
            "requests would be billed per token. Remove it to run this bot."
        )

    env = build_subprocess_env()
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        fallback = AuthInfo(
            AuthMethod.OAUTH_TOKEN,
            "CLAUDE_CODE_OAUTH_TOKEN is set (from `claude setup-token`)",
            scrubbed,
        )
    else:
        fallback = AuthInfo(
            AuthMethod.UNKNOWN, "could not read `claude auth status`", scrubbed
        )

    try:
        process = await asyncio.create_subprocess_exec(
            claude_binary,
            "auth",
            "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        log.warning("`%s auth status` timed out after %.0fs", claude_binary, timeout)
        return _require_acceptable(fallback)
    except OSError as exc:
        raise AuthError(f"could not run `{claude_binary} auth status`: {exc}") from exc

    text = (stdout or b"").decode("utf-8", "replace").strip()
    if not text:
        text = (stderr or b"").decode("utf-8", "replace").strip()

    info = _classify(text, scrubbed) or fallback
    return _require_acceptable(info)


def _classify(text: str, scrubbed: tuple[str, ...]) -> AuthInfo | None:
    """Read `claude auth status` output, JSON or human-readable."""
    if not text:
        return None

    payload: dict | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            payload = parsed
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        blob = " ".join(
            str(payload.get(key, ""))
            for key in ("method", "authMethod", "auth_method", "source", "type", "provider")
        ).lower()
        detail = (
            payload.get("account")
            or payload.get("email")
            or payload.get("organization")
            or payload.get("status")
            or "reported by `claude auth status`"
        )
        method = _method_from_text(blob or json.dumps(payload).lower())
        return AuthInfo(method, str(detail), scrubbed)

    lowered = text.lower()
    first_line = text.splitlines()[0].strip() if text.splitlines() else text
    return AuthInfo(_method_from_text(lowered), first_line[:200], scrubbed)


def _method_from_text(blob: str) -> AuthMethod:
    if any(word in blob for word in ("bedrock", "vertex", "foundry", "gateway")):
        return AuthMethod.CLOUD_PROVIDER
    if "apikeyhelper" in blob or "api key helper" in blob:
        return AuthMethod.API_KEY_HELPER
    if "api" in blob and "key" in blob:
        return AuthMethod.API_KEY
    if "oauth_token" in blob or "setup-token" in blob or "claude_code_oauth_token" in blob:
        return AuthMethod.OAUTH_TOKEN
    if any(word in blob for word in ("claude.ai", "claudeai", "subscription", "pro", "max",
                                     "team", "enterprise", "login", "logged in")):
        return AuthMethod.SUBSCRIPTION
    if "console" in blob:
        return AuthMethod.API_KEY
    return AuthMethod.UNKNOWN


def _require_acceptable(info: AuthInfo) -> AuthInfo:
    if info.method.acceptable:
        log.info("Claude Code auth: %s", info.describe())
        return info

    if info.method is AuthMethod.UNKNOWN:
        # Not proven bad, but not proven good either. Since the env is scrubbed and no
        # apiKeyHelper is configured, an API key cannot reach the child -- so this is a
        # warning rather than a refusal. Refusing here would break anyone whose
        # `claude auth status` output we simply do not recognise.
        log.warning(
            "could not positively identify the Claude Code auth method (%s). "
            "The subprocess environment is scrubbed of API credentials, so billing "
            "should still fall through to your subscription login. Run "
            "`claude auth status --text` to confirm.",
            info.detail,
        )
        return info

    raise AuthError(
        f"refusing to start: Claude Code would authenticate with {info.method.value} "
        f"({info.detail}). This bot must run on a Claude subscription, and "
        f"{info.method.value} bills per token instead. "
        "Run `unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN` and `claude /login`, or "
        "set CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`."
    )
