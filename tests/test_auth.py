"""The subscription-billing constraint. These tests guard money, not behaviour."""

from __future__ import annotations

import pytest

from voicecode.bridge import auth
from voicecode.bridge.base import AuthError


def test_every_higher_precedence_credential_is_stripped(monkeypatch):
    """Claude Code ranks all of these above a subscription login."""
    for name in auth.SCRUBBED_VARS:
        monkeypatch.setenv(name, "value")
    env = auth.build_subprocess_env()
    for name in auth.SCRUBBED_VARS:
        assert name not in env, f"{name} would flip billing to metered API usage"


def test_the_two_named_vars_are_not_the_whole_list():
    """The brief named two; the docs rank four sources above /login."""
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "ANTHROPIC_PROFILE",
    ):
        assert name in auth.SCRUBBED_VARS


def test_oauth_token_is_preserved(monkeypatch):
    """CLAUDE_CODE_OAUTH_TOKEN is a subscription credential, not an API key."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    assert auth.build_subprocess_env()["CLAUDE_CODE_OAUTH_TOKEN"] == "tok"


def test_bare_flag_is_refused():
    """--bare reads neither OAuth credentials nor CLAUDE_CODE_OAUTH_TOKEN."""
    with pytest.raises(AuthError, match="--bare"):
        auth.check_forbidden_flags(["claude", "-p", "hi", "--bare"])
    auth.check_forbidden_flags(["claude", "-p", "hi"])  # no raise


@pytest.mark.parametrize(
    "blob,expected",
    [
        ("logged in with claude.ai (max)", auth.AuthMethod.SUBSCRIPTION),
        ("anthropic api key", auth.AuthMethod.API_KEY),
        ("using bedrock", auth.AuthMethod.CLOUD_PROVIDER),
        ("vertex", auth.AuthMethod.CLOUD_PROVIDER),
        ("claude_code_oauth_token", auth.AuthMethod.OAUTH_TOKEN),
        ("apiKeyHelper".lower(), auth.AuthMethod.API_KEY_HELPER),
        ("something else entirely", auth.AuthMethod.UNKNOWN),
    ],
)
def test_auth_method_classification(blob, expected):
    assert auth._method_from_text(blob) is expected


@pytest.mark.parametrize(
    "method",
    [auth.AuthMethod.API_KEY, auth.AuthMethod.CLOUD_PROVIDER, auth.AuthMethod.API_KEY_HELPER],
)
def test_metered_auth_refuses_startup(method):
    info = auth.AuthInfo(method, "detail", ())
    with pytest.raises(AuthError):
        auth._require_acceptable(info)


@pytest.mark.parametrize(
    "method", [auth.AuthMethod.SUBSCRIPTION, auth.AuthMethod.OAUTH_TOKEN]
)
def test_subscription_auth_is_accepted(method):
    assert auth._require_acceptable(auth.AuthInfo(method, "ok", ())).method is method


def test_unknown_auth_warns_but_proceeds():
    """The env is scrubbed, so an API key cannot reach the child; don't hard-fail."""
    info = auth.AuthInfo(auth.AuthMethod.UNKNOWN, "unparseable", ())
    assert auth._require_acceptable(info) is info
