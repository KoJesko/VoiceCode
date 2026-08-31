"""The voice session's safety-critical behaviour.

These are the paths where a bug is expensive rather than annoying: answering a
permission prompt without a human, retry-storming a rate limit against the
shared subscription pool, or talking over someone. They had no coverage.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from voicecode.audio.turn import TurnEvent, TurnEventKind, Utterance
from voicecode.bridge.base import BridgeEvent, EventKind, PermissionDecision
from voicecode.config import AllowlistSnapshot, ConfigStore, Settings, WakeMode
from voicecode.discord_app.session import VoiceSession

GUILD, VOICE, TEXT, USER = 1, 2, 5, 3


def snapshot() -> AllowlistSnapshot:
    return AllowlistSnapshot(
        frozenset({GUILD}), frozenset({VOICE}), frozenset({USER}), {VOICE: TEXT}, False, 0
    )


class FakeVoiceClient:
    def __init__(self):
        self.guild = type("G", (), {"id": GUILD, "get_member": lambda self, i: None})()
        self.channel = type("C", (), {"id": VOICE, "name": "dev"})()
        self.playing = False
        self.stop_calls = 0

    def is_connected(self):
        return True

    def is_playing(self):
        return self.playing

    def play(self, source, after=None):
        self.playing = True

    def stop(self):
        self.playing = False
        self.stop_calls += 1

    async def disconnect(self, force=False):
        self.playing = False


class FakeASR:
    name = "fake"

    def __init__(self, text="fix the bug"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        from voicecode.asr.base import Transcript

        self.calls += 1
        return Transcript(self.text, 1.0, self.name)

    def describe(self):
        return "fake asr"


class FakeTTS:
    enabled = True

    def synthesize(self, text):
        from voicecode.tts.kokoro_engine import Speech

        return Speech(np.zeros(2400, dtype=np.float32), 100.0, 1.0)

    def describe(self):
        return "fake tts"


class FakeMirror:
    def __init__(self):
        self.sent = []

    async def send(self, g, c, content):
        self.sent.append(content)
        return True

    async def transcript(self, g, c, speaker, text):
        self.sent.append(f"transcript:{text}")
        return True

    async def claude_output(self, g, c, text):
        self.sent.append(f"output:{text}")
        return True

    async def notice(self, g, c, text):
        self.sent.append(f"notice:{text}")
        return True


class FakeBridge:
    kind = "fake"

    def __init__(self, events):
        self.events = events
        self.sends = []
        self.decisions = []
        self.interrupts = 0

    async def send(self, text):
        self.sends.append(text)
        for event in self.events:
            yield event

    async def interrupt(self):
        self.interrupts += 1

    async def respond_to_permission(self, decision):
        self.decisions.append(decision)

    async def health(self):
        from voicecode.bridge.base import BridgeHealth

        return BridgeHealth(self.kind, "fake", True)

    async def close(self):
        pass


def build(events, *, asr_text="fix the bug", wake_mode=WakeMode.ALWAYS):
    store = ConfigStore(settings=Settings(wake_mode=wake_mode))
    store.replace_snapshot(snapshot())
    bridge = FakeBridge(events)
    mirror = FakeMirror()
    session = VoiceSession(
        bot=None,
        voice_client=FakeVoiceClient(),
        config=store,
        asr=FakeASR(asr_text),
        tts=FakeTTS(),
        bridge=bridge,
        mirror=mirror,
    )
    return session, bridge, mirror


def utterance_event(user_id=USER):
    return TurnEvent(
        TurnEventKind.UTTERANCE,
        user_id,
        utterance=Utterance(
            user_id, np.zeros(16000, dtype=np.float32), 1000.0, time.perf_counter()
        ),
    )


# -- permission latch ---------------------------------------------------------

PERMISSION = BridgeEvent(
    EventKind.PERMISSION, "needs permission", {"tool": "Bash", "prompt": "run rm -rf?"}
)


async def test_permission_latches_the_session():
    session, _, _ = build([PERMISSION])
    await session.on_utterance(utterance_event())
    assert session.pending_permission is not None
    assert session.pending_permission.tool == "Bash"


async def test_speech_while_latched_is_refused_not_sent():
    """The whole point: nothing spoken can answer a permission prompt."""
    session, bridge, mirror = build([PERMISSION])
    await session.on_utterance(utterance_event())
    assert len(bridge.sends) == 1

    session.asr.text = "yes"
    await session.on_utterance(utterance_event())

    assert len(bridge.sends) == 1, "a spoken 'yes' reached the bridge"
    assert bridge.decisions == [], "a spoken 'yes' resolved the prompt"
    assert session.pending_permission is not None
    assert any("approve" in m.lower() for m in mirror.sent)


@pytest.mark.parametrize("spoken", ["yes", "yeah do it", "approve", "1", "go ahead"])
async def test_no_spoken_phrase_releases_the_latch(spoken):
    session, bridge, _ = build([PERMISSION])
    await session.on_utterance(utterance_event())
    session.asr.text = spoken
    await session.on_utterance(utterance_event())
    assert session.pending_permission is not None
    assert bridge.decisions == []


async def test_explicit_approve_resolves_and_forwards():
    session, bridge, _ = build([PERMISSION])
    await session.on_utterance(utterance_event())
    message = await session.resolve_permission(PermissionDecision.APPROVE)
    assert bridge.decisions == [PermissionDecision.APPROVE]
    assert session.pending_permission is None
    assert "Bash" in message


async def test_explicit_deny_resolves_without_granting():
    session, bridge, _ = build([PERMISSION])
    await session.on_utterance(utterance_event())
    await session.resolve_permission(PermissionDecision.DENY)
    assert bridge.decisions == [PermissionDecision.DENY]
    assert session.pending_permission is None


async def test_resolving_nothing_is_a_no_op():
    session, bridge, _ = build([])
    assert "No permission" in await session.resolve_permission(PermissionDecision.APPROVE)
    assert bridge.decisions == []


# -- rate limit breaker -------------------------------------------------------

RATE_LIMIT = BridgeEvent(
    EventKind.RATE_LIMIT, "usage limit reached", {"retry_delay_ms": 600_000}
)


async def test_rate_limit_opens_the_breaker():
    session, _, _ = build([RATE_LIMIT])
    await session.on_utterance(utterance_event())
    assert session.rate_limited


async def test_breaker_refuses_turns_without_contacting_claude():
    """A rate limit must not become a retry storm against the shared pool."""
    session, bridge, _ = build([RATE_LIMIT])
    await session.on_utterance(utterance_event())
    assert len(bridge.sends) == 1

    for _ in range(5):
        await session.on_utterance(utterance_event())

    assert len(bridge.sends) == 1, "retried while rate limited"


async def test_breaker_uses_the_structured_window():
    session, _, _ = build([RATE_LIMIT])
    await session.on_utterance(utterance_event())
    assert 590 < session._rate_limit.remaining_s() / 60 * 60 <= 600


async def test_breaker_falls_back_when_no_window_is_given():
    """The tmux bridge never supplies a delay; back off anyway."""
    session, _, _ = build([BridgeEvent(EventKind.RATE_LIMIT, "limit", {"heuristic": True})])
    await session.on_utterance(utterance_event())
    assert session.rate_limited
    assert session._rate_limit.remaining_s() > 60


# -- barge-in -----------------------------------------------------------------

async def test_barge_in_stops_playback():
    session, _, _ = build([])
    session.playback.start(session.voice_client)
    session.voice_client.playing = True
    await session.on_speech_start(USER)
    assert session.voice_client.stop_calls == 1


async def test_barge_in_when_idle_is_harmless():
    session, _, _ = build([])
    await session.on_speech_start(USER)
    assert session.voice_client.stop_calls == 0


# -- scoping and gating -------------------------------------------------------

async def test_turn_from_a_non_allowlisted_user_is_dropped():
    session, bridge, _ = build([])
    await session.on_utterance(utterance_event(user_id=999))
    assert bridge.sends == []


async def test_empty_transcript_does_not_reach_the_bridge():
    session, bridge, _ = build([], asr_text="   ")
    await session.on_utterance(utterance_event())
    assert bridge.sends == []


async def test_wakeword_mode_suppresses_unprefixed_speech():
    session, bridge, _ = build([], asr_text="fix the bug", wake_mode=WakeMode.WAKEWORD)
    await session.on_utterance(utterance_event())
    assert bridge.sends == []


async def test_wakeword_mode_strips_the_wake_word():
    session, bridge, _ = build([], asr_text="Claude, fix the bug", wake_mode=WakeMode.WAKEWORD)
    await session.on_utterance(utterance_event())
    assert bridge.sends == ["fix the bug"]


async def test_ptt_mode_gates_on_the_toggle():
    session, bridge, _ = build([], wake_mode=WakeMode.PTT)
    await session.on_utterance(utterance_event())
    assert bridge.sends == []
    session.set_ptt(True)
    await session.on_utterance(utterance_event())
    assert bridge.sends == ["fix the bug"]


# -- normal turn --------------------------------------------------------------

async def test_prose_turn_mirrors_transcript_and_output():
    session, bridge, mirror = build(
        [BridgeEvent(EventKind.PROSE, "Fixed it."), BridgeEvent(EventKind.DONE, "")]
    )
    await session.on_utterance(utterance_event())
    assert bridge.sends == ["fix the bug"]
    assert any(m.startswith("transcript:") for m in mirror.sent)
    assert any("Fixed it." in m for m in mirror.sent)


async def test_tool_noise_is_mirrored_but_never_spoken():
    session, _, mirror = build([
        BridgeEvent(EventKind.RAW, "[Bash] rm -rf build"),
        BridgeEvent(EventKind.PROSE, "Done."),
        BridgeEvent(EventKind.DONE, ""),
    ])
    spoken = []
    original = session.tts.synthesize
    session.tts.synthesize = lambda t: (spoken.append(t), original(t))[1]

    await session.on_utterance(utterance_event())

    assert any("rm -rf build" in m for m in mirror.sent), "tool output missing from mirror"
    assert not any("rm -rf" in s for s in spoken), "tool output was spoken aloud"


async def test_bridge_error_is_reported_not_swallowed():
    session, _, mirror = build([BridgeEvent(EventKind.ERROR, "claude exited 1")])
    await session.on_utterance(utterance_event())
    assert any("claude exited 1" in m for m in mirror.sent)
