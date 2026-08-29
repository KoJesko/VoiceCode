# VoiceCode — design review

Status: **awaiting review.** Nothing beyond this document is implemented yet.

Everything below was verified against upstream on 2026-08-29 by fetching docs and by
installing the packages and introspecting them, not from memory. Where a claim came
from running code, the probe is quoted.

---

## 1. What changed since the spec was written

### 1.1 BLOCKER — Discord enforces DAVE E2EE; `discord-ext-voice-recv` cannot decrypt it

Discord globally enforced the DAVE (Audio & Video End-to-End Encryption) protocol for
**non-Stage** voice calls on **2026-03-02**. Voice gateway rejects clients that advertise
`max_dave_protocol_version: 0` with close code **4017**.

- `discord.py` 2.7.1 *does* implement DAVE, via the `davey` native module (a hard
  dependency now: `client.py` logs `davey is not installed, voice will NOT be supported`).
  So the bot connects fine and **TTS playback works**.
- `discord-ext-voice-recv` 0.5.2a179 is the newest release and its last commit is
  **2025-06-18**. It has **zero** DAVE/MLS/`davey` references. Its `PacketDecryptor`
  only undoes the *transport* layer (`aead_xchacha20_poly1305_rtpsize`, `xsalsa20_*`).

The consequence is the exact failure reported across the ecosystem: the bot joins, plays
audio fine, and **receives nothing usable** — after transport decryption the payload is
still MLS-encrypted, so Opus decode yields garbage and speaking events look dead.

**This kills the receive path, which is the entire point of the project.** It is fixable,
and the fix is not a hack:

`davey.DaveSession` exposes exactly what is needed, confirmed by introspection:

```
decrypt(self, /, user_id, media_type, packet)
can_passthrough(self, /, user_id)
ready, epoch, status
```

`discord.py` already negotiates and maintains the MLS session for us
(`voice_state.py` drives `process_welcome` / `process_commit` / `set_external_sender`),
and parks it on `voice_client._connection.dave_session`. We only need the *read* side.

**So the sink declares `wants_opus() -> True`** and we own the tail of the pipeline:

```
RTP → voice_recv transport-decrypt → [our code] DAVE decrypt → Opus decode → PCM
```

Opus decoding is `discord.opus.Decoder`, one instance per SSRC (verified:
`SAMPLING_RATE 48000`, `CHANNELS 2`, `FRAME_SIZE 3840` — which matches the spec's
20 ms / 3840-byte figure exactly). Per-SSRC decoders are required anyway for correct
packet-loss concealment, so this is the architecture we wanted regardless.

Frames are passed through untouched when `can_passthrough(user_id)` is true or no DAVE
session is ready, so the same code path works on Stage channels and if Discord ever
downgrades a session.

**Risk I am accepting on your behalf unless you object:** we depend on
`voice_client._connection.dave_session`, a private attribute of discord.py. It is
isolated behind one adapter module (`audio/dave.py`) with a startup self-check that
fails loudly if the attribute moves, rather than silently degrading to garbage audio.

**Alternative if you'd rather not carry that:** Stage channels are exempt from the
enforcement, so a Stage-only bot works with stock voice_recv. I don't recommend it —
Stage channels have a speaker/audience model that fights everything else in this spec.

### 1.2 `--bare` would silently break your subscription-auth requirement

From the headless docs, verbatim: *"In bare mode, Claude Code never reads OAuth
credentials or the system keychain"* and *"Bare mode does not read
`CLAUDE_CODE_OAUTH_TOKEN`. If your script passes `--bare`, authenticate with
`ANTHROPIC_API_KEY` or an `apiKeyHelper` instead."*

`--bare` is the flag the docs otherwise recommend for scripted use, and it is being made
the `-p` default in a future release. It is **incompatible with both of your supported
auth paths**. The headless bridge must never pass it, and must pin/verify behaviour when
the `-p` default flips. Guarded by a test asserting `--bare` is absent from the argv we build.

### 1.3 Auth precedence is worse than the spec assumes

Documented order — the first four all outrank your subscription login:

1. Cloud provider (`CLAUDE_CODE_USE_BEDROCK` / `_VERTEX` / `_FOUNDRY`)
2. `ANTHROPIC_AUTH_TOKEN`
3. `ANTHROPIC_API_KEY`  ← *"In non-interactive mode (`-p`), the key is always used when present."*
4. `apiKeyHelper`
5. `CLAUDE_CODE_OAUTH_TOKEN`
6. Anthropic profile / federation credentials
7. Subscription OAuth from `/login`

Stripping the two variables you named is necessary but **not sufficient**. The subprocess
env scrubber removes: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`,
`CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX`, `CLAUDE_CODE_USE_FOUNDRY`,
`ANTHROPIC_PROFILE`, `ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID`.
`apiKeyHelper` is a *settings* key, not an env var, so it is checked at startup instead.

Startup verification uses the real command rather than inference:
`claude auth status` (JSON; `--text` for humans). Refuse to start unless the active
method is a claude.ai/Console subscription login or `CLAUDE_CODE_OAUTH_TOKEN`.

### 1.4 Rate limits are a structured event, not a string to grep

`--output-format stream-json` emits `system` / `subtype: "api_retry"` carrying
`error` (one of `authentication_failed`, `oauth_org_not_allowed`, `billing_error`,
**`rate_limit`**, `overloaded`, …), plus `attempt`, `max_retries`, `retry_delay_ms`.

So the headless bridge detects usage limits structurally. **The tmux bridge cannot** —
it only ever sees rendered text, so there it stays a (fragile) pattern match on the
pane. Worth knowing before you pick a default bridge.

Also: `stream-json` requires `--verbose`. Token-level deltas additionally require
`--include-partial-messages`. Without the latter you get whole assistant messages,
which is *better* for us — sentence-streaming to Kokoro wants clauses, not tokens.

### 1.5 `nvidia/parakeet-unified-en-0.6b` is real, but its streaming API is not usable here

Verified on the Hub: released **2026-04-07**, NVIDIA Open Model License (note: *not*
CC-BY-4.0 like the v2/v3 models), 600M params, offline WER 5.91.

But the streaming story does not fit a live socket:

- The model card documents streaming only as (a) a NeMo example *script* over a dataset
  manifest, or (b) `PipelineBuilder.build_pipeline(cfg)` + `pipeline.run(audios)` where
  `audios` is **a list of file paths**. There is no documented feed-a-chunk API.
- It is **buffered** streaming, not cache-aware: *"left context is recomputed for each
  chunk"*, with a 5.6 s default left context. Every 560 ms chunk re-encodes ~6 s of audio.
- Model card says *"Runtime Engine: NeMo 2.7.3"*, and 2.7.3 was the **final pre-split**
  release — `nemo-toolkit` is now 3.0.0 and the repo *"has pivoted to focus on audio,
  speech, and multimodal LLMs."* The 3.0 API surface for this model is unverified.

**Recommendation: don't stream the ASR at all.** Your turn logic already endpoints on
trailing silence, so at endpoint we hold a complete utterance and run *offline*
`transcribe()`. That is more accurate (5.91 vs 6.29 WER at 1.12 s), far simpler, and
avoids the recompute cost entirely. For a 5 s utterance on a modern GPU this is well
inside the 1.5 s budget — and the budget is measured from *end of speech*, which is
precisely when offline transcription starts.

Streaming would only help if you wanted live captions *while* speaking. Say the word and
I'll add it as a display-only side channel, but it should not be on the critical path.

The ASR backend is therefore an interface with two implementations, `PARAKEET_MODEL_ID`
selecting between them, defaulting to unified:
- `nvidia/parakeet-unified-en-0.6b` via `nemo.collections.asr` (your stated preference)
- `nvidia/parakeet-tdt-0.6b-v3` — now carries `library: transformers` and loads via
  `AutoModel`, 2.4M downloads vs 949. Much lower install friction if NeMo 3.0 fights you.

### 1.6 Python 3.12 is the *only* version that works — good call, but it's tight

- `kokoro` 0.9.4 requires `>=3.10,<3.13`
- `nemo-toolkit` 3.0.0 requires **"Python 3.12 or above"** (tested on 3.13)

The intersection is exactly **3.12**. Pinned in `.python-version` and asserted at startup.

### 1.7 Speaking events don't reach the bot, and don't run on the event loop

The spec assumes `@bot.event on_voice_member_speaking_start`. Reading
`voice_recv/reader.py` and `voice_client.py`:

- `voice_member_speaking_start` / `_stop` are dispatched via
  `voice_client.dispatch_sink(...)` **only** — they never reach `client.dispatch`.
  A `@bot.event` handler for them **will never fire.** They must be `@AudioSink.listener()`
  methods on the sink.
- `voice_member_speaking_state(member, ssrc, state)` *does* go through `vc.dispatch`,
  so that one **is** available at bot level. It's the raw gateway speaking flag.
- Both these and `AudioSink.write()` run on **library threads**, not the asyncio loop.
  Every hop into bot logic goes through `asyncio.run_coroutine_threadsafe`. Getting this
  wrong is the classic source of silent deadlocks in this ext.
- `SpeakingTimer.speaking_timeout_delay` is hardcoded `0.2` s — confirms these events are
  only usable as the coarse gate you described. silero-vad remains the real endpointer.

### 1.8 Smaller confirmations

| Thing | Verified |
|---|---|
| `discord-ext-voice-recv` | `0.5.2a179`, pinned exactly. Requires `discord.py[voice]>=2.5`; resolves 2.7.1 |
| `AudioSink` abstract set | `frozenset({'write', 'cleanup', 'wants_opus'})`; `write(self, user: Optional[User], data: VoiceData)` |
| `VoiceData` fields | `opus`, `pcm`, `packet`, `source` |
| Transport mode | `aead_xchacha20_poly1305_rtpsize` — supported by both discord.py 2.7.1 and voice_recv. No mismatch |
| `kokoro` | 0.9.4, `KPipeline(lang_code='a')`, yields `(gs, ps, audio)`, audio 24 kHz float32 |
| `soxr` | 1.1.0. `resample(x, in_rate, out_rate, quality='HQ')`; `ResampleStream` for stateful 24k→48k |
| `claude setup-token` | Confirmed. One-year OAuth, Pro/Max/Team/Enterprise, inference-scoped, prints without saving |
| `claude auth status` | Confirmed subcommand, JSON by default, `--text` for humans |
| `--resume <id>` | Confirmed; since v2.1.223 resolves a session ID from any directory |
| silero-vad | 16 kHz requires a **512-sample** window. Not negotiable — the model is fixed-window |

---

## 2. File layout

```
VoiceCode/
├── pyproject.toml              # uv-managed, deps pinned
├── uv.lock
├── .python-version             # 3.12
├── .env.example
├── README.md
├── docs/DESIGN.md              # this file
└── src/voicecode/
    ├── __main__.py             # entrypoint; --selftest lives here
    ├── config.py               # pydantic-settings; hot-reloadable Allowlists
    ├── logging_setup.py        # structured logs + per-stage latency timers
    │
    ├── audio/
    │   ├── dave.py             # §1.1 DAVE decrypt adapter + startup self-check
    │   ├── opus_decode.py      # per-SSRC discord.opus.Decoder pool
    │   ├── resample.py         # soxr; 48k↔16k↔24k, stereo↔mono
    │   ├── sink.py             # VoiceCodeSink(AudioSink), wants_opus=True
    │   ├── vad.py              # silero-vad, 512-sample windows
    │   ├── turn.py             # per-user buffer, endpointing, barge-in
    │   └── playback.py         # discord.AudioSource over a 20 ms frame queue
    │
    ├── asr/
    │   ├── base.py             # ASREngine protocol
    │   ├── nemo_unified.py     # parakeet-unified-en-0.6b
    │   └── hf_tdt.py           # parakeet-tdt-0.6b-v3 via transformers
    │
    ├── tts/
    │   ├── kokoro_engine.py    # KPipeline, shared, OOM-degradable
    │   └── sentences.py        # sentence splitter feeding the synth queue
    │
    ├── bridge/
    │   ├── base.py             # ClaudeBridge + BridgeEvent (§3)
    │   ├── auth.py             # env scrub + claude auth status verification
    │   ├── tmux.py             # send-keys / capture-pane diffing
    │   └── headless.py         # claude -p --output-format stream-json
    │
    ├── speech/
    │   └── sanitize.py         # code fences, ANSI, paths, caps → speakable text
    │
    └── discord_app/
        ├── bot.py              # client, intents, guild-scoped command sync
        ├── scoping.py          # §4 — every gate lives here
        ├── commands.py         # slash commands
        └── mirror.py           # bound-text-channel transcript mirroring
```

`sanitize.py`, `scoping.py`, `resample.py`, and `config.py` are pure and unit-tested
without a GPU or a Discord connection. That's most of the logic that can actually be
wrong in a way tests catch.

---

## 3. `ClaudeBridge`

**I want to change the signature you specified.** `send(text) -> AsyncIterator[str]`
cannot carry the information your own requirements depend on: an opaque string stream
can't distinguish prose (speak it) from tool noise (mirror only), can't signal "Claude is
blocked on a permission prompt" without the bot pattern-matching rendered text — which is
exactly the failure mode you said to avoid ("Do not send bare `yes` by pattern-matching")
— and can't surface a structured `rate_limit`.

So the stream is typed. Everything else is as you specified.

```python
class EventKind(StrEnum):
    PROSE      = "prose"       # speakable; goes to TTS and the mirror
    RAW        = "raw"         # mirror only, never spoken (tool calls, diffs, logs)
    PERMISSION = "permission"  # Claude Code is blocked; requires explicit confirmation
    RATE_LIMIT = "rate_limit"  # subscription usage limit
    ERROR      = "error"
    DONE       = "done"

@dataclass(frozen=True)
class BridgeEvent:
    kind: EventKind
    text: str = ""
    # PERMISSION: {"prompt", "options", "tool"}
    # RATE_LIMIT: {"resets_at", "retry_delay_ms", "attempt", "max_retries"}
    # DONE:       {"session_id", "turns", "duration_ms"}
    meta: Mapping[str, Any] = field(default_factory=dict)

class ClaudeBridge(Protocol):
    async def start(self) -> None:
        """Verify auth, attach to the session. Raises AuthError to abort startup."""

    def send(self, text: str) -> AsyncIterator[BridgeEvent]:
        """Inject one user turn. Yields until DONE, ERROR, or RATE_LIMIT."""

    async def interrupt(self) -> None:
        """ESC to tmux / SIGINT to headless. Never cancels the process."""

    async def respond_to_permission(self, decision: PermissionDecision) -> None:
        """Answer a pending PERMISSION event. Only ever called from an explicit
        human confirmation — never inferred from transcript text."""

    async def health(self) -> BridgeHealth:
        """auth_method, session_id, alive, approx_turns — feeds /status."""

    async def close(self) -> None: ...
```

Notes:

- **Permission handling.** A `PERMISSION` event latches the bridge. The bot speaks the
  prompt, mirrors it in full, and refuses to send anything until a confirmation arrives
  via `/approve` (or a wake-worded spoken phrase matched against the *offered options*,
  never a bare affirmative). Any transcript arriving while latched is mirrored and
  dropped with a spoken "waiting on your approval". This is the one place I'd rather be
  annoying than clever.
- **Rate limits.** `RATE_LIMIT` speaks a one-liner, mirrors detail, and opens a circuit
  breaker until `resets_at`. No retries while open. Approximate turn count is logged per
  turn so the subscription burn is visible.
- **tmux bridge degradation.** It cannot produce structured `PERMISSION` or `RATE_LIMIT`
  events — it only sees rendered text. It emits them heuristically and marks
  `meta["heuristic"] = True`; `/status` reports the bridge as degraded on those two.
  If you care about permission safety more than about driving your existing interactive
  session, `headless` is the stronger default. Flagged for your call, defaulting to
  `tmux` as you specified.

### Open question I need you to answer

`CLAUDE_BRIDGE` defaults to `tmux`, which needs `TMUX_SESSION`. **You haven't told me the
session name** — you asked me to ask. If it's unset at startup the bot refuses to start
and prints the choice rather than guessing. Tell me the session name, or say `headless`.

---

## 4. Channel-scoping enforcement points

All gates live in `discord_app/scoping.py`. Every one **fails closed**; an empty
allowlist denies. Configuration is reloaded through an `AllowlistSnapshot` swapped
atomically, so hot-reload never leaves a half-applied policy.

| # | Point | Gate | On failure |
|---|---|---|---|
| 1 | `on_message` / every gateway event carrying a guild | `guild.id in GUILD_ALLOWLIST` | Drop silently |
| 2 | Command tree sync | Commands registered **guild-scoped** to `GUILD_ALLOWLIST` only; no global sync | Not registered |
| 3 | Every slash command, pre-dispatch | Interaction guild + invoking user | Ephemeral refusal naming the reason |
| 4 | `/join` | Target channel in `VOICE_CHANNEL_ALLOWLIST` **and** has a `TEXT_CHANNEL_BINDING` entry | Refuse, state which of the two failed |
| 5 | `on_voice_state_update`, **bot's own member** | Current channel still allowlisted **and** still bound | Disconnect immediately, log at WARNING with actor, guild, channel |
| 6 | `on_voice_state_update`, other members | `AUTO_JOIN` join/leave decisions; user must be in `USER_ALLOWLIST` | No action |
| 7 | **`AudioSink.write`, first statement** | `user.id in USER_ALLOWLIST` | Return before any buffering — no decrypt, no decode, no ASR |
| 8 | Speaking-state listeners on the sink | Same user gate | Ignore |
| 9 | Turn dispatch to the bridge | Re-check guild + channel + user against the *current* snapshot | Drop the utterance, log |
| 10 | Mirror write | Resolve target from `TEXT_CHANNEL_BINDING` at write time, never cached | Drop, log at ERROR |
| 11 | Playback start | Voice channel still allowlisted | Stop playback, disconnect |
| 12 | Post-hot-reload sweep | Re-validate every active connection against the new snapshot | Disconnect the now-disallowed |

Point 7 is the one that matters most for your privacy requirement: it sits **above** the
DAVE decrypt, so a non-allowlisted user's audio is discarded while still encrypted. It
never reaches a decoder, a buffer, or the GPU.

Point 5 covers the drag-the-bot-in case. Point 12 is what makes hot-reload meaningful —
without it, tightening an allowlist wouldn't evict an existing connection.

---

## 5. Decisions taken

All four open questions were answered before implementation:

1. **DAVE workaround** — proceed with the `davey` adapter and the private-attribute
   read, guarded by a startup preflight. Implemented in `audio/dave.py`.
2. **Bridge** — build both, default to `headless`. `tmux` still requires
   `TMUX_SESSION`; selecting it without one is a startup error naming the fix, since
   the session name was never supplied.
3. **Event stream** — typed `BridgeEvent`, not `AsyncIterator[str]`.
4. **ASR** — offline at the VAD endpoint, defaulting to `parakeet-unified-en-0.6b`
   with `parakeet-tdt-0.6b-v3` selectable via `ASR_BACKEND=hf_tdt`.

## 6. What could not be verified here

This environment has no GPU, no `ffmpeg`, and no `espeak-ng`, so the model-loading
paths were written against the APIs verified in section 1 but not executed. What *was*
executed here:

- the DAVE preflight, against the real `davey` and `discord.py` builds;
- auth detection, against the real `claude` binary;
- the whole pure-logic layer, under 113 tests;
- `--selftest`, end to end, degrading correctly with the models absent.

Still needs a run on the target machine: NeMo 3.0 loading `parakeet-unified-en-0.6b`
(the model card names NeMo 2.7.3, which was the final pre-split release), Kokoro
synthesis, silero-vad, and a real Discord voice connection to confirm the DAVE
decryption path against live traffic.
