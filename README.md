# VoiceCode

Talk to a running Claude Code session from a Discord voice channel. You speak,
[NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) transcribes,
the text goes into Claude Code, and [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)
reads the answer back. All inference is local; no cloud STT or TTS.

Everything Claude says also lands in a bound text channel, in full. **That channel is
the source of truth** — speech is a summary, and it is allowed to say "twelve lines of
Python" where the text channel shows the code.

---

## Read this first: two things that will bite you

### Discord requires DAVE, and the receive extension does not implement it

Discord enforced the **DAVE** end-to-end encryption protocol for non-Stage voice calls
on **2026-03-02**. Clients that do not support it are rejected with close code `4017`.

`discord.py` 2.7 implements DAVE through the `davey` module, so the bot connects and
speaks. But `discord-ext-voice-recv` — pinned here at `0.5.2a179`, whose last commit
is **2025-06-18** — has no DAVE support at all. Its decryptor undoes only the transport
layer. Left alone, the bot joins, plays audio fine, and **hears nothing**: every
received frame is still MLS-encrypted, so Opus decoding produces noise.

VoiceCode supplies the missing step itself, in `src/voicecode/audio/dave.py`. The sink
takes raw Opus (`wants_opus() -> True`), decrypts it with the `DaveSession` that
`discord.py` already negotiates, and decodes it per-SSRC.

That reaches one private attribute, `voice_client._connection.dave_session`.
`discord.py` uses the same path internally for its public `voice_privacy_code`
property, so it is unlikely to move without a major version — but a startup preflight
checks it and **refuses to start** if the shape has changed. Failing loudly beats a bot
that looks healthy and transcribes static.

### Never pass `--bare` to `claude`

It is the flag the docs otherwise recommend for scripted use, and it is slated to
become the default for `-p`. It reads neither OAuth credentials nor
`CLAUDE_CODE_OAUTH_TOKEN`, so it is incompatible with **both** supported auth paths.
The headless bridge never passes it, and a test enforces that.

---

## Requirements

- **Python 3.12 exactly.** `kokoro` requires `<3.13`; `nemo-toolkit` 3.0 requires
  `>=3.12`. 3.12 is the only version that satisfies both.
- **CUDA GPU.** It will fall back to CPU and tell you, but not at conversational speed.
- **Claude Code**, logged in with a Claude subscription.
- System packages: `ffmpeg`, `espeak-ng`, `libopus`, and `tmux` if you use that bridge.

```bash
# Debian / Ubuntu
sudo apt install -y ffmpeg espeak-ng libopus0 libsodium23 tmux

# macOS
brew install ffmpeg espeak-ng opus libsodium tmux
```

`espeak-ng` is not optional: Kokoro's `misaki[en]` frontend falls back to it for words
outside its dictionary, which for a coding assistant is most identifiers.

---

## Install

```bash
git clone https://github.com/KoJesko/VoiceCode
cd VoiceCode

uv venv --python 3.12
uv pip install -e '.[nemo,dev]'      # or '.[hf,dev]' -- see below

cp .env.example .env
$EDITOR .env
```

### Choosing the ASR backend

The ASR model is an extra because NeMo is a large install and its 3.0 API surface for
the unified model is not yet exercised widely. Everything else works without it.

| | `nemo_unified` (default) | `hf_tdt` |
|---|---|---|
| Model | `nvidia/parakeet-unified-en-0.6b` | `nvidia/parakeet-tdt-0.6b-v3` |
| Install | `uv pip install -e '.[nemo]'` | `uv pip install -e '.[hf]'` |
| Download | ~2.5 GB | ~2.5 GB |
| Offline WER | 5.91 | 6.34 |
| Languages | English | 25 European languages |
| Licence | NVIDIA Open Model Licence | CC-BY-4.0 |
| Caveat | NeMo is a heavy dependency | needs `transformers` from source until `AutoModelForTDT` ships in a release |

Kokoro-82M is a further ~330 MB, downloaded on first run.

**Why offline transcription and not streaming.** `parakeet-unified` does support
streaming, but it exposes no feed-a-chunk API — the documented paths take file paths or
a dataset manifest — and it is *buffered*, recomputing 5.6 s of left context for every
chunk. Since the VAD already endpoints each turn, and the latency budget is measured
from end-of-speech, offline transcription starts exactly when the clock does, scores
better (5.91 vs 6.29 WER at 1.12 s), and avoids the recompute entirely.

---

## Discord setup

In the [developer portal](https://discord.com/developers/applications):

1. **New Application** → **Bot** → **Reset Token**, and put it in `DISCORD_TOKEN`.
2. Under **Privileged Gateway Intents**, enable **Server Members Intent**.
   This is required: resolving a speaking SSRC to a member is what every allowlist
   check depends on. *Message Content Intent is not needed* — the bot reads no
   message text.
3. **OAuth2 → URL Generator**: scopes `bot` and `applications.commands`; bot
   permissions **Connect**, **Speak**, **Use Voice Activity**, **Send Messages**,
   **Embed Links**. Invite it with the generated URL.
4. Turn on Developer Mode in Discord (Settings → Advanced) so you can right-click to
   **Copy ID** for the guild, channels, and users you need.

Fill in `.env`, then run `voicecode`.

---

## Auth: subscription, never API billing

This bot drives the `claude` CLI on your subscription. It never calls the Messages API
and never uses the Agent SDK in an API-key configuration.

Supported, in order of preference:

1. **Your existing `/login` credentials.** Nothing to configure.
2. **`CLAUDE_CODE_OAUTH_TOKEN`** from `claude setup-token` — for running as a systemd
   service or in a container. One year, Pro/Max/Team/Enterprise only, inference-scoped.

### Why the environment scrubbing is wider than you might expect

Claude Code's documented precedence puts **four** credential sources above your
subscription login:

1. Cloud provider (`CLAUDE_CODE_USE_BEDROCK` / `_VERTEX` / `_FOUNDRY`)
2. `ANTHROPIC_AUTH_TOKEN`
3. `ANTHROPIC_API_KEY` — *"In non-interactive mode (`-p`), the key is always used when
   present."* There is no approval prompt to catch it.
4. `apiKeyHelper`
5. `CLAUDE_CODE_OAUTH_TOKEN`
6. Anthropic profile / federation credentials
7. Subscription OAuth from `/login`

So VoiceCode strips **eight** variables from every subprocess it starts, not the two
that are obvious. `apiKeyHelper` is a settings key rather than an env var and cannot be
stripped, so it is detected at startup and refuses the run.

At startup the bot runs `claude auth status` under the scrubbed environment and logs
what it found. If that reports an API key or a cloud provider, it exits rather than
quietly billing you per token.

**The tmux bridge has a hole here that scrubbing cannot close.** The tmux server has its
own environment, inherited from whatever shell started it, long before this bot ran.
So `TmuxBridge.start()` reads the session's environment directly with
`tmux show-environment` and refuses if a credential is present there. If it complains:

```bash
tmux kill-session -t claude
env -u ANTHROPIC_API_KEY tmux new -s claude claude
```

### Rate limits

Agentic turns burn a shared subscription pool quickly. A usage limit opens a circuit
breaker: the bot speaks a short notice, posts the detail to the text channel, and
**refuses turns without contacting Claude at all** until the window passes. It does not
retry. Approximate turn counts are logged, and `/status` shows them.

---

## Choosing a bridge

### `headless` (default)

Runs `claude -p <text> --output-format stream-json --verbose --resume <session_id>` and
parses the JSON. Session IDs persist across turns.

Permission requests and rate limits arrive as **structured events**, which is what makes
them safe to act on.

### `tmux`

Injects into the interactive session you already have open:

```bash
tmux new -s claude claude          # in another terminal
# then in .env:
#   CLAUDE_BRIDGE=tmux
#   TMUX_SESSION=claude
```

It sends with `send-keys -l` — without `-l`, a transcript containing something like
"C-c" would be delivered as a keystroke — then diffs `capture-pane` output, filtering
prompts, spinners, box drawing, and token counters.

**It can only see rendered text.** Permission prompts and rate limits are therefore
recognised heuristically, flagged as such on every event, and reported as degraded by
`/status`. If permission safety matters more to you than driving your existing session,
use `headless`.

---

## Permission prompts are never answered automatically

When Claude Code needs permission, the bot:

1. stops the turn and latches;
2. says out loud what is being asked for;
3. posts the full prompt to the text channel;
4. **refuses every further utterance** until you run `/approve` or `/deny`.

Nothing spoken can release the latch. There is no matching of "yes" — say it as many
times as you like and the bot will keep telling you to use the slash command. Approval
is one-shot: it allows that single tool for one retry of the same request, then expires.

---

## Slash commands

Registered **guild-scoped to `GUILD_ALLOWLIST` only** — never globally — and every one
re-checks scope before acting.

| Command | |
|---|---|
| `/join [channel]` | Join and start listening. Refuses with a specific reason. |
| `/leave` | Disconnect. |
| `/mute [muted]` | Stop or resume transcribing without leaving. |
| `/voice <name>` | Change the Kokoro voice. |
| `/mode <gate>` | `always`, `wakeword`, or `ptt`. |
| `/ptt [open]` | Open or close push-to-talk. |
| `/stop` | Kill in-flight playback. |
| `/interrupt` | Interrupt the current Claude Code turn (ESC on tmux, SIGINT on headless). |
| `/approve`, `/deny` | Answer a pending permission prompt. The only way to do so. |
| `/reload` | Reload allowlists from `.env` and evict any now-disallowed connections. |
| `/status` | Auth method, scope, GPU, DAVE session, bridge degradation. |

---

## Channel scoping

Every allowlist **fails closed**: blank or unset denies everything. This inverts the
usual convention deliberately — a typo should silence the bot, not widen it.

There are twelve enforcement points, not one:

| # | Where | What it checks |
|---|---|---|
| 1 | Every gateway event | guild in `GUILD_ALLOWLIST` |
| 2 | Command registration | guild-scoped sync only, never global |
| 3 | Every slash command | guild **and** invoking user |
| 4 | `/join` | channel allowlisted **and** has a text binding |
| 5 | Bot's own voice state | dragged into a disallowed channel → immediate disconnect |
| 6 | Other members' voice state | auto-join / auto-leave |
| 7 | **`AudioSink.write`, first statement** | user in `USER_ALLOWLIST` |
| 8 | Speaking-state listeners | same user gate |
| 9 | Turn dispatch | re-checked against the *current* snapshot |
| 10 | Mirror write | binding resolved at write time, never cached |
| 11 | Playback start | channel still allowlisted |
| 12 | After `/reload` | sweep live connections, evict the now-disallowed |

Point 7 is the one that matters for privacy: it sits **above** DAVE decryption, so a
non-allowlisted speaker's audio is discarded while still encrypted. It never reaches a
decoder, a buffer, or the GPU.

Point 12 is what makes hot-reload real. Without it, removing a channel from the
allowlist would not remove the bot from it.

---

## Self-test

Runs the whole audio path with no Discord connection and prints per-stage latencies.

```bash
voicecode --selftest                    # Kokoro synthesizes a phrase, Parakeet reads it back
voicecode --selftest sample.wav         # use your own 16-bit wav
voicecode --selftest --selftest-bridge  # include a real Claude Code turn (uses one)
```

It checks the DAVE preflight and auth first, then times VAD, ASR, sanitisation, TTS and
framing, and writes the audio to `selftest-out/` so you can listen to what the bot would
have said.

---

## First run on real hardware

Everything above this line is verifiable without a GPU or a Discord connection. These two
steps are not, and they are the whole of the bring-up.

**1. The models — one command.**

```bash
voicecode --selftest
```

This is the only check that exercises NeMo loading `parakeet-unified-en-0.6b`, Kokoro
synthesis, and silero-vad. It fails loudly and per stage, so a missing `espeak-ng` or an
ASR backend that will not load is named rather than inferred. A pass means the local
inference half of the bot works.

**2. The voice connection — join, speak, `/status`.**

The receive path cannot be tested offline: it needs real DAVE-encrypted traffic from a
real Discord voice server. Join a channel, say one sentence, then run `/status`. The
**Receive path** section reports what actually happened to your audio:

```
in=312 decrypted=312 passthrough=0 dropped=0 decoded=312 decode_failed=0 utterances=1
healthy -- 312 frame(s) decoded, 1 utterance(s) endpointed
```

Do not judge this by whether the bot replied. Four unrelated faults produce the same
symptom — a bot that sits there silently — because the receive path fails without raising:
decrypting with the wrong key yields bytes that are not Opus, and the decoder discards
them at DEBUG level. The counters separate them:

| What `/status` shows | What it means |
|---|---|
| `in=0` | Nothing is arriving. Bot server-deafened, `voice_states` intent off, or the speaker is not in `USER_ALLOWLIST` — that gate drops audio before decryption. |
| `in` high, `decoded=0`, `passthrough` high | Frames are arriving still encrypted. The decrypt step is being skipped. **This is the failure `audio/dave.py` exists to prevent** — see the DAVE section at the top. |
| `in` high, `decoded=0`, `dropped` high | DAVE is active and rejecting our `decrypt` calls. A `davey` or `discord.py` version mismatch. |
| `decoded` high, `utterances=0` | Audio is fine; the VAD is the problem. Lower `VAD_THRESHOLD` or `MIN_UTTERANCE_MS`. |
| `healthy -- ...` | Working. |

You do not have to remember to check: after 5 seconds of audio with nothing surviving, the
bot logs one `WARNING` with the same verdict. One, not one per 20 ms frame.

---

## Latency

Target is **under 1.5 s** from end-of-speech to first spoken audio. Set
`LOG_LEVEL=DEBUG` for a per-turn breakdown:

```
turn user:123 | endpoint=0ms asr=180ms bridge_first=610ms tts_first=890ms first_frame=910ms | total=915ms
```

All stages are cumulative from end-of-speech, so the line reads as a budget. What helps:

- Both models load and warm up at startup with a dummy tensor, and stay resident.
- Synthesis is sentence-by-sentence, so playback starts on sentence one while Claude is
  still writing sentence three.
- Transcription starts at the endpoint, when the clock starts.

---

## Graceful degradation

| Failure | Behaviour |
|---|---|
| GPU OOM during synthesis | TTS disabled, voice connection and text mirror stay up |
| silero-vad unavailable | falls back to energy-based endpointing, logs it |
| Kokoro fails to load | bot starts anyway, text-only |
| ASR fails to load | startup aborts — without it there is no bot |
| DAVE preflight fails | startup aborts, rather than transcribing noise |
| Bound text channel unreachable | logged at ERROR; the bot refuses to join without a binding |

---

## Development

```bash
uv pip install -e '.[dev]'
pytest                    # 112 tests, no GPU or network required
ruff check src tests
```

The pure-logic modules — sanitising, scoping, config, resampling, endpointing, auth
scrubbing, stream parsing — are tested without a GPU or a Discord connection. That is
most of the code that can be wrong in a way tests catch.

`docs/DESIGN.md` records what was verified upstream, and why several parts of the design
depart from the obvious approach.

---

## Licence

MIT. Model licences differ: Kokoro-82M is Apache-2.0, `parakeet-tdt-0.6b-v3` is
CC-BY-4.0, and `parakeet-unified-en-0.6b` is under the NVIDIA Open Model Licence.
