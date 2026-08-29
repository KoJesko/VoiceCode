"""Entry point: `voicecode` to run the bot, `voicecode --selftest` to test the pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .asr.base import ASR_SAMPLE_RATE, build_engine
from .bridge.base import AuthError, BridgeError, EventKind
from .config import BridgeKind, ConfigStore, describe_scope
from .logging_setup import setup_logging
from .speech.sanitize import sanitize_for_speech
from .tts.kokoro_engine import KokoroTTS

log = logging.getLogger("voicecode")


# -- shared construction ---------------------------------------------------------

def build_bridge(config: ConfigStore):
    settings = config.settings
    if settings.claude_bridge is BridgeKind.TMUX:
        from .bridge.tmux import TmuxBridge

        if not settings.tmux_session:
            raise BridgeError(
                "CLAUDE_BRIDGE=tmux but TMUX_SESSION is not set.\n"
                "Run `tmux ls` to find the session running Claude Code and set "
                "TMUX_SESSION in .env, or set CLAUDE_BRIDGE=headless to run Claude "
                "Code as a subprocess instead."
            )
        return TmuxBridge(
            session=settings.tmux_session,
            claude_binary=settings.claude_binary,
            poll_interval_ms=settings.tmux_poll_interval_ms,
            idle_settle_ms=settings.tmux_idle_settle_ms,
        )

    from .bridge.headless import HeadlessBridge

    return HeadlessBridge(
        claude_binary=settings.claude_binary,
        cwd=settings.claude_cwd or None,
        permission_mode=settings.claude_permission_mode,
    )


def load_wav_mono16(path: Path) -> np.ndarray:
    """Read a wav file to 16 kHz mono float32, resampling if needed."""
    from .audio.resample import pcm_bytes_to_mono_float, resample

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM wav files are supported")
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    mono = pcm_bytes_to_mono_float(raw, channels=channels)
    return resample(mono, rate, ASR_SAMPLE_RATE)


def write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    ints = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(ints.tobytes())


# -- selftest --------------------------------------------------------------------

async def run_selftest(
    config: ConfigStore, wav_path: Path | None, use_bridge: bool, out_dir: Path
) -> int:
    """Exercise the whole audio path with no Discord connection.

    With no wav file, Kokoro synthesizes a known phrase and Parakeet transcribes it
    back -- a genuine round trip that also tells you whether the two models agree
    about what English sounds like.
    """
    from .audio.dave import DaveUnavailable, preflight
    from .audio.resample import KOKORO_RATE, discord_frames_from_float
    from .audio.turn import TurnBuffer
    from .audio.vad import SileroVad

    settings = config.settings
    stages: list[tuple[str, float]] = []
    failures: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def stage(name: str, started: float) -> None:
        stages.append((name, (time.perf_counter() - started) * 1000.0))

    print("VoiceCode selftest")
    print("=" * 62)

    # 0. Environment preflight ---------------------------------------------------
    try:
        preflight()
        print("  DAVE receive path .......... available")
    except DaveUnavailable as exc:
        failures.append(f"DAVE: {exc}")
        print(f"  DAVE receive path .......... FAILED: {exc}")

    try:
        auth_bridge = build_bridge(config)
        await auth_bridge.start()
        health = await auth_bridge.health()
        print(f"  Claude Code auth ........... {health.auth_method}")
    except (AuthError, BridgeError) as exc:
        auth_bridge = None
        failures.append(f"auth/bridge: {exc}")
        print(f"  Claude Code auth ........... FAILED: {exc}")

    # 1. TTS ---------------------------------------------------------------------
    tts = KokoroTTS(settings.tts_lang_code, settings.tts_voice, settings.tts_speed)
    phrase = "Refactor the resampler and run the tests."
    started = time.perf_counter()
    try:
        tts.load()
        stage("tts_load", started)
    except Exception as exc:
        failures.append(f"TTS load: {exc}")
        print(f"  Kokoro load ................ FAILED: {exc}")
        tts = None  # type: ignore[assignment]

    # 2. Input audio -------------------------------------------------------------
    if wav_path is not None:
        audio16 = load_wav_mono16(wav_path)
        print(f"  input ...................... {wav_path} "
              f"({audio16.size / ASR_SAMPLE_RATE:.2f}s)")
    elif tts is not None:
        started = time.perf_counter()
        speech = tts.synthesize(phrase)
        stage("tts_synth", started)
        from .audio.resample import resample

        audio16 = resample(speech.audio, KOKORO_RATE, ASR_SAMPLE_RATE)
        write_wav(out_dir / "selftest_input.wav", speech.audio, KOKORO_RATE)
        print(f"  input ...................... synthesized {audio16.size / ASR_SAMPLE_RATE:.2f}s "
              f'from "{phrase}"')
    else:
        print("  input ...................... unavailable")
        print("\nCannot continue: no wav file was given and Kokoro could not be loaded "
              "to synthesize one.")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    # 3. VAD / endpointing -------------------------------------------------------
    started = time.perf_counter()
    try:
        SileroVad.load()
        vad = SileroVad(settings.vad_threshold)
        buffer = TurnBuffer(
            user_id=0,
            vad=vad,
            endpoint_silence_ms=settings.endpoint_silence_ms,
            min_utterance_ms=settings.min_utterance_ms,
        )
        # Feed the audio plus a tail of silence so the endpoint actually fires.
        padded = np.concatenate([audio16, np.zeros(ASR_SAMPLE_RATE, dtype=np.float32)])
        events = []
        for offset in range(0, padded.size, 320):
            events += buffer._advance(padded[offset : offset + 320])
        stage("vad_endpoint", started)
        kinds = [e.kind.value for e in events]
        print(f"  VAD ........................ {len(events)} event(s): {kinds}")
    except Exception as exc:
        failures.append(f"VAD: {exc}")
        print(f"  VAD ........................ FAILED: {exc}")

    # 4. ASR ---------------------------------------------------------------------
    transcript_text = ""
    started = time.perf_counter()
    try:
        asr = build_engine(settings.asr_backend.value, settings.asr_model_id, settings.asr_device)
        asr.load()
        stage("asr_load", started)
        started = time.perf_counter()
        transcript = asr.transcribe(audio16)
        stage("asr", started)
        transcript_text = transcript.text
        print(f"  ASR ........................ {transcript_text!r}")
    except Exception as exc:
        failures.append(f"ASR: {exc}")
        print(f"  ASR ........................ FAILED: {exc}")

    # 5. Bridge (optional) -------------------------------------------------------
    reply = (
        "I updated `src/audio/resample.py`.\n\n"
        "```python\ndef resample(x):\n    return soxr.resample(x)\n```\n\n"
        "All 14 tests pass."
    )
    if use_bridge and auth_bridge is not None and transcript_text:
        started = time.perf_counter()
        collected: list[str] = []
        try:
            async for event in auth_bridge.send(transcript_text):
                if event.kind is EventKind.PROSE:
                    if not collected:
                        stage("bridge_first", started)
                    collected.append(event.text)
                elif event.kind in (EventKind.DONE, EventKind.ERROR, EventKind.RATE_LIMIT):
                    break
            stage("bridge_total", started)
            reply = "\n".join(collected) or reply
            print(f"  bridge ..................... {len(reply)} chars from Claude Code")
        except Exception as exc:
            failures.append(f"bridge: {exc}")
            print(f"  bridge ..................... FAILED: {exc}")
    else:
        print("  bridge ..................... skipped (pass --selftest-bridge to include)")

    # 6. Sanitize ----------------------------------------------------------------
    started = time.perf_counter()
    result = sanitize_for_speech(reply, settings.speak_char_limit)
    stage("sanitize", started)
    print(f"  sanitize ................... {result.spoken!r}")
    print(f"                               (dropped {result.dropped_lines} line(s), "
          f"{result.code_blocks} code block(s), truncated={result.truncated})")

    # 7. TTS + framing -----------------------------------------------------------
    if tts is not None and result.has_speech:
        started = time.perf_counter()
        try:
            speech = tts.synthesize(result.spoken)
            stage("tts", started)
            started = time.perf_counter()
            frames = discord_frames_from_float(speech.audio)
            stage("first_frame", started)
            write_wav(out_dir / "selftest_output.wav", speech.audio, KOKORO_RATE)
            print(f"  TTS ........................ {speech.duration_ms:.0f}ms audio, "
                  f"{len(frames)} Discord frame(s)")
            print(f"  wrote ...................... {out_dir / 'selftest_output.wav'}")
        except Exception as exc:
            failures.append(f"TTS synth: {exc}")
            print(f"  TTS ........................ FAILED: {exc}")

    if auth_bridge is not None:
        await auth_bridge.close()

    # -- report ------------------------------------------------------------------
    print("\nper-stage latency")
    print("-" * 62)
    for name, ms in stages:
        print(f"  {name:.<28} {ms:8.1f} ms")

    hot = [ms for name, ms in stages if name in ("asr", "bridge_first", "tts", "first_frame")]
    if hot:
        total = sum(hot)
        budget = "within" if total < 1500 else "OVER"
        label = "end-of-speech to first frame"
        print(f"  {label:.<28} {total:8.1f} ms  ({budget} the 1.5s target)")

    if failures:
        print(f"\n{len(failures)} stage(s) failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall stages passed")
    return 0


# -- bot -------------------------------------------------------------------------

async def run_bot(config: ConfigStore) -> int:
    from .audio.dave import DaveUnavailable, preflight
    from .audio.vad import SileroVad
    from .discord_app.bot import VoiceCodeBot

    settings = config.settings
    if not settings.discord_token.get_secret_value():
        log.error("DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.")
        return 2

    try:
        preflight()
    except DaveUnavailable as exc:
        log.error(
            "voice receive cannot work in this environment: %s\n"
            "Discord enforces DAVE end-to-end encryption on non-Stage voice channels; "
            "without it the bot would join and hear nothing.",
            exc,
        )
        return 2

    bridge = build_bridge(config)
    try:
        await bridge.start()
    except AuthError as exc:
        log.error("%s", exc)
        return 2
    except BridgeError as exc:
        log.error("bridge failed to start: %s", exc)
        return 2

    asr = build_engine(settings.asr_backend.value, settings.asr_model_id, settings.asr_device)
    asr.load()

    tts = KokoroTTS(settings.tts_lang_code, settings.tts_voice, settings.tts_speed)
    try:
        tts.load()
    except Exception as exc:
        # Degrade rather than refuse to start: a bot that hears and answers in text
        # is useful; one that will not connect is not.
        log.error("TTS unavailable, continuing without speech: %s", exc)
        tts.disable(str(exc))

    try:
        SileroVad.load()
    except Exception:
        log.exception("silero-vad unavailable; endpointing will use the energy fallback")

    log.info("scope: %s", describe_scope(config.snapshot))

    bot = VoiceCodeBot(config=config, asr=asr, tts=tts, bridge=bridge)
    try:
        await bot.start(settings.discord_token.get_secret_value())
    except KeyboardInterrupt:
        pass
    finally:
        await bot.close()
    return 0


# -- cli -------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voicecode",
        description="Full-duplex Discord voice frontend for a Claude Code session.",
    )
    parser.add_argument(
        "--selftest",
        nargs="?",
        const="",
        metavar="WAV",
        help="Run the full audio path with no Discord connection. Optionally give a "
             "16-bit wav file; with no file, Kokoro synthesizes one and Parakeet "
             "transcribes it back.",
    )
    parser.add_argument(
        "--selftest-bridge",
        action="store_true",
        help="Include a real Claude Code turn in the selftest (uses a subscription turn).",
    )
    parser.add_argument(
        "--out-dir", default="selftest-out", help="Where the selftest writes wav files"
    )
    parser.add_argument("--env-file", default=".env", help="Path to the .env file")
    parser.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ConfigStore(env_file=args.env_file)
    setup_logging(args.log_level or config.settings.log_level)

    if args.selftest is not None:
        wav = Path(args.selftest) if args.selftest else None
        if wav is not None and not wav.is_file():
            print(f"no such wav file: {wav}", file=sys.stderr)
            return 2
        return asyncio.run(
            run_selftest(config, wav, args.selftest_bridge, Path(args.out_dir))
        )

    try:
        return asyncio.run(run_bot(config))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
