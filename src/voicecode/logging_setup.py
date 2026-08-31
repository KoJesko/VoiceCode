"""Logging plus the per-stage latency instrumentation the spec asks for.

The target is <1.5s from end-of-speech to first spoken audio. `TurnTimer` records each
stage against a single t0 (the endpoint decision) so the DEBUG line reads as a
cumulative budget rather than a set of unrelated durations.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger("voicecode.latency")

# Stage order is fixed so the log line is scannable across turns.
STAGES = ("endpoint", "asr", "bridge_first", "tts_first", "first_frame")


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These are chatty at DEBUG and drown out our own timing lines.
    for noisy in ("discord", "websockets", "nemo_logger", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(max(root.level, logging.INFO))


@dataclass
class TurnTimer:
    """Cumulative stage timings for one voice turn, measured from end-of-speech."""

    label: str = ""
    t0: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str) -> float:
        """Record milliseconds elapsed since t0. Returns the value."""
        elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
        # First write wins: "first_frame" should mean the first, not the latest.
        self.marks.setdefault(stage, elapsed_ms)
        return elapsed_ms

    @contextmanager
    def stage(self, stage: str):
        """Mark `stage` when the block exits, even on exception."""
        try:
            yield
        finally:
            self.mark(stage)

    def total_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0

    def emit(self) -> None:
        parts = [f"{s}={self.marks[s]:.0f}ms" for s in STAGES if s in self.marks]
        extra = [f"{k}={v:.0f}ms" for k, v in self.marks.items() if k not in STAGES]
        log.debug(
            "turn %s | %s | total=%.0fms",
            self.label or "-",
            " ".join(parts + extra) or "no stages",
            self.total_ms(),
        )
