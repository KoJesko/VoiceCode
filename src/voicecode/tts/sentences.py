"""Incremental sentence assembly for streamed synthesis.

The bridge yields prose in chunks that do not align with sentences. Waiting for the
whole response before synthesising would put the entire generation time on the critical
path. This accumulates deltas and releases each sentence the moment it is complete, so
Kokoro starts on sentence one while the model is still writing sentence three.

`flush()` releases whatever is left when the turn ends, since the final sentence often
arrives without trailing punctuation.
"""

from __future__ import annotations

import re

from ..speech.sanitize import split_sentences

# A sentence is releasable once terminal punctuation is followed by a space or newline.
_TERMINAL = re.compile(r"[.!?][\"')\]]?(?=\s)|\n\n")
# Everything released has already been cut at terminal punctuation, so short is not the
# same as incomplete -- "Done." is a whole answer, and holding it back would delay
# speech in exactly the case where Claude replies briefly. The only thing worth
# suppressing is an item with no letters in it, such as a stray "3." from a list.
_HAS_LETTERS = re.compile(r"[A-Za-z]")


class SentenceStreamer:
    """Feed text deltas, take complete sentences out."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, delta: str) -> list[str]:
        """Add a chunk of text. Returns any sentences that are now complete."""
        if not delta:
            return []
        self._pending += delta

        last = None
        for match in _TERMINAL.finditer(self._pending):
            last = match
        if last is None:
            return []

        head = self._pending[: last.end()]
        tail = self._pending[last.end() :]

        released = [s for s in split_sentences(head) if _HAS_LETTERS.search(s)]
        self._pending = tail
        return released

    def flush(self) -> list[str]:
        """Release everything remaining, terminal punctuation or not."""
        remainder = self._pending.strip()
        self._pending = ""
        if not remainder:
            return []
        return [s for s in split_sentences(remainder) if s.strip()] or [remainder]

    @property
    def pending(self) -> str:
        return self._pending
