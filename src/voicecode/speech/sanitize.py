"""Turn Claude Code's terminal output into something worth listening to.

Claude Code emits mostly code, diffs, paths, tool calls and spinner frames. Read
verbatim by a TTS engine, that is unlistenable -- a path like
`src/voicecode/audio/dave.py` becomes twelve spoken syllables of no value, and a
progress bar becomes a minute of noise.

The rule this module follows: the bound text channel is the source of truth and gets
everything untouched. Speech is a summary, and it is allowed to drop anything that
does not carry meaning out loud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

# --- Patterns ------------------------------------------------------------------

# CSI sequences (colour, cursor movement) and OSC sequences (window titles, hyperlinks).
_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_OTHER = re.compile(r"\x1b[@-Z\\-_]")

# Box drawing, block elements, geometric shapes, braille (spinner frames live here).
_DECORATION = re.compile(r"[\u2500-\u257f\u2580-\u259f\u25a0-\u25ff\u2800-\u28ff]+")

# A fenced code block, with an optional language tag.
_FENCE = re.compile(r"^[ \t]*(?:```+|~~~+)[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*$", re.MULTILINE)

# A path is worth collapsing when it has a separator and a file-ish tail. Requiring a
# dotted extension keeps this off prose like "and/or" or "read/write".
_PATH = re.compile(r"(?<![\w.])((?:[\w.@~-]+)?(?:/[\w.@+-]+)+\.[A-Za-z0-9]{1,8})(?![\w/])")
# Directory paths with no extension, e.g. src/voicecode/audio/ or ./foo/bar
_DIR_PATH = re.compile(r"(?<![\w.])((?:\.{1,2})?(?:/[\w.@+-]+){2,})/?(?![\w/.])")

# Claude Code's tool-call gutter markers, and common status furniture.
_TOOL_LINE = re.compile(
    r"^\s*(?:[\u23fa\u25cf\u25cb\u2b24\u2514\u251c\u23bf\u2ba1\u21b3>]|\u2937)\s*"
    r"(?:[A-Z][A-Za-z]*\s*\(|Running|Reading|Writing|Editing|Searching|Fetching)"
)
_TOOL_KEYWORDS = re.compile(
    r"^\s*(?:tool_use|tool_result|\[tool\]|Tool use:|Called the \w+ tool)\b", re.IGNORECASE
)
# A line that is almost entirely one repeated character is a rule or a progress bar.
_BAR = re.compile(r"^\s*(.)\1{6,}\s*$")
_PROGRESS = re.compile(r"^\s*\[?[=#\->.\s]{8,}\]?\s*(?:\d{1,3}\s*%)?\s*$")
_SPINNER_ASCII = re.compile(r"^\s*[|/\\-]\s*$")
# "12/48 files" style counters, and bare percentages.
_COUNTER = re.compile(r"^\s*\d+\s*/\s*\d+\s*\w*\s*$|^\s*\d{1,3}\s*%\s*$")

# A row of a drawn box: decoration at both ends with content between.
_BOXED = re.compile(r"^[\u2500-\u257f].*[\u2500-\u257f]$", re.DOTALL)
# A spinner frame followed by a status word ("\u28f9 Thinking\u2026").
_SPINNER_PREFIX = re.compile(r"^[\u2800-\u28ff\u25cf\u25cb\u25e6]+\s*\S")
_DIFF_LINE = re.compile(r"^\s*(?:[+-]{3}\s|@@\s|[+-]\s?\S)")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_MD_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_WS = re.compile(r"[ \t]{2,}")

_LANG_NAMES = {
    "py": "Python", "python": "Python", "js": "JavaScript", "javascript": "JavaScript",
    "ts": "TypeScript", "typescript": "TypeScript", "tsx": "TypeScript", "jsx": "JavaScript",
    "sh": "shell", "bash": "shell", "zsh": "shell", "console": "shell", "shell": "shell",
    "rs": "Rust", "rust": "Rust", "go": "Go", "golang": "Go", "java": "Java",
    "c": "C", "cpp": "C++", "cc": "C++", "h": "C", "hpp": "C++", "cs": "C sharp",
    "rb": "Ruby", "ruby": "Ruby", "php": "PHP", "swift": "Swift", "kt": "Kotlin",
    "sql": "SQL", "html": "HTML", "css": "CSS", "scss": "CSS",
    "json": "JSON", "yaml": "YAML", "yml": "YAML", "toml": "TOML", "xml": "XML",
    "md": "Markdown", "markdown": "Markdown", "diff": "diff", "patch": "diff",
    "": "code",
}


@dataclass(frozen=True, slots=True)
class SpeechResult:
    """What to say, and whether saying it lost anything."""

    spoken: str
    truncated: bool
    dropped_lines: int
    code_blocks: int

    @property
    def has_speech(self) -> bool:
        return bool(self.spoken.strip())


def strip_ansi(text: str) -> str:
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_CSI.sub("", text)
    return _ANSI_OTHER.sub("", text)


def describe_language(tag: str) -> str:
    return _LANG_NAMES.get(tag.strip().lower(), tag.strip() or "code")


def replace_code_blocks(text: str) -> tuple[str, int]:
    """Swap fenced blocks for "<N> lines of <lang>". Returns (text, block_count).

    Walks fences pairwise rather than using a single regex, so an unterminated fence
    (common in streamed output) consumes to end-of-text instead of silently swallowing
    the prose after it.
    """
    fences = list(_FENCE.finditer(text))
    if not fences:
        return text, 0

    out: list[str] = []
    cursor = 0
    count = 0
    i = 0
    while i < len(fences):
        opener = fences[i]
        closer = fences[i + 1] if i + 1 < len(fences) else None
        out.append(text[cursor : opener.start()])

        body_start = opener.end()
        body_end = closer.start() if closer else len(text)
        body = text[body_start:body_end].strip("\n")
        n_lines = len([ln for ln in body.split("\n") if ln.strip()]) if body else 0
        lang = describe_language(opener.group(1))

        if n_lines == 0:
            out.append(f"an empty {lang} block.")
        elif n_lines == 1:
            out.append(f"one line of {lang}.")
        else:
            out.append(f"{n_lines} lines of {lang}.")
        count += 1

        cursor = closer.end() if closer else len(text)
        i += 2

    out.append(text[cursor:])
    return "".join(out), count


def collapse_paths(text: str) -> str:
    """Reduce paths to their basename. `src/audio/dave.py` -> `dave.py`."""
    text = _PATH.sub(lambda m: m.group(1).rsplit("/", 1)[-1], text)

    def _dir(match: re.Match[str]) -> str:
        tail = match.group(1).rstrip("/").rsplit("/", 1)[-1]
        return f"{tail}/" if tail else match.group(0)

    return _DIR_PATH.sub(_dir, text)


def is_noise_line(line: str) -> bool:
    """True for lines that carry no meaning when spoken aloud."""
    stripped = line.strip()
    if not stripped:
        return False  # blank lines are paragraph structure, handled later
    # Chrome first: a boxed row or a spinner frame is furniture no matter what text
    # it wraps, so these are checked before the content-based rules below.
    if _BOXED.match(stripped) or _SPINNER_PREFIX.match(stripped):
        return True
    if _TOOL_LINE.search(line) or _TOOL_KEYWORDS.match(line):
        return True
    if _DIFF_LINE.match(line):
        return True
    # The remaining rules run against the decoration-stripped text, so a progress bar
    # drawn with block characters is recognised by the counter it carries.
    bare = _DECORATION.sub(" ", stripped).strip()
    if not re.search(r"[A-Za-z0-9]", bare):
        return True
    if _BAR.match(bare) or _PROGRESS.match(bare) or _SPINNER_ASCII.match(bare):
        return True
    if _COUNTER.match(bare):
        return True
    return False


def _flatten_markdown(text: str) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    return text


def split_sentences(text: str) -> Iterator[str]:
    """Yield sentences, so synthesis can start before the full response arrives.

    Deliberately simple, but it does avoid the two splits that sound worst: inside a
    decimal or version number, and after a common abbreviation.
    """
    guard = re.sub(r"(\d)\.(\d)", "\\1\u0000\\2", text)
    guard = re.sub(
        r"\b(Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e|approx|Fig|No|Inc|Ltd|Jr|Sr)\.",
        lambda m: m.group(1) + "\u0000",
        guard,
        flags=re.IGNORECASE,
    )
    for chunk in re.split(r"(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z0-9])|\n{2,}", guard):
        if chunk is None:
            continue
        sentence = chunk.replace("\u0000", ".").strip()
        if sentence:
            yield sentence


def truncate_for_speech(text: str, limit: int) -> tuple[str, bool]:
    """Cap spoken length, preferring a sentence boundary over a hard cut."""
    if len(text) <= limit:
        return text, False

    kept: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        if used + len(sentence) + 1 > limit:
            break
        kept.append(sentence)
        used += len(sentence) + 1

    if kept:
        return " ".join(kept), True

    # A single sentence longer than the whole budget: cut at a word boundary.
    head = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (head or text[:limit]), True


def sanitize_for_speech(text: str, char_limit: int = 600) -> SpeechResult:
    """Full pipeline: raw Claude Code output in, speakable prose out."""
    if not text or not text.strip():
        return SpeechResult("", False, 0, 0)

    cleaned = strip_ansi(text)
    cleaned, code_blocks = replace_code_blocks(cleaned)

    kept: list[str] = []
    dropped = 0
    for line in cleaned.split("\n"):
        if is_noise_line(line):
            dropped += 1
            continue
        kept.append(_DECORATION.sub(" ", line))

    body = "\n".join(kept)
    body = _flatten_markdown(body)
    body = _URL.sub("a link", body)
    body = collapse_paths(body)

    # Collapse whitespace, but keep paragraph breaks as sentence boundaries.
    paragraphs = []
    for para in re.split(r"\n{2,}", body):
        joined = " ".join(_WS.sub(" ", ln).strip() for ln in para.split("\n") if ln.strip())
        if joined:
            paragraphs.append(joined)
    body = "\n\n".join(paragraphs).strip()
    body = re.sub(r" +([.,;:!?])", r"\1", body)

    spoken, truncated = truncate_for_speech(body, char_limit)
    return SpeechResult(
        spoken=spoken.strip(),
        truncated=truncated,
        dropped_lines=dropped,
        code_blocks=code_blocks,
    )
