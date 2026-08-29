"""The sanitizer decides what a listener actually hears."""

from __future__ import annotations

from voicecode.speech.sanitize import (
    collapse_paths,
    replace_code_blocks,
    sanitize_for_speech,
    split_sentences,
    strip_ansi,
    truncate_for_speech,
)


def test_ansi_is_stripped():
    assert strip_ansi("\x1b[32mgreen\x1b[0m text") == "green text"
    assert strip_ansi("\x1b]0;title\x07body") == "body"


def test_code_block_becomes_a_line_count():
    text = "Here:\n```python\na = 1\nb = 2\n```\nDone."
    out, count = replace_code_blocks(text)
    assert count == 1
    assert "2 lines of Python." in out
    assert "a = 1" not in out


def test_single_line_block_reads_naturally():
    out, _ = replace_code_blocks("```sh\nls\n```")
    assert "one line of shell." in out


def test_unterminated_fence_does_not_swallow_following_prose():
    """Streamed output often arrives with a fence still open."""
    out, count = replace_code_blocks("Intro.\n```python\nx = 1\n")
    assert count == 1
    assert "Intro." in out


def test_paths_collapse_to_basenames():
    assert collapse_paths("edit /home/u/proj/main.py now") == "edit main.py now"
    assert collapse_paths("in src/voicecode/audio/dave.py") == "in dave.py"


def test_path_collapsing_leaves_prose_alone():
    """'and/or' is not a path."""
    assert collapse_paths("read and/or write") == "read and/or write"


def test_terminal_chrome_is_dropped():
    pane = "\n".join([
        "⏺ Bash(pytest -q)",
        "  ⎿  Running…",
        "╭────────────────╮",
        "│ Editing main.py │",
        "╰────────────────╯",
        "⠹ Thinking…",
        "The tests pass now.",
        "████████████ 100%",
    ])
    result = sanitize_for_speech(pane)
    assert result.spoken == "The tests pass now."
    assert result.dropped_lines >= 7


def test_diff_lines_are_dropped():
    result = sanitize_for_speech("I changed it.\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+import os")
    assert result.spoken == "I changed it."


def test_prose_that_looks_like_chrome_survives():
    """Over-eager filtering is worse than under-eager: it eats the answer."""
    text = "Running the tests fixed it. I read config.py and it works."
    assert sanitize_for_speech(text).spoken == text


def test_markdown_links_are_flattened_not_mangled():
    result = sanitize_for_speech("See [the docs](https://example.com/a/b) for details.")
    assert result.spoken == "See the docs for details."


def test_hard_wrapped_lines_join_into_one_sentence():
    result = sanitize_for_speech("The decoder was\nreading the wrong file.")
    assert "\n" not in result.spoken


def test_truncation_prefers_a_sentence_boundary():
    text = " ".join(f"Sentence {i} here." for i in range(1, 40))
    spoken, truncated = truncate_for_speech(text, 120)
    assert truncated
    assert len(spoken) <= 120
    assert spoken.endswith(".")


def test_truncation_falls_back_to_a_word_boundary():
    spoken, truncated = truncate_for_speech("word " * 200, 50)
    assert truncated
    assert len(spoken) <= 50
    assert not spoken.endswith(" ")


def test_version_numbers_do_not_split_sentences():
    assert list(split_sentences("Version 2.7.1 works.")) == ["Version 2.7.1 works."]


def test_abbreviations_do_not_split_sentences():
    assert len(list(split_sentences("Run it e.g. like this. Then stop."))) == 2


def test_empty_input_is_not_speech():
    assert not sanitize_for_speech("").has_speech
    assert not sanitize_for_speech("   \n\n ").has_speech
