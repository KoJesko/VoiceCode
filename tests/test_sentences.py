"""Sentence streaming is what lets playback start before synthesis finishes."""

from __future__ import annotations

from voicecode.tts.sentences import SentenceStreamer


def test_sentences_release_as_soon_as_they_complete():
    streamer = SentenceStreamer()
    assert streamer.feed("I fixed the b") == []
    assert streamer.feed("ug in dave.py. ") == ["I fixed the bug in dave.py."]


def test_remainder_is_held_until_flush():
    streamer = SentenceStreamer()
    streamer.feed("Done. And then")
    assert streamer.pending.strip() == "And then"
    assert streamer.flush() == ["And then"]


def test_flush_when_empty_returns_nothing():
    assert SentenceStreamer().flush() == []


def test_a_short_complete_sentence_is_released_immediately():
    """"Done." is a whole answer; holding it back would delay the brief-reply case."""
    assert SentenceStreamer().feed("Done. ") == ["Done."]


def test_items_with_no_letters_are_not_spoken():
    """A stray "3." from a list is not a sentence."""
    streamer = SentenceStreamer()
    assert streamer.feed("3. ") == []


def test_multiple_sentences_in_one_delta():
    streamer = SentenceStreamer()
    released = streamer.feed("First one here. Second one here. ")
    assert len(released) == 2
