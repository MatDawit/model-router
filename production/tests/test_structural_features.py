import pytest

from core.router.features.structural import STRUCTURAL_NAMES, structural_features, structural_matrix


def _by_name(text):
    return dict(zip(STRUCTURAL_NAMES, structural_features(text)))


def test_there_are_twelve_signals():
    assert len(STRUCTURAL_NAMES) == 12
    assert len(structural_features("hello")) == 12


def test_simple_question_values():
    f = _by_name("What is 2+2?")
    assert f["char_length"] == 12
    assert f["word_count"] == 3
    assert f["question_count"] == 1
    assert f["math_symbol_count"] == 1
    assert f["digit_ratio"] == pytest.approx(2 / 12)
    assert f["avg_word_length"] == pytest.approx(10 / 3)


def test_multiple_choice_markers_are_counted():
    text = "Which is a fruit?\nA. rock\nB. apple\nC. chair\nD. cloud"
    assert _by_name(text)["choice_marker_count"] == 4


def test_non_english_text_is_detected():
    assert _by_name("你好")["non_ascii_ratio"] == 1.0
    assert _by_name("hello")["non_ascii_ratio"] == 0.0


def test_code_markers_are_counted():
    assert _by_name("def f():\n    return 1")["code_marker_count"] >= 2


def test_empty_prompt_does_not_crash():
    f = _by_name("")
    assert f["char_length"] == 0
    assert f["avg_word_length"] == 0.0


def test_matrix_shape_including_empty_input():
    assert structural_matrix(["a", "b b", "c c c"]).shape == (3, 12)
    assert structural_matrix([]).shape == (0, 12)
