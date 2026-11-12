"""12 cheap, hand-built signals about a prompt's shape, designed for REAL benchmarks
(multiple choice, math word problems, code, non-English text), not the Phase 1
synthetic generator.
"""

import re

import numpy as np

STRUCTURAL_NAMES = [
    "char_length",
    "word_count",
    "line_count",
    "avg_word_length",
    "digit_ratio",
    "upper_ratio",
    "non_ascii_ratio",
    "math_symbol_count",
    "question_count",
    "code_marker_count",
    "choice_marker_count",
    "instruction_word_count",
]

_MATH_SYMBOLS = "+*/=<>%^"
_CODE_MARKERS = ("def ", "class ", "import ", "return ", "```", "->", "==", "();")
_INSTRUCTION_WORDS = ("must", "only", "exactly", "ensure", "do not", "don't", "step by step", "explain", "write")
# Multiple-choice options such as "A.", "B)", "(C)" at the start of a line or after a space.
_CHOICE_RE = re.compile(r"(?:^|\s)\(?[A-D][\).:]\s", re.MULTILINE)


def structural_features(text):
    """Return the 12 signals for one prompt, in STRUCTURAL_NAMES order."""
    n_chars = max(len(text), 1)  # avoid dividing by zero on an empty prompt
    words = text.split()
    n_words = len(words)
    lower = text.lower()
    return [
        float(len(text)),
        float(n_words),
        float(text.count("\n") + 1),
        (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        sum(c.isdigit() for c in text) / n_chars,
        sum(c.isupper() for c in text) / n_chars,
        sum(ord(c) > 127 for c in text) / n_chars,
        float(sum(text.count(s) for s in _MATH_SYMBOLS)),
        float(text.count("?")),
        float(sum(lower.count(m) for m in _CODE_MARKERS)),
        float(len(_CHOICE_RE.findall(text))),
        float(sum(lower.count(w) for w in _INSTRUCTION_WORDS)),
    ]


def structural_matrix(texts):
    """Stack the signals for many prompts into an (n_prompts, 12) array."""
    rows = [structural_features(t) for t in texts]
    return np.array(rows, dtype=float).reshape(len(rows), len(STRUCTURAL_NAMES))
