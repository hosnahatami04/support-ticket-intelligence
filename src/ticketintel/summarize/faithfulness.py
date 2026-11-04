from __future__ import annotations

import re
from dataclasses import dataclass, field

# Entities that must be grounded in the source. A summary is free to paraphrase
# prose, but it may not introduce a number, address or identifier of its own.
_NUMBER = re.compile(r"\d[\d,.]*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
# Identifier-ish tokens: an alphanumeric code with at least one digit and one
# letter, e.g. order numbers, SKUs, error codes, ticket ids.
_CODE = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9-]{3,}\b")

_WS = re.compile(r"\s+")
# Word-level tokens for the quote check. The entity check matches substrings
# and needs punctuation left in place, so this is deliberately separate: a
# quote of "down" must match a source reading "down." rather than be reported
# as a fabrication.
_WORD = re.compile(r"[a-z0-9']+")
# Negations invert a phrase's meaning, so dropping one turns a grounded quote
# into a claim the ticket contradicts. They are tracked rather than treated as
# ordinary words.
_NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "without",
        "cannot",
        "cant",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "hasnt",
        "havent",
        "didnt",
        "doesnt",
        "dont",
        "wont",
        "nothing",
        "nobody",
        "neither",
        "nor",
    }
)


@dataclass
class EntityCheck:
    unsupported: list[str] = field(default_factory=list)

    @property
    def is_faithful(self) -> bool:
        return not self.unsupported


def _normalise(text: str) -> str:
    return _WS.sub(" ", text.lower())


def _normalise_number(token: str) -> str:
    # Compare numbers by value-ish form so "1,200" in the summary matches
    # "1200" in the source and a trailing period is not treated as a decimal.
    return token.replace(",", "").rstrip(".")


def extract_entities(text: str) -> list[str]:
    found: list[str] = []
    found.extend(_EMAIL.findall(text))
    found.extend(_URL.findall(text))
    found.extend(_CODE.findall(text))
    found.extend(_NUMBER.findall(text))
    return found


def check_entities(summary_texts: list[str], source: str) -> EntityCheck:
    source_norm = _normalise(source)
    source_numbers = {_normalise_number(n) for n in _NUMBER.findall(source_norm)}
    unsupported: list[str] = []

    for text in summary_texts:
        for entity in extract_entities(text):
            entity_norm = _normalise(entity)
            if _NUMBER.fullmatch(entity):
                if _normalise_number(entity_norm) in source_numbers:
                    continue
            elif entity_norm in source_norm:
                continue
            unsupported.append(entity)
    return EntityCheck(unsupported=unsupported)


def split_sentences(text: str) -> list[str]:
    # Deliberately simple: the summary fields are short, and a heavier sentence
    # splitter would add a dependency for no measurable gain.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def ungrounded_quotes(
    quoted: list[str],
    source: str,
    min_overlap: float = 0.6,
    min_pair_support: float = 0.5,
) -> list[str]:
    """Quoted phrases whose wording is largely absent from the source.

    The entity check only catches numbers, addresses and identifiers, so a
    summary can invent a whole phrase and still score as faithful. Fields that
    are meant to be quotations from the ticket - urgency signals above all -
    are checked here by word overlap, which tolerates light paraphrase but
    catches a phrase the ticket never contained.
    """
    source_words = _words(source)
    flagged: list[str] = []
    for phrase in quoted:
        words = _words_in_order(phrase)
        if not words:
            continue
        overlap = sum(1 for w in words if w in source_words) / len(words)
        if overlap < min_overlap:
            flagged.append(phrase)
            continue
        # Word overlap ignores order, so a phrase can score perfectly while
        # saying the opposite of the ticket: "the system is down" against a
        # source reading "the system is not down". Require the phrase's own
        # negation count to be supported by the source wording.
        if _drops_a_negation(words, source):
            flagged.append(phrase)
            continue
        # Overlap also cannot tell a quotation from a bag of the ticket's words
        # reassembled into a phrase it never contained. Adjacent word pairs do
        # distinguish them, and a real quotation keeps its pairs intact.
        if _pair_support(words, source) < min_pair_support:
            flagged.append(phrase)
    return flagged


def _pair_support(words: list[str], source: str) -> float:
    """Fraction of the phrase's adjacent word pairs that occur in the source.

    A single word has no pairs and is fully supported by the overlap test
    alone, so it scores 1.0 rather than being penalised for being short.
    """
    pairs = _adjacent_pairs(words)
    if not pairs:
        return 1.0
    source_pairs = _adjacent_pairs(_words_in_order(source))
    return sum(1 for pair in pairs if pair in source_pairs) / len(pairs)


def _adjacent_pairs(words: list[str]) -> set[tuple[str, str]]:
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def _words(text: str) -> set[str]:
    return set(_WORD.findall(_normalise(text).replace("'", "")))


def _words_in_order(text: str) -> list[str]:
    return _WORD.findall(_normalise(text).replace("'", ""))


def _drops_a_negation(words: list[str], source: str) -> bool:
    """True when the phrase negates where the source does not, or vice versa.

    Only applied to phrases that already pass the overlap test, so every word
    is present in the source and word order is the only thing left that can
    change the meaning.

    The comparison is made against the single best-matching span of the source
    rather than the span between the first and last matching word: a common
    word such as "the" matches early in a long ticket and would otherwise drag
    the window across unrelated clauses, flagging a correctly quoted phrase.
    """
    phrase_negated = any(w in _NEGATIONS for w in words)
    wanted = {w for w in words if w not in _NEGATIONS}
    if not wanted:
        return False
    source_tokens = _words_in_order(source)
    if not source_tokens:
        return False
    # Slide a window the length of the phrase and keep the position that covers
    # the most of its words; ties go to the earliest, which is deterministic.
    width = max(len(words), 1)
    best_lo, best_hit = 0, -1
    for lo in range(max(len(source_tokens) - width + 1, 1)):
        hit = sum(1 for w in source_tokens[lo : lo + width] if w in wanted)
        if hit > best_hit:
            best_lo, best_hit = lo, hit
    window = source_tokens[max(0, best_lo - 1) : best_lo + width + 1]
    source_negated = any(w in _NEGATIONS for w in window)
    return phrase_negated != source_negated
