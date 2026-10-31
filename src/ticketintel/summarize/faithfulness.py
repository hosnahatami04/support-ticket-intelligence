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
