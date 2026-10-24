from __future__ import annotations

import json
import logging

from ticketintel.logging import configure_logging, get_logger


def test_configure_logging_emits_json(capsys) -> None:
    configure_logging("INFO")
    logger = get_logger("ticketintel.test")
    logger.info("hello", extra={"ticket_id": "T-1"})

    captured = capsys.readouterr().err.strip().splitlines()
    record = json.loads(captured[-1])
    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["ticket_id"] == "T-1"


def test_configure_logging_sets_level() -> None:
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
