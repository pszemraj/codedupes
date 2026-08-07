"""Tests for the shared dependency-logger quieting helpers."""

import logging

from codedupes.logging_utils import (
    NOISY_EXTERNAL_LOGGERS,
    quiet_dependency_loggers,
    quiet_unconfigured_dependency_loggers,
)


def _restore_levels(levels: dict[str, int]) -> None:
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_quiet_dependency_loggers_pins_requested_level() -> None:
    prior = {name: logging.getLogger(name).level for name in NOISY_EXTERNAL_LOGGERS}
    try:
        quiet_dependency_loggers(logging.ERROR)
        for name in NOISY_EXTERNAL_LOGGERS:
            assert logging.getLogger(name).level == logging.ERROR
    finally:
        _restore_levels(prior)


def test_quiet_unconfigured_dependency_loggers_respects_explicit_levels() -> None:
    httpx_logger = logging.getLogger("httpx")
    prior = {name: logging.getLogger(name).level for name in NOISY_EXTERNAL_LOGGERS}
    try:
        for name in NOISY_EXTERNAL_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
        httpx_logger.setLevel(logging.INFO)

        quiet_unconfigured_dependency_loggers()

        # Explicitly configured logger untouched; unconfigured ones quieted.
        assert httpx_logger.level == logging.INFO
        for name in NOISY_EXTERNAL_LOGGERS:
            if name != "httpx":
                assert logging.getLogger(name).level == logging.WARNING
    finally:
        _restore_levels(prior)
