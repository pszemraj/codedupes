"""Logging helpers shared by the CLI and the Python API."""

import logging

NOISY_EXTERNAL_LOGGERS = (
    "httpx",
    "huggingface_hub",
    "jax",
    "numexpr",
    "sentence_transformers",
    "tensorflow",
    "torch.utils.cpp_extension",
    "transformers",
    "urllib3",
)


def quiet_dependency_loggers(level: int = logging.WARNING) -> None:
    """Pin known-noisy third-party loggers to ``level``.

    Call after configuring application logging (for example
    ``logging.basicConfig(level=logging.INFO)``) to keep model-download and
    inference chatter — httpx request lines, transformers/sentence-transformers
    loading noise — out of INFO-level output. The CLI applies this automatically;
    Python API callers opt in explicitly.

    :param level: Level assigned to each noisy dependency logger.
    """
    for logger_name in NOISY_EXTERNAL_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


def quiet_unconfigured_dependency_loggers() -> None:
    """Quiet noisy dependency loggers the application has not explicitly configured.

    Applied before model loading so API callers with a bare root-level INFO config
    are not spammed by dependency chatter they never asked for. Only loggers still
    at ``NOTSET`` (inheriting the root level) are touched; any logger whose level
    was explicitly assigned — including the CLI's verbose DEBUG mode — is left
    alone.
    """
    for logger_name in NOISY_EXTERNAL_LOGGERS:
        dep_logger = logging.getLogger(logger_name)
        if dep_logger.level == logging.NOTSET:
            dep_logger.setLevel(logging.WARNING)
