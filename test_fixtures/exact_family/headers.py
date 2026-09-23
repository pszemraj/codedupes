"""Header parsing that duplicates nothing else in the fixture."""


def parse_header(line):
    """Split a ``key: value`` header line.

    :param line: Raw header line.
    :return: ``(key, value)`` with surrounding whitespace removed.
    """
    key, separator, value = line.partition(":")
    if not separator:
        raise ValueError(f"not a header line: {line!r}")
    return key.strip(), value.strip()
