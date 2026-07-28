"""Turn delimited text into a list of string records."""

from __future__ import annotations


class LoadError(ValueError):
    """The input could not be read as a table."""


def split_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def split_cells(row: str, delimiter: str) -> list[str]:
    return [cell.strip() for cell in row.split(delimiter)]


def parse_table(text: str, delimiter: str = ",") -> list[dict[str, str]]:
    rows = split_rows(text)
    if not rows:
        return []
    header = split_cells(rows[0], delimiter)
    records: list[dict[str, str]] = []
    # Numbering from 2 so the message points at the line the user can see,
    # counting the header as line 1.
    for lineno, row in enumerate(rows[1:], start=2):
        cells = split_cells(row, delimiter)
        if len(cells) != len(header):
            raise LoadError(
                f"row {lineno}: expected {len(header)} fields, got {len(cells)}"
            )
        records.append(dict(zip(header, cells)))
    return records
