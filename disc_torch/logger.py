"""Minimal key-value logger writing stdout tables and ``progress.csv``.

The CSV layout matches baselines' logger so existing plotting scripts work.
"""

import csv
from pathlib import Path
from typing import IO, Any


class Logger:
    """Collects key-value pairs and dumps them once per iteration."""

    def __init__(self, log_dir: str | Path | None) -> None:
        self.kvs: dict[str, Any] = {}
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(  # pylint: disable=consider-using-with
                self.log_dir / "progress.csv", "w", newline="", encoding="utf-8"
            )

    def logkv(self, key: str, value: Any) -> None:
        """Set a value for the current iteration."""
        self.kvs[key] = value

    def dumpkvs(self) -> None:
        """Print and write the current iteration, then clear it."""
        width = max(len(k) for k in self.kvs)
        lines = [
            f"| {k:<{width}} | {format_value(v):>12} |"
            for k, v in self.kvs.items()
        ]
        border = "-" * len(lines[0])
        print("\n".join([border, *lines, border]), flush=True)
        if self._file is not None:
            if self._writer is None:
                self._writer = csv.DictWriter(
                    self._file, fieldnames=list(self.kvs)
                )
                self._writer.writeheader()
            self._writer.writerow(self.kvs)
            self._file.flush()
        self.kvs = {}

    def close(self) -> None:
        """Close the CSV file."""
        if self._file is not None:
            self._file.close()


def format_value(value: Any) -> str:
    """Compact string form of a logged value."""
    if isinstance(value, float):
        return f"{value:<12.3g}".strip()
    return str(value)
