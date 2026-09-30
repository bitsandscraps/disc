"""Minimal key-value logger writing TensorBoard scalars and ``progress.csv``.

The CSV layout matches baselines' logger so existing plotting scripts work.
"""

import csv
from pathlib import Path
from typing import IO, Any

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Collects key-value pairs and dumps them once per iteration."""

    def __init__(self, log_dir: str | Path | None) -> None:
        self.kvs: dict[str, Any] = {}
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._tb: SummaryWriter | None = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(  # pylint: disable=consider-using-with
                self.log_dir / "progress.csv", "w", newline="", encoding="utf-8"
            )
            self._tb = SummaryWriter(str(self.log_dir))

    def logkv(self, key: str, value: Any) -> None:
        """Set a value for the current iteration."""
        self.kvs[key] = value

    def dumpkvs(self, step: int) -> None:
        """Write the current iteration at ``step``, then clear it."""
        if self._tb is not None:
            for key, value in self.kvs.items():
                self._tb.add_scalar(key, value, step)
            self._tb.flush()
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
        """Close the CSV file and the TensorBoard writer."""
        if self._file is not None:
            self._file.close()
        if self._tb is not None:
            self._tb.close()
