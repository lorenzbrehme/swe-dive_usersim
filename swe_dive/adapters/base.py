"""Adapter protocol: any benchmark → a directory of Harbor tasks.

Implementing an adapter for a new benchmark means defining a class that satisfies
this protocol. The SWE-Bench adapter (``swe_dive.adapters.swebench``) is the
reference implementation and PoC.
"""

from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class Adapter(Protocol):
    """Translate benchmark instances into Harbor task directories."""

    def iter_instances(self) -> Iterator[dict[str, Any]]:
        """Yield one dict per benchmark instance (fields adapter-defined)."""
        ...

    def emit_task(self, instance: dict[str, Any], out_dir: Path) -> Path:
        """Write a complete Harbor task directory for ``instance`` under ``out_dir``.

        Returns the path to the task directory that was created.
        """
        ...
