"""TF_TRIM=1: malloc_trim(0) after each request, so freed blocks go back to the OS instead of an arena."""

from __future__ import annotations

import ctypes
import os

TRIM = os.environ.get("TF_TRIM", "0") == "1"
_LIBC = None


def trim() -> None:
    """Trim glibc's free blocks, all arenas, when TF_TRIM is on; a no-op elsewhere (Mac, a libc without it)."""

    global _LIBC
    if not TRIM:
        return
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            _LIBC = False
    if _LIBC:
        _LIBC(0)
