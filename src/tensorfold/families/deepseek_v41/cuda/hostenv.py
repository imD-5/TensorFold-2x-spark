"""Host settings of the DeepSeek-V4.1 TP path that need no torch: the NCCL defaults and the page cache release.

NCCL cuMem off: on GB10 the driver can run out of memory when
NCCL allocates with the cuMem API under many streams, and later two-rank starts then fail until a reboot. The official
TensorFold turns these two NCCL settings off for the same reason (its pull request #995, Apache-2.0). The code here is
written for this fork.

Page cache release: the weight loader reads about 106 GB a rank through the page cache. On GB10 the CPU and the GPU
share one memory, so those pages take memory that the serving process needs. The official TensorFold drops the pages of
a loaded file with ``fadvise(DONTNEED)`` (its pull requests #766 and #916). ``drop_page_cache`` does the same.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from tensorfold.cuda.roce_gid import SYSFS, detect_gid_index, nccl_hca_ports

NCCL_DEFAULTS = {"NCCL_CUMEM_ENABLE": "0", "NCCL_CUMEM_HOST_ENABLE": "0"}
RELEASE_VAR = "TF_DS_RELEASE_PAGE_CACHE"


def _is_set(env: Mapping[str, str], name: str) -> bool:
    return bool((env.get(name) or "").strip())


def apply_nccl_defaults(*, env: dict | None = None, root: Path | str = SYSFS) -> dict[str, str]:
    """Set what the user did not set, before NCCL starts. Returns the variables that this call set.

    * ``NCCL_CUMEM_ENABLE`` and ``NCCL_CUMEM_HOST_ENABLE`` become 0.
    * ``NCCL_IB_GID_INDEX`` becomes the RoCE v2 IPv4 GID index of the ports that NCCL_IB_HCA lets NCCL use (every
      active RoCE port when it is unset), when each of those ports has the same index. With no such port, an
      exclusion list in NCCL_IB_HCA, or ports that differ, NCCL keeps its own choice.
      A value that is set (also -1, which is NCCL's own detection) stays as it is.
      When this call sets the index, it also sets ``TF_RDMA_GID_INDEX=-1`` (if that is unset) so that the RDMA
      gather keeps detecting each port itself.
    """

    env = os.environ if env is None else env
    done = {}
    for name, value in NCCL_DEFAULTS.items():
        if not _is_set(env, name):
            env[name] = done[name] = value
    ports = nccl_hca_ports(env, root=root) if not _is_set(env, "NCCL_IB_GID_INDEX") else None
    if ports:
        found = {detect_gid_index(d, p, root=root) for d, p in ports}
        if len(found) == 1 and None not in found:
            env["NCCL_IB_GID_INDEX"] = done["NCCL_IB_GID_INDEX"] = str(found.pop())
            # the RDMA gather reads NCCL_IB_GID_INDEX as the user's choice; this marker keeps its own detection on
            if not _is_set(env, "TF_RDMA_GID_INDEX"):
                env["TF_RDMA_GID_INDEX"] = "-1"
    return done


def release_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True unless TF_DS_RELEASE_PAGE_CACHE is 0 (or off, false, no)."""

    env = os.environ if env is None else env
    return (env.get(RELEASE_VAR) or "1").strip().lower() not in ("0", "off", "false", "no")


def drop_page_cache(target, *, env: Mapping[str, str] | None = None) -> bool:
    """Ask the kernel to drop the cached pages of a file: ``target`` is an open file descriptor or a path.

    The call is a hint: the kernel keeps dirty pages and pages that another process maps. It returns True when the
    kernel took the request, and False when the switch is off, the system has no ``posix_fadvise`` or the call failed.
    """

    advise = getattr(os, "posix_fadvise", None)
    flag = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or flag is None or not release_enabled(env):
        return False
    try:
        if isinstance(target, int):
            advise(target, 0, 0, flag)
        else:
            fd = os.open(target, os.O_RDONLY)
            try:
                advise(fd, 0, 0, flag)
            finally:
                os.close(fd)
    except OSError:
        return False
    return True


def meminfo_gib(field: str, meminfo: str = "/proc/meminfo") -> float | None:
    """One field of /proc/meminfo (for example "MemAvailable" or "Cached") in GiB, or None when it is not there."""

    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / (1 << 20)
    except (OSError, ValueError, IndexError):
        pass
    return None
