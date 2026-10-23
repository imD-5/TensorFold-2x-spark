"""Find the RoCE v2 IPv4 GID index of an RDMA port (no torch needed).

The GID table of a CX7 port can differ from node to node, and it can change after a link flap. A fixed index
then points at the wrong entry, or at an empty one. The kernel shows the table in sysfs:
``<root>/<device>/ports/<port>/gids/<n>`` holds the GID and ``.../gid_attrs/types/<n>`` holds its type.
An entry is the one we want when its type is "RoCE v2" and its GID is an IPv4-mapped address.

The scan idea follows ``gidIndex`` in ``zig/src/cuda/roce.zig`` of the official TensorFold
(https://github.com/ashhart/TensorFold, Apache-2.0). The code here is written for this fork.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

SYSFS = Path("/sys/class/infiniband")
# The first 96 bits of an IPv4-mapped IPv6 address (::ffff:a.b.c.d), as the kernel prints a GID.
IPV4_MAPPED = "0000:0000:0000:0000:0000:ffff:"
# The variables that can fix the index. The first one that has a value decides. A value of -1 means "detect".
OVERRIDE_VARS = ("TF_RDMA_GID_INDEX", "NCCL_IB_GID_INDEX")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:                         # an empty table entry has no type file that can be read
        return None


def _entries(directory: Path) -> list[int]:
    try:
        return sorted(int(p.name) for p in directory.iterdir() if p.name.isdigit())
    except OSError:
        return []


def gid_entries(device: str, port: int = 1, *, root: Path | str = SYSFS) -> list[tuple[int, str, str]]:
    """Every readable entry of the GID table as ``(index, type, gid)``, in index order."""

    base = Path(root) / device / "ports" / str(port)
    out = []
    for n in _entries(base / "gids"):
        kind = _read(base / "gid_attrs" / "types" / str(n))
        gid = _read(base / "gids" / str(n))
        if kind is not None and gid is not None:
            out.append((n, kind, gid))
    return out


def detect_gid_index(device: str, port: int = 1, *, root: Path | str = SYSFS) -> int | None:
    """The lowest index of a RoCE v2 entry with an IPv4 address, or None when the table has none."""

    for n, kind, gid in gid_entries(device, port, root=root):
        if not kind.startswith("RoCE v2"):
            continue
        gid = gid.lower()
        if gid.startswith(IPV4_MAPPED) and gid[len(IPV4_MAPPED):] != "0000:0000":
            return n
    return None


def active_roce_devices(*, root: Path | str = SYSFS) -> list[str]:
    """The RDMA devices whose port 1 is ACTIVE with an Ethernet link layer (RoCE), in name order."""

    root = Path(root)
    out = []
    try:
        names = sorted(p.name for p in root.iterdir())
    except OSError:
        return out
    for d in names:
        state, layer = _read(root / d / "ports" / "1" / "state"), _read(root / d / "ports" / "1" / "link_layer")
        if state is not None and "ACTIVE" in state and layer == "Ethernet":
            out.append(d)
    return out


def nccl_hca_ports(env: Mapping[str, str] | None = None, *, root: Path | str = SYSFS) -> list[tuple[str, int]] | None:
    """The ``(device, port)`` pairs that NCCL_IB_HCA lets NCCL use, among the active RoCE devices.

    NCCL_IB_HCA is ``[^][=]<name>[:<port>],...``: a name matches the start of a device name, or the whole name after
    ``=``. Unset or empty: every active RoCE device (port 1). A list with ``^`` excludes devices: returns None, as the
    scan does not follow NCCL's rules for it. A pair without a port uses port 1.
    """

    env = os.environ if env is None else env
    raw = (env.get("NCCL_IB_HCA") or "").strip()
    active = active_roce_devices(root=root)
    if not raw:
        return [(d, 1) for d in active]
    if raw.startswith("^"):
        return None
    exact = raw.startswith("=")
    out = []
    for item in raw.lstrip("=").split(","):
        name, _, port = item.strip().partition(":")
        if not name:
            continue
        try:
            number = int(port) if port.strip() else 1
        except ValueError:
            return None
        for d in active:
            if (d == name if exact else d.startswith(name)) and (d, number) not in out:
                out.append((d, number))
    return out


def gid_override(env: Mapping[str, str] | None = None) -> int | None:
    """The index that the user fixed, or None when the user did not fix one (variable unset, empty or -1)."""

    env = os.environ if env is None else env
    for name in OVERRIDE_VARS:
        raw = (env.get(name) or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be a whole number (got {raw!r})") from None
        return value if value >= 0 else None
    return None


def resolve_gid_indices(devices: Sequence[str], port: int = 1, *, fixed: int | None = None,
                        env: Mapping[str, str] | None = None,
                        root: Path | str = SYSFS) -> list[tuple[int, str]]:
    """One ``(index, how)`` for each device. ``how`` is "fixed" (an argument or a variable) or "detected".

    ``fixed`` (an argument) wins over the variables. A negative ``fixed`` means "detect", and the variables are not read.
    Raises RuntimeError when a device has no RoCE v2 IPv4 entry and nothing fixes the index.
    """

    if fixed is None:
        fixed = gid_override(env)
    elif fixed < 0:
        fixed = None
    out = []
    for device in devices:
        if fixed is not None:
            out.append((fixed, "fixed"))
            continue
        found = detect_gid_index(device, port, root=root)
        if found is None:
            raise RuntimeError(f"{device} port {port} has no RoCE v2 IPv4 GID in {Path(root) / device}; "
                               f"set {OVERRIDE_VARS[0]} to the index to use")
        out.append((found, "detected"))
    return out
