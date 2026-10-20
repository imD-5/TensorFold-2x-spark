"""GB10 display-memory carveout: put large, rarely-read tensors in the ~2 GiB the firmware reserves for scanout.

On a DGX Spark (GB10) the firmware reserves ~2 GiB of the unified memory for the display. MemTotal never counts it,
so a headless node leaves it unused. With ``nvidia_drm modeset=1 fbdev=0`` a DRM dumb buffer can be allocated in it,
mapped into the process and registered with CUDA as I/O memory; on GB10's unified address space the device pointer
equals the host pointer, so torch can view it as an ordinary CUDA tensor.

It reads at about half the bandwidth of normal memory, so it only suits tensors that are large but touched a few rows
at a time: the token embedding table (a gather of the step's token rows) and the DSpark Markov embedding.

Enable with TF_DS_DISPLAY_MEM=1 (skip with a log line if the carveout is unavailable) or =require (fail instead).
The container needs ``--device /dev/dri/card0`` (TF_DS_DISPLAY_CARD picks another node).
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import struct

import torch

MiB = 2**20
DEFAULT_MIB = 1792          # 1.75 GiB: the size other GB10 recipes allocate from the ~2 GiB reservation
ALIGN = 64 * 1024

_REG_MAPPED, _REG_IOMEMORY = 0x02, 0x04


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("d") << 8) | nr


_CREATE_DUMB = _iowr(0xB2, 32)   # struct drm_mode_create_dumb {u32 height, width, bpp, flags, handle, pitch; u64 size}
_MAP_DUMB = _iowr(0xB3, 16)      # struct drm_mode_map_dumb {u32 handle, pad; u64 offset}


class Carveout:
    """One display buffer, mapped and registered with CUDA for the life of the process (never freed: CUDA graphs
    and kernels hold its addresses)."""

    def __init__(self, nbytes: int, card: str = "/dev/dri/card0"):
        if nbytes % 16384:
            raise ValueError("size must be a multiple of 16 KiB (4096 px x 32 bpp rows)")
        self.fd = os.open(card, os.O_RDWR | os.O_CLOEXEC)
        req = bytearray(struct.pack("IIIIIIQ", nbytes // 16384, 4096, 32, 0, 0, 0, 0))
        fcntl.ioctl(self.fd, _CREATE_DUMB, req, True)
        self.handle, size = struct.unpack("IIIIIIQ", req)[4], struct.unpack("IIIIIIQ", req)[6]
        if size != nbytes:
            raise RuntimeError(f"DRM dumb buffer is {size} bytes, asked for {nbytes}")
        req = bytearray(struct.pack("IIQ", self.handle, 0, 0))
        fcntl.ioctl(self.fd, _MAP_DUMB, req, True)
        self.mm = mmap.mmap(self.fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=struct.unpack("IIQ", req)[2])
        self.host = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        rt = ctypes.CDLL("libcudart.so")
        rc = rt.cudaHostRegister(ctypes.c_void_p(self.host), ctypes.c_size_t(nbytes), _REG_MAPPED | _REG_IOMEMORY)
        if rc:
            raise RuntimeError(f"cudaHostRegister(IoMemory) returned {rc}")
        dev = ctypes.c_void_p()
        rc = rt.cudaHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(self.host), 0)
        if rc:
            raise RuntimeError(f"cudaHostGetDevicePointer returned {rc}")
        self.ptr, self.nbytes, self.used = dev.value, nbytes, 0
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (self.ptr, False),
                                         "strides": None, "version": 3}
        self.bytes = torch.as_tensor(self, device="cuda")
        if self.bytes.data_ptr() != self.ptr:
            raise RuntimeError("torch copied the carveout instead of viewing it")

    def free(self) -> int:
        return self.nbytes - self.used

    def place(self, t: torch.Tensor) -> torch.Tensor | None:
        """A copy of ``t`` inside the carveout, or None if it doesn't fit."""

        n = t.numel() * t.element_size()
        if n > self.free():
            return None
        dst = self.bytes[self.used:self.used + n].view(t.dtype).view(t.shape)
        dst.copy_(t)
        self.used = (self.used + n + ALIGN - 1) // ALIGN * ALIGN
        return dst


_carveouts: list[Carveout] = []      # process lifetime


def offload(w, log=print) -> int:
    """Move the token and Markov embeddings of Weights ``w`` into the carveout. Returns the bytes moved."""

    mode = os.environ.get("TF_DS_DISPLAY_MEM", "")
    if mode in ("", "0"):
        return 0
    try:
        c = Carveout(int(os.environ.get("TF_DS_DISPLAY_MIB") or DEFAULT_MIB) * MiB,
                     os.environ.get("TF_DS_DISPLAY_CARD") or "/dev/dri/card0")
    except (OSError, RuntimeError, ValueError) as exc:
        if mode == "require":
            raise
        log(f"[tensorfold] rank {w.rank}: display memory unavailable ({exc}); tensors stay in normal memory")
        return 0
    _carveouts.append(c)
    moved = []
    targets = [("embed", w, "embed")]
    if getattr(w, "dspark", None) is not None:
        targets.append(("markov_embed", w.dspark, "markov_embed"))
    for label, owner, attr in targets:
        t = getattr(owner, attr)
        dst = c.place(t)
        if dst is None:
            log(f"[tensorfold] rank {w.rank}: display memory: {label} "
                f"({t.numel() * t.element_size() / MiB:.0f} MiB) does not fit in {c.free() / MiB:.0f} MiB")
            continue
        setattr(owner, attr, dst)
        moved.append(f"{label} {t.numel() * t.element_size() / MiB:.0f} MiB")
        del t
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log(f"[tensorfold] rank {w.rank}: display memory {c.nbytes / MiB:.0f} MiB at {c.ptr:#x}: "
        f"{', '.join(moved) or 'nothing moved'} ({c.used / MiB:.0f} MiB used)")
    return c.used
