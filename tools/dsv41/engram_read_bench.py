"""How long a prompt chunk's Engram read takes: one layer's weight and scale rows for rows x cols random
row ids (cold: a fresh random set each repeat, the tables are ~100 GB a layer), through the reader pool the prompt
lane uses (engram_io.gather_rows2_at on pool 1) at a given thread count. The pools are sized at their first call, so a
process measures one thread count; run it once per count.

  python3 tools/dsv41/engram_read_bench.py --engram DIR [--layer 1] [--rows 2048] [--cols 12] [--threads 128]
      [--repeats 3] [--out F]                                               (CPU only; quiet drive)
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--engram", required=True)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--rows", type=int, default=2048)
    p.add_argument("--cols", type=int, default=12)
    p.add_argument("--threads", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--direct", action="store_true", help="O_DIRECT descriptors (gather_rows2_direct_at)")
    p.add_argument("--check", action="store_true", help="also compare the rows with the buffered preads'")
    p.add_argument("--out")
    a = p.parse_args()
    from torch.utils import cpp_extension

    src = Path(__file__).resolve().parents[2] / "src/tensorfold/families/deepseek_v41/cuda/engram_io.cpp"
    io = cpp_extension.load(name="tf_ds_engram_io_v7", sources=[str(src)], extra_cflags=["-O3"], verbose=False)
    files = {}
    for path in sorted(Path(a.engram).glob("*.safetensors")):
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
        fd = os.open(str(path), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        dfd = os.open(str(path), os.O_RDONLY | os.O_DIRECT)
        for name, e in header.items():
            if name.endswith("engram.embed.weight") or name.endswith("engram.embed.scale"):
                lo, _ = e["data_offsets"]
                files[name] = (dfd if a.direct else fd, 8 + size + lo, int(np.prod(e["shape"][1:])),
                               int(e["shape"][0]), fd)
    fw, bw, rw, nrows, fw_b = files[f"layers.{a.layer}.engram.embed.weight"]
    fs, bs, rs, _, fs_b = files[f"layers.{a.layer}.engram.embed.scale"]
    read = io.gather_rows2_direct_at if a.direct else io.gather_rows2_at
    m = a.rows * a.cols
    out_w = torch.empty((m, rw), dtype=torch.uint8)
    out_s = torch.empty((m, rs), dtype=torch.uint8)
    rng = np.random.default_rng(int(time.time()))
    res: dict = {"layer": a.layer, "rows": a.rows, "cols": a.cols, "reads": 2 * m, "row_bytes": [rw, rs],
                 "table_rows": nrows, "threads": a.threads, "direct": a.direct, "ms": []}
    for _ in range(a.repeats):
        idx = torch.from_numpy(rng.integers(0, nrows, size=m, dtype=np.int64))
        t0 = time.perf_counter()
        read(1, fw, bw, rw, fs, bs, rs, idx, out_w, out_s, a.threads)
        res["ms"].append(round(1000 * (time.perf_counter() - t0), 1))
        if a.check:
            ref_w, ref_s = torch.empty_like(out_w), torch.empty_like(out_s)
            io.gather_rows2_at(0, fw_b, bw, rw, fs_b, bs, rs, idx, ref_w, ref_s, a.threads)
            res.setdefault("same_bytes", []).append(bool(torch.equal(ref_w, out_w) and torch.equal(ref_s, out_s)))
    res["reads_per_s"] = round(2 * m / (min(res["ms"]) / 1000))
    res["passed"] = all(res.get("same_bytes", [True]))
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
