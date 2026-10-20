"""Several GPU unit tools in one process slot (a candidate stacking several changes): each tools/dsv41 script runs as
its own python process with its own arguments; exits 1 when any of them does.

  python3 tools/dsv41/run_tests.py -- tool1.py [args] ++ tool2.py [args] ++ ... [--out F]
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    out = None
    if len(argv) >= 2 and argv[-2] == "--out":            # (lane slots append --out F)
        out, argv = argv[-1], argv[:-2]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    runs, cur = [], []
    for a in argv:
        if a == "++":
            runs.append(cur)
            cur = []
        else:
            cur.append(a)
    if cur:
        runs.append(cur)
    here = Path(__file__).parent
    res: dict = {"runs": [], "passed": True}
    for r in runs:
        t0 = time.time()
        sub_out = f"{out}.{Path(r[0]).stem}.json" if out else None
        cmd = [sys.executable, str(here / r[0]), *r[1:]] + (["--out", sub_out] if sub_out else [])
        print(f"== {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        res["runs"].append({"tool": r[0], "args": r[1:], "exit": rc, "seconds": round(time.time() - t0, 1)})
        res["passed"] = res["passed"] and rc == 0
    print(json.dumps(res, indent=1))
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
