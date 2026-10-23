"""DeepSeek-V4.1 host settings: the RoCE GID scan, the NCCL defaults and the page cache release (CPU only).

Run with ``python -m pytest tests/test_dsv41_host_env.py`` or ``python -m unittest``. The weight loader tests need
PyTorch and skip without it.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tensorfold.cuda import roce_gid
from tensorfold.families.deepseek_v41.cuda import hostenv

V4 = "0000:0000:0000:0000:0000:ffff:c0a8:0a01"       # 192.168.10.1
V4_B = "0000:0000:0000:0000:0000:ffff:c0a8:0a02"
LINK_LOCAL = "fe80:0000:0000:0000:aaaa:bbff:fecc:ddee"
EMPTY = "0000:0000:0000:0000:0000:0000:0000:0000"
FADV_DONTNEED = 4                                    # the Linux value; Windows has no os.POSIX_FADV_DONTNEED


def make_port(root: Path, device: str, entries: dict, port: int = 1, *, state: str = "4: ACTIVE",
              layer: str = "Ethernet") -> None:
    """A fake sysfs port: ``entries`` maps an index to (type, gid); a type of None leaves the type file out."""

    base = root / device / "ports" / str(port)
    (base / "gids").mkdir(parents=True)
    (base / "state").write_text(state + "\n")
    (base / "link_layer").write_text(layer + "\n")
    (base / "gid_attrs" / "types").mkdir(parents=True)
    for n, (kind, gid) in entries.items():
        (base / "gids" / str(n)).write_text(gid + "\n")
        if kind is not None:
            (base / "gid_attrs" / "types" / str(n)).write_text(kind + "\n")


class GidScanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_takes_the_roce_v2_ipv4_entry(self):
        make_port(self.root, "rocep1s0f0", {
            0: ("IB/RoCE v1", LINK_LOCAL),
            1: ("RoCE v2", LINK_LOCAL),
            2: ("IB/RoCE v1", V4),
            3: ("RoCE v2", V4),
        })
        self.assertEqual(roce_gid.detect_gid_index("rocep1s0f0", root=self.root), 3)

    def test_entries_sort_by_number_not_by_text(self):
        make_port(self.root, "d", {2: ("RoCE v2", LINK_LOCAL), 10: ("RoCE v2", V4), 11: ("RoCE v2", V4_B)})
        self.assertEqual(roce_gid.detect_gid_index("d", root=self.root), 10)
        self.assertEqual([n for n, _, _ in roce_gid.gid_entries("d", root=self.root)], [2, 10, 11])

    def test_entry_without_a_type_file_is_skipped(self):
        # an empty slot of the table: the gid file reads as zeros and the type file cannot be read
        make_port(self.root, "d", {0: (None, EMPTY), 1: (None, V4), 2: ("RoCE v2", V4)})
        self.assertEqual(roce_gid.detect_gid_index("d", root=self.root), 2)

    def test_zero_ipv4_address_is_skipped(self):
        make_port(self.root, "d", {0: ("RoCE v2", "0000:0000:0000:0000:0000:ffff:0000:0000"), 1: ("RoCE v2", V4)})
        self.assertEqual(roce_gid.detect_gid_index("d", root=self.root), 1)

    def test_no_match_gives_none(self):
        make_port(self.root, "d", {0: ("IB/RoCE v1", V4), 1: ("RoCE v2", LINK_LOCAL)})
        self.assertIsNone(roce_gid.detect_gid_index("d", root=self.root))
        self.assertIsNone(roce_gid.detect_gid_index("missing", root=self.root))

    def test_port_number_is_read(self):
        make_port(self.root, "d", {0: ("RoCE v2", V4)}, port=2)
        self.assertIsNone(roce_gid.detect_gid_index("d", 1, root=self.root))
        self.assertEqual(roce_gid.detect_gid_index("d", 2, root=self.root), 0)

    def test_each_device_has_its_own_index(self):
        make_port(self.root, "a", {5: ("RoCE v2", V4)})
        make_port(self.root, "b", {3: ("RoCE v2", V4_B)})
        got = roce_gid.resolve_gid_indices(["a", "b"], env={}, root=self.root)
        self.assertEqual(got, [(5, "detected"), (3, "detected")])

    def test_unset_empty_and_minus_one_all_mean_detect(self):
        make_port(self.root, "a", {4: ("RoCE v2", V4)})
        for env in ({}, {"NCCL_IB_GID_INDEX": ""}, {"NCCL_IB_GID_INDEX": "-1"}, {"TF_RDMA_GID_INDEX": " -1 "}):
            with self.subTest(env=env):
                self.assertEqual(roce_gid.resolve_gid_indices(["a"], env=env, root=self.root), [(4, "detected")])

    def test_a_fixed_index_wins_over_the_table(self):
        make_port(self.root, "a", {4: ("RoCE v2", V4)})
        self.assertEqual(roce_gid.resolve_gid_indices(["a"], env={"NCCL_IB_GID_INDEX": "7"}, root=self.root),
                         [(7, "fixed")])
        self.assertEqual(roce_gid.resolve_gid_indices(["a"], fixed=2, env={}, root=self.root), [(2, "fixed")])
        # the argument wins over the variables; a negative argument means detect
        self.assertEqual(roce_gid.resolve_gid_indices(["a"], fixed=2, env={"TF_RDMA_GID_INDEX": "9"},
                                                      root=self.root), [(2, "fixed")])
        self.assertEqual(roce_gid.resolve_gid_indices(["a"], fixed=-1, env={}, root=self.root), [(4, "detected")])

    def test_a_negative_argument_bypasses_the_variables(self):
        make_port(self.root, "a", {4: ("RoCE v2", V4)})
        for env in ({"NCCL_IB_GID_INDEX": "9"}, {"TF_RDMA_GID_INDEX": "9"}):
            with self.subTest(env=env):
                self.assertEqual(roce_gid.resolve_gid_indices(["a"], fixed=-1, env=env, root=self.root),
                                 [(4, "detected")])

    def test_tf_variable_decides_before_the_nccl_variable(self):
        env = {"TF_RDMA_GID_INDEX": "6", "NCCL_IB_GID_INDEX": "3"}
        self.assertEqual(roce_gid.gid_override(env), 6)
        # TF_RDMA_GID_INDEX=-1 asks for detection even when NCCL_IB_GID_INDEX holds an index
        self.assertIsNone(roce_gid.gid_override({"TF_RDMA_GID_INDEX": "-1", "NCCL_IB_GID_INDEX": "3"}))

    def test_bad_variable_is_named(self):
        with self.assertRaisesRegex(ValueError, "NCCL_IB_GID_INDEX"):
            roce_gid.gid_override({"NCCL_IB_GID_INDEX": "five"})

    def test_missing_entry_is_an_error_that_names_the_device(self):
        make_port(self.root, "a", {0: ("IB/RoCE v1", V4)})
        with self.assertRaisesRegex(RuntimeError, "a port 1 has no RoCE v2 IPv4 GID"):
            roce_gid.resolve_gid_indices(["a"], env={}, root=self.root)


class DeviceListTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        make_port(self.root, "rocep1s0f0", {}, state="1: DOWN")
        make_port(self.root, "rocep1s0f1", {})
        make_port(self.root, "roceP2p1s0f1", {})
        make_port(self.root, "mlx5_ib", {}, layer="InfiniBand")

    def test_active_roce_devices_are_the_active_ethernet_ports(self):
        self.assertEqual(roce_gid.active_roce_devices(root=self.root), ["roceP2p1s0f1", "rocep1s0f1"])
        self.assertEqual(roce_gid.active_roce_devices(root=self.root / "none"), [])

    def test_nccl_hca_ports(self):
        both = [("roceP2p1s0f1", 1), ("rocep1s0f1", 1)]
        cases = [
            ({}, both),
            ({"NCCL_IB_HCA": ""}, both),
            ({"NCCL_IB_HCA": "rocep1s0f1"}, [("rocep1s0f1", 1)]),
            ({"NCCL_IB_HCA": "=rocep1s0f1,roceP2p1s0f1"}, [("rocep1s0f1", 1), ("roceP2p1s0f1", 1)]),
            ({"NCCL_IB_HCA": "rocep1s0f"}, [("rocep1s0f1", 1)]),                  # a prefix; f0 is down
            ({"NCCL_IB_HCA": "=rocep1s0f"}, []),                                   # "=" is an exact match
            ({"NCCL_IB_HCA": "rocep1s0f1:1"}, [("rocep1s0f1", 1)]),
            ({"NCCL_IB_HCA": "rocep1s0f1:2"}, [("rocep1s0f1", 2)]),
            ({"NCCL_IB_HCA": "mlx5"}, []),                                         # not RoCE
            ({"NCCL_IB_HCA": "^rocep1s0f1"}, None),                                # an exclusion list is not followed
            ({"NCCL_IB_HCA": "rocep1s0f1:x"}, None),
        ]
        for env, want in cases:
            with self.subTest(env=env):
                self.assertEqual(roce_gid.nccl_hca_ports(env, root=self.root), want)


class NcclDefaultsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        make_port(self.root, "a", {3: ("RoCE v2", V4)})
        make_port(self.root, "b", {3: ("RoCE v2", V4_B)})
        self.nothing = self.root / "none"                      # a sysfs root with no device: the cumem tests only

    def apply(self, env: dict) -> dict:
        return hostenv.apply_nccl_defaults(env=env, root=self.root)

    def test_cumem_is_off_by_default(self):
        env: dict = {}
        done = hostenv.apply_nccl_defaults(env=env, root=self.nothing)
        self.assertEqual(env, {"NCCL_CUMEM_ENABLE": "0", "NCCL_CUMEM_HOST_ENABLE": "0"})
        self.assertEqual(done, env)

    def test_a_value_that_the_user_set_stays(self):
        env = {"NCCL_CUMEM_ENABLE": "1", "NCCL_CUMEM_HOST_ENABLE": "1"}
        self.assertEqual(hostenv.apply_nccl_defaults(env=env, root=self.nothing), {})
        self.assertEqual(env, {"NCCL_CUMEM_ENABLE": "1", "NCCL_CUMEM_HOST_ENABLE": "1"})

    def test_one_set_and_one_unset(self):
        env = {"NCCL_CUMEM_HOST_ENABLE": "1"}
        hostenv.apply_nccl_defaults(env=env, root=self.nothing)
        self.assertEqual(env, {"NCCL_CUMEM_ENABLE": "0", "NCCL_CUMEM_HOST_ENABLE": "1"})

    def test_an_empty_value_counts_as_unset(self):
        env = {"NCCL_CUMEM_ENABLE": " "}
        hostenv.apply_nccl_defaults(env=env, root=self.nothing)
        self.assertEqual(env["NCCL_CUMEM_ENABLE"], "0")

    def test_the_process_environment_is_the_default_target(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            hostenv.apply_nccl_defaults(root=self.nothing)
            self.assertEqual(os.environ["NCCL_CUMEM_ENABLE"], "0")
            self.assertEqual(os.environ["NCCL_CUMEM_HOST_ENABLE"], "0")

    def test_gid_index_is_set_from_the_table(self):
        env: dict = {}
        done = self.apply(env)
        self.assertEqual(env["NCCL_IB_GID_INDEX"], "3")
        self.assertEqual(done["NCCL_IB_GID_INDEX"], "3")
        # the gather keeps its own per-port detection
        self.assertEqual(env["TF_RDMA_GID_INDEX"], "-1")
        self.assertEqual(roce_gid.resolve_gid_indices(["a", "b"], env=env, root=self.root),
                         [(3, "detected"), (3, "detected")])

    def test_gid_index_that_the_user_set_stays(self):
        for value in ("5", "-1"):
            with self.subTest(value=value):
                env = {"NCCL_IB_GID_INDEX": value}
                self.apply(env)
                self.assertEqual(env["NCCL_IB_GID_INDEX"], value)
                self.assertNotIn("TF_RDMA_GID_INDEX", env)

    def test_user_tf_variable_stays(self):
        env = {"TF_RDMA_GID_INDEX": "8"}
        self.apply(env)
        self.assertEqual(env["TF_RDMA_GID_INDEX"], "8")

    def test_gid_index_comes_from_the_hcas_that_nccl_uses(self):
        make_port(self.root, "c", {6: ("RoCE v2", V4)})
        for hca, want in (("=c", "6"), ("c", "6"), ("=a", "3"), ("=a,b", "3")):
            with self.subTest(hca=hca):
                env = {"NCCL_IB_HCA": hca}
                self.apply(env)
                self.assertEqual(env["NCCL_IB_GID_INDEX"], want)

    def test_gid_index_stays_unset_when_the_ports_differ_or_cannot_be_named(self):
        make_port(self.root, "c", {6: ("RoCE v2", V4)})
        make_port(self.root, "d", {0: ("IB/RoCE v1", V4)})
        for hca in ("=a,c", "a,c", "=d", "^a", "=none", "a:x"):
            with self.subTest(hca=hca):
                env = {"NCCL_IB_HCA": hca}
                self.apply(env)
                self.assertNotIn("NCCL_IB_GID_INDEX", env)
                self.assertNotIn("TF_RDMA_GID_INDEX", env)

    def test_all_active_ports_count_when_nccl_ib_hca_is_unset(self):
        make_port(self.root, "c", {6: ("RoCE v2", V4)})
        env: dict = {}
        self.apply(env)                                       # a, b and c are active: 3, 3 and 6
        self.assertNotIn("NCCL_IB_GID_INDEX", env)

    def test_no_ports_leave_the_gid_index_alone(self):
        env: dict = {}
        hostenv.apply_nccl_defaults(env=env, root=self.root / "none")
        self.assertNotIn("NCCL_IB_GID_INDEX", env)


class PageCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "weights.bin"
        self.path.write_bytes(os.urandom(1 << 16))
        self.on = mock.patch.dict(os.environ, {}, clear=False)
        self.on.start()
        self.addCleanup(self.on.stop)
        os.environ.pop(hostenv.RELEASE_VAR, None)

    def patched(self, **kwargs):
        advise = mock.Mock(return_value=None, **kwargs)
        return advise, (mock.patch.object(os, "posix_fadvise", advise, create=True),
                        mock.patch.object(os, "POSIX_FADV_DONTNEED", FADV_DONTNEED, create=True))

    def test_switch_default_and_values(self):
        self.assertTrue(hostenv.release_enabled({}))
        self.assertTrue(hostenv.release_enabled({hostenv.RELEASE_VAR: "1"}))
        for off in ("0", "off", "False", " no "):
            self.assertFalse(hostenv.release_enabled({hostenv.RELEASE_VAR: off}))

    def test_a_file_descriptor_gets_dontneed_on_the_whole_file(self):
        advise, (a, b) = self.patched()
        with a, b, open(self.path, "rb") as f:
            self.assertTrue(hostenv.drop_page_cache(f.fileno()))
            advise.assert_called_once_with(f.fileno(), 0, 0, FADV_DONTNEED)

    def test_a_path_is_opened_advised_and_closed(self):
        seen = []
        advise, (a, b) = self.patched()
        advise.side_effect = lambda fd, off, n, flag: seen.append(os.fstat(fd).st_size)
        with a, b:
            self.assertTrue(hostenv.drop_page_cache(self.path))
            self.assertTrue(hostenv.drop_page_cache(str(self.path)))
        self.assertEqual(seen, [1 << 16, 1 << 16])
        self.assertEqual(advise.call_count, 2)

    def test_the_switch_turns_it_off(self):
        advise, (a, b) = self.patched()
        with a, b, mock.patch.dict(os.environ, {hostenv.RELEASE_VAR: "0"}), open(self.path, "rb") as f:
            self.assertFalse(hostenv.drop_page_cache(f.fileno()))
        advise.assert_not_called()

    def test_a_refused_call_gives_false(self):
        advise, (a, b) = self.patched()
        advise.side_effect = OSError(22, "Invalid argument")
        with a, b, open(self.path, "rb") as f:
            self.assertFalse(hostenv.drop_page_cache(f.fileno()))

    def test_a_missing_path_gives_false(self):
        advise, (a, b) = self.patched()
        with a, b:
            self.assertFalse(hostenv.drop_page_cache(self.path.with_name("none.bin")))
        advise.assert_not_called()

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "this system has no posix_fadvise")
    def test_the_real_call_works_on_a_temp_file(self):
        self.assertTrue(hostenv.drop_page_cache(self.path))
        with open(self.path, "rb") as f:
            self.assertTrue(hostenv.drop_page_cache(f.fileno()))
            self.assertEqual(len(f.read()), 1 << 16)           # the file reads the same bytes after the drop

    def test_meminfo_reads_one_field(self):
        f = Path(self._tmp.name) / "meminfo"
        f.write_text("MemTotal:  131072000 kB\nMemAvailable:   2097152 kB\nCached:   1048576 kB\nSwapCached: 5 kB\n")
        self.assertAlmostEqual(hostenv.meminfo_gib("MemAvailable", str(f)), 2.0)
        self.assertAlmostEqual(hostenv.meminfo_gib("Cached", str(f)), 1.0)       # not SwapCached
        self.assertIsNone(hostenv.meminfo_gib("Missing", str(f)))
        self.assertIsNone(hostenv.meminfo_gib("Cached", str(f.with_name("none"))))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "needs PyTorch (the CUDA backend's code)")
class LoaderReleaseTests(unittest.TestCase):
    """The weight loader gives its files' pages back when it closes them."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        from tensorfold.families.deepseek_v41.cuda import weights

        self.weights = weights

    def test_shards_close_drops_every_open_file(self):
        import json
        import struct

        for name, tensor in (("a.safetensors", "x"), ("b.safetensors", "y")):
            header = json.dumps({tensor: {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
            (self.dir / name).write_bytes(struct.pack("<Q", len(header)) + header + b"\x01\x02\x03\x04")
        sh = self.weights.Shards(self.dir)
        sh.get("x")
        sh.get("y")
        with mock.patch.object(self.weights, "drop_page_cache") as drop:
            sh.close()
        self.assertEqual(drop.call_count, 2)
        self.assertTrue(all(isinstance(c.args[0], int) for c in drop.call_args_list))

    def test_rank_cache_drops_its_file_when_it_writes_and_when_it_reads(self):
        import torch

        model = self.dir / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "w.safetensors").write_bytes(b"x")
        cache_dir = self.dir / "cache"
        with mock.patch.object(self.weights, "drop_page_cache") as drop:
            writer = self.weights.RankCache.open(str(cache_dir), model, 0, 2, "t")
            self.assertFalse(writer.reading)
            writer.put("k", torch.arange(8, dtype=torch.int16))
            self.assertTrue(writer.close(ok=True))
            self.assertEqual(drop.call_count, 1)
            reader = self.weights.RankCache.open(str(cache_dir), model, 0, 2, "t")
            self.assertTrue(reader.reading)
            self.assertEqual(reader.take("k").tolist(), list(range(8)))
            self.assertTrue(reader.close(ok=True))
            self.assertEqual(drop.call_count, 2)


if __name__ == "__main__":
    unittest.main()
