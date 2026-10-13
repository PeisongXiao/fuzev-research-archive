"""Tiny stdlib-only mask-cache checks; no Torch, model, or real corpus imports.

Run directly with: python3 -B tests/test_mask_descriptors.py
"""

from collections import OrderedDict
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import weakref


def load_features():
    # Load the real source in a private package; bypass acquisition dependencies
    # without replacing modules used by other tests in a full pytest run.
    root = Path(__file__).resolve().parents[1] / "fuzev_research"
    package = ModuleType("_mask_cache_fixture")
    package.__path__ = [str(root)]
    acquire = ModuleType("_mask_cache_fixture.acquire")
    acquire.write_json = lambda path, value: path.write_text(json.dumps(value))
    paths = ModuleType("_mask_cache_fixture.paths")

    def no_project_paths(path):
        raise AssertionError("No project paths")

    paths.project_path = no_project_paths
    spec = importlib.util.spec_from_file_location("_mask_cache_fixture.features", root / "features.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package.__name__: package, acquire.__name__: acquire,
                                  paths.__name__: paths}):
        spec.loader.exec_module(module)
    return module


features = load_features()


class Vector:
    def __init__(self, values):
        self.values = values

    def __lt__(self, other):
        return Vector([value < other for value in self.values])

    def any(self):
        return any(self.values)

    def nonzero(self):
        return [index for index, value in enumerate(self.values) if value]


class Masks:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return [list(row) for row in self.rows]

    def __getitem__(self, indices):
        return Masks([self.rows[index] for index in indices])

    def sum(self, dim):
        assert dim == 1
        return Vector([sum(row) for row in self.rows])

    def any(self, dim):
        assert dim == 0
        return Vector([any(column) for column in zip(*self.rows)])


class MaskDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fuzev-mask-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "identity.json").write_text("{}")
        (self.directory / "index.sqlite3").touch()
        self.masks, self.opens, self.references = {}, [], []
        self.closed_handles = 0
        self.read_error = None
        fake = ModuleType("safetensors")
        fake.safe_open = self.safe_open
        modules = patch.dict(sys.modules, {"safetensors": fake})
        modules.start()
        self.addCleanup(modules.stop)
        no_full_shards = patch.object(features.FeatureStore, "shard_tensors",
                                     side_effect=AssertionError("Mask lookup loaded a full shard"))
        no_full_shards.start()
        self.addCleanup(no_full_shards.stop)
        self.stores = []

    def store(self, budget=1 << 20):
        store = features.FeatureStore(self.directory, {}, cache_bytes=0,
                                      mask_cache_bytes=budget, read_only=True)
        self.stores.append(store)
        self.addCleanup(lambda: store.close() if store in self.stores else None)
        return store

    def shard(self, shard, masks):
        path = self.directory / f"shard-{shard:06d}.safetensors"
        path.write_bytes(b"fixture")
        self.masks[str(path)] = masks
        return path

    @staticmethod
    def rows(shard, *indices):
        return [{"shard": shard, "row_": index} for index in indices]

    def safe_open(self, path, framework):
        self.assertEqual(framework, "pt")
        self.opens.append(path)
        owner = self

        class Handle:
            def __enter__(self):
                return self

            def get_tensor(self, name):
                owner.assertEqual(name, "mask")
                if owner.read_error:
                    raise owner.read_error
                masks = Masks(owner.masks[path])
                owner.references.append(weakref.ref(masks))
                return masks

            def __exit__(self, *args):
                owner.closed_handles += 1

        return Handle()

    def test_immutable_hit_preserves_subset_width_order_duplicates_and_rng(self):
        store = self.store()
        self.shard(0, [[True, False, True] + [False] * 13, [True] * 5 + [False] * 11,
                       [False, True, False, True] + [False] * 12, [True] + [False] * 14 + [True]])
        rows = self.rows(0, 2, 0, 2)
        before, rng = [dict(row) for row in rows], random.getstate()
        self.assertEqual(store.mask_width(rows), 4)
        self.assertEqual(store.mask_width(self.rows(0, 0)), 3)
        self.assertEqual(store.mask_width(rows), 4)
        self.assertEqual(store.mask_width(self.rows(0, -1)), 16)
        self.assertEqual(rows, before)
        self.assertEqual(random.getstate(), rng)
        self.assertEqual(len(self.opens), 1)
        self.assertEqual(self.closed_handles, 1)
        self.assertTrue(all(reference() is None for reference in self.references))
        stamp, descriptor = store._mask_descriptors[0]
        self.assertIsInstance(stamp, tuple)
        self.assertEqual(descriptor, bytes([3, 5, 4, 16]))
        self.assertEqual(store._mask_descriptor_bytes, 4)
        self.assertEqual(store.loaded, OrderedDict())
        self.assertEqual(store.loaded_bytes, 0)

    def test_invalid_rows_only_fail_when_selected_including_cache_hits(self):
        store = self.store()
        self.shard(0, [[True, True, False], [False, False, True], [False] * 3])
        with self.assertRaisesRegex(ValueError, "at least two"):
            store.mask_width(self.rows(0, 1))
        self.assertEqual(store.mask_width(self.rows(0, 0)), 2)
        for index in (1, 2, 1):
            with self.assertRaisesRegex(ValueError, "at least two"):
                store.mask_width(self.rows(0, 0, index))
        with self.assertRaises(IndexError):
            store.mask_width(self.rows(0, 3))
        self.assertEqual(len(self.opens), 1)

    def test_loaded_shard_keeps_existing_selected_mask_validation(self):
        store = self.store()
        self.shard(0, [[True, True, False]])
        self.assertEqual(store.mask_width(self.rows(0, 0)), 2)
        store.loaded[0] = {"mask": Masks([[True, False, True], [False, False, True]])}
        with patch.object(features, "_shard_stamp", side_effect=AssertionError("Loaded shard stat")):
            self.assertEqual(store.mask_width(self.rows(0, 0)), 3)
            with self.assertRaisesRegex(ValueError, "at least two"):
                store.mask_width(self.rows(0, 1))
        self.assertEqual(len(self.opens), 1)

    def test_byte_budget_and_lru_refresh(self):
        store = self.store(budget=4)
        for shard in range(3):
            self.shard(shard, [[True, True], [True, True]])
        for shard in (0, 1, 0, 2):
            self.assertEqual(store.mask_width(self.rows(shard, 0)), 2)
        self.assertEqual(list(store._mask_descriptors), [0, 2])
        self.assertEqual(store._mask_descriptor_bytes, 4)
        self.assertEqual(len(self.opens), 3)
        store.mask_width(self.rows(1, 0))
        self.assertEqual(list(store._mask_descriptors), [2, 1])
        self.assertEqual(store._mask_descriptor_bytes, 4)
        self.assertEqual(len(self.opens), 4)
        self.shard(3, [[True, True]] * 5)
        store.mask_width(self.rows(3, 0))
        self.assertEqual(list(store._mask_descriptors), [2, 1])
        self.assertEqual(store._mask_descriptor_bytes, 4)

    def test_entry_limit_is_independent_of_byte_budget(self):
        store = self.store()
        with patch.object(features, "_MASK_CACHE_SHARDS", 2):
            for shard in range(3):
                self.shard(shard, [[True, True]])
                store.mask_width(self.rows(shard, 0))
        self.assertEqual(list(store._mask_descriptors), [1, 2])
        self.assertEqual(store._mask_descriptor_bytes, 2)

    def test_disabled_and_oversized_entries_do_not_cache(self):
        self.shard(0, [[True, True], [True, True]])
        for budget in (-1, 0, 1):
            with self.subTest(budget=budget):
                store = self.store(budget=budget)
                for _ in range(2):
                    self.assertEqual(store.mask_width(self.rows(0, 0)), 2)
                self.assertFalse(store._mask_descriptors)
                self.assertEqual(store._mask_descriptor_bytes, 0)
        self.assertEqual(len(self.opens), 6)

    def test_every_stamp_field_invalidates_a_descriptor(self):
        store = self.store()
        path = self.shard(0, [[True, True, False]])
        previous = (1, 2, 3, 4, 5)
        with patch.object(features, "_shard_stamp", return_value=previous):
            store.mask_width(self.rows(0, 0))
        for index in range(len(previous)):
            stamp = tuple(value + (field == index) for field, value in enumerate(previous))
            self.masks[str(path)] = [[True, False, True], [True, True, False]]
            with patch.object(features, "_shard_stamp", return_value=stamp):
                self.assertEqual(store.mask_width(self.rows(0, 0)), 3)
                self.assertEqual(store._mask_descriptors[0][0], stamp)
                self.assertEqual(store._mask_descriptor_bytes, 2)
            previous = stamp
        self.assertEqual(len(self.opens), 6)

    def test_same_size_and_mtime_atomic_replacement_is_not_a_hit(self):
        store = self.store()
        path = self.shard(0, [[True, True, False]])
        self.assertEqual(store.mask_width(self.rows(0, 0)), 2)
        before = path.stat()
        replacement = path.with_suffix(".tmp")
        replacement.write_bytes(b"fixture")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
        self.masks[str(path)] = [[True, False, True]]
        self.assertEqual(path.stat().st_size, before.st_size)
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(store.mask_width(self.rows(0, 0)), 3)
        self.assertEqual(len(self.opens), 2)

    def test_missing_shard_does_not_reuse_cached_descriptor(self):
        store = self.store()
        path = self.shard(0, [[True, True]])
        store.mask_width(self.rows(0, 0))
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            store.mask_width(self.rows(0, 0))
        self.assertEqual(len(self.opens), 1)

    def test_change_during_mask_read_fails_without_caching(self):
        store = self.store()
        self.shard(0, [[True, True]])
        with patch.object(features, "_shard_stamp", side_effect=[(1,), (2,)]):
            with self.assertRaisesRegex(ValueError, "changed during mask verification"):
                store.mask_width(self.rows(0, 0))
        self.assertFalse(store._mask_descriptors)
        self.assertEqual(store._mask_descriptor_bytes, 0)
        self.assertEqual(self.closed_handles, 1)

    def test_changed_corrupt_shard_does_not_fall_back_to_old_descriptor(self):
        store = self.store()
        self.shard(0, [[True, True]])
        store.mask_width(self.rows(0, 0))
        self.read_error = ValueError("fixture corrupt mask")
        with patch.object(features, "_shard_stamp", return_value=("changed",)):
            with self.assertRaisesRegex(ValueError, "fixture corrupt mask"):
                store.mask_width(self.rows(0, 0))
        self.assertFalse(store._mask_descriptors)
        self.assertEqual(store._mask_descriptor_bytes, 0)
        self.assertEqual(self.closed_handles, 2)

    def test_close_releases_descriptors(self):
        store = self.store()
        self.shard(0, [[True, True]])
        store.mask_width(self.rows(0, 0))
        store.close()
        self.stores.remove(store)
        self.assertFalse(store._mask_descriptors)
        self.assertEqual(store._mask_descriptor_bytes, 0)


if __name__ == "__main__":
    unittest.main()
