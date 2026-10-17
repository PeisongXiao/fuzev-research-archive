import base64
import json
from pathlib import Path
import tempfile
import unittest

import prepare_sugarcrepe as panel


class PreparationTests(unittest.TestCase):
    def test_images_command_verifies_selected_local_images(self):
        from PIL import Image

        image_dir = self.root / "images"
        image_dir.mkdir()
        image_path = image_dir / "000000000001.jpg"
        Image.new("RGB", (3, 2), (12, 34, 56)).save(image_path)
        for role in panel.ROLES:
            (self.root / f"{role}.jsonl").write_text(json.dumps({
                "source_image_id": image_path.name,
                "image_url": "https://images.cocodataset.org/val2017/" + image_path.name}) + "\n")
        panel.main(["images", "--output", str(self.root)])
        with Image.open(image_path) as image:
            self.assertEqual(image.size, (3, 2))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def pairs(self, count=6):
        return [{"source_example_id": str(i), "hf_row": i,
                 "source_image_id": f"{i:012d}.jpg", "positive_caption": "A red square.",
                 "negative_caption": "A blue square.",
                 "changed_span": panel.changed_span("A red square.", "A blue square.")}
                for i in range(1, count + 1)]

    def test_official_captions_gold_and_exact_changed_span_survive(self):
        official = self.root / "official.json"
        examples = self.root / "examples.jsonl"
        source = {"x": {"filename": "000000000001.jpg", "caption": "A red square.",
                        "negative_caption": "A blue square."},
                  "y": {"filename": "000000000002.jpg", "caption": "A red square.",
                        "negative_caption": "A small blue circle."}}
        official.write_text(json.dumps(source))
        rows = [{"image_file_name": row["filename"], "original_file_name": "replace_att",
                 "positive_caption": [row["caption"]], "negative_caption": [row["negative_caption"]]}
                for row in source.values()]
        examples.write_text("\n".join(map(json.dumps, rows)))
        pairs, counts = panel.load_pairs(examples, official)
        self.assertEqual(counts, {"official_pairs": 2, "single_token_pairs": 1})
        self.assertEqual(pairs[0]["positive_caption"], source["x"]["caption"])
        for name in ("positive", "negative"):
            span = pairs[0]["changed_span"][name]
            text = pairs[0][f"{name}_caption"]
            self.assertEqual(text[span["start"]:span["end"]], span["text"])
        rows[0]["positive_caption"] = ["An altered gold caption."]
        examples.write_text("\n".join(map(json.dumps, rows)))
        with self.assertRaisesRegex(ValueError, "differs"):
            panel.load_pairs(examples, official)
        self.assertIsNone(panel.changed_span("A red square.", "a red square."))

    def test_seeded_roles_and_both_orders_preserve_one_image_pair(self):
        pairs = self.pairs()
        pairs.append({**pairs[0], "source_example_id": "alternate", "negative_caption": "A green square."})
        counts = {"fit": 1, "development": 2, "test": 2}
        selected, eligible = panel.select_pairs(pairs, {"6"}, counts, seed=21)
        repeated, _ = panel.select_pairs(pairs[::-1], {"6"}, counts, seed=21)
        self.assertEqual(selected, repeated)
        self.assertEqual(eligible, 5)
        rows = list(panel.records(selected, self.root))
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row["group"] for row in rows}), 5)
        for a, b in zip(rows[::2], rows[1::2]):
            self.assertEqual(a["group"], b["group"])
            self.assertEqual(a["role"], b["role"])
            self.assertEqual(a["candidates"], b["candidates"][::-1])
            self.assertEqual(a["target"], b["target"][::-1])
            for row in (a, b):
                winner = row["candidates"][row["target"].index(1.)]
                self.assertEqual(winner["text"], row["positive_caption"])
        self.assertEqual({role: sum(p["role"] == role for p in selected) for role in counts}, counts)
        with self.assertRaisesRegex(ValueError, "only 0 eligible"):
            panel.select_pairs(pairs, {str(i) for i in range(1, 7)}, counts, seed=21)

    def test_native_request_contains_inputs_without_gold_or_provenance(self):
        try:
            from fuzev.service_contracts import ServiceDecisionRequest
        except ModuleNotFoundError:
            self.skipTest("Install the existing runtime request dependencies for this contract test")
        pair = {**self.pairs(1)[0], "group": "coco:val2017:1", "role": "test", "positive_id": "1"}
        row = next(panel.records([pair], self.root))
        image_bytes = b"fixture image bytes"
        (self.root / row["image_path"]).write_bytes(image_bytes)
        request = panel.native_request(row, self.root)
        self.assertIsInstance(request, ServiceDecisionRequest)
        self.assertEqual([c.model_dump() for c in request.candidates], row["candidates"])
        self.assertEqual(request.messages, row["context"])
        self.assertEqual(base64.b64decode(request.image.split(",", 1)[1]), image_bytes)
        self.assertTrue({"target", "positive_id", "role", "changed_span", "source_example_id"}.isdisjoint(
            request.model_dump()))


if __name__ == "__main__":
    unittest.main()
