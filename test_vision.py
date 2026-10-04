import base64
import queue
import tempfile
import unittest
from pathlib import Path

import strata_grpc_backend as bridge


class FakeTokenizer:
    def encode(self, text, parse_special=True):
        if text == bridge.VISION_START:
            return [10]
        if text == bridge.IMAGE_PAD:
            return [11] if parse_special else [20, 21]
        raise AssertionError(text)


class FakeVision:
    def __init__(self, path, counts):
        self.path = path
        self.counts = counts

    def combine(self, sources):
        self.sources = list(sources)
        return self.path, self.counts


class FakeEngine:
    def __init__(self):
        self.lines = queue.Queue()
        self.lines.put("DONE")
        self.ended = False
        self.sent = []

    def _send(self, line):
        self.sent.append(line)

    def alive(self):
        return True


class VisionBridgeTests(unittest.TestCase):
    def test_raw_localai_base64_is_decoded(self):
        vision = bridge.Vision.__new__(bridge.Vision)
        vision.max_bytes = 100
        payload = b"\x89PNG\r\n\x1a\nexample"
        self.assertEqual(vision._load(base64.b64encode(payload).decode()), payload)

    def test_image_pad_is_expanded_per_encoder_count(self):
        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / "request.sve"
            combined.write_bytes(b"embeddings")
            backend = bridge.StrataBackend.__new__(bridge.StrataBackend)
            backend.vision_cfg = {"gpu": False}
            backend.vision = FakeVision(combined, [2, 3])
            backend.tok = FakeTokenizer()
            backend._ensure_loaded = lambda: None

            ids, path = backend._prepare_images(["first", "second"], [10, 11, 12, 10, 11, 13])

            self.assertEqual(ids, [10, 11, 11, 12, 10, 11, 11, 11, 13])
            self.assertEqual(path, combined)
            self.assertEqual(backend.vision.sources, ["first", "second"])

    def test_prompt_image_mismatch_fails_closed_and_removes_combined_file(self):
        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / "request.sve"
            combined.write_bytes(b"embeddings")
            backend = bridge.StrataBackend.__new__(bridge.StrataBackend)
            backend.vision_cfg = {"gpu": False}
            backend.vision = FakeVision(combined, [2])
            backend.tok = FakeTokenizer()
            backend._ensure_loaded = lambda: None

            with self.assertRaisesRegex(ValueError, "prompt and its images do not match"):
                backend._prepare_images(["image"], [99, 11])
            self.assertFalse(combined.exists())

    def test_geni_command_includes_embedding_file(self):
        backend = bridge.StrataBackend.__new__(bridge.StrataBackend)
        backend.engine = FakeEngine()
        backend._ensure_loaded = lambda: None

        result = list(backend._generate([1, 2, 3], 9, " temperature=0.2", Path("/tmp/request.sve")))

        self.assertEqual(result, [])
        self.assertEqual(backend.engine.sent, ["GENI 9 temperature=0.2 /tmp/request.sve 1,2,3"])


if __name__ == "__main__":
    unittest.main()
