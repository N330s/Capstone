import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_smolvla_batches import choose_batch, summarize


class BatchBenchmarkTests(unittest.TestCase):
    def test_fastest_safe_not_largest(self):
        self.assertEqual(choose_batch([
            {"batch_size": 4, "safe": True, "samples_per_second": 10},
            {"batch_size": 8, "safe": True, "samples_per_second": 14},
            {"batch_size": 16, "safe": False, "samples_per_second": 20},
        ]), 8)
        self.assertEqual(choose_batch([
            {"batch_size": 8, "safe": True, "samples_per_second": 14},
            {"batch_size": 16, "safe": True, "samples_per_second": 13},
        ]), 8)
        with self.assertRaises(RuntimeError):
            choose_batch([{"batch_size": 16, "safe": False}])

    def test_reserved_peak_protects_between_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "report.json").write_text(json.dumps({"completed": True, "updates": 40,
                "peak_reserved_gb": 7., "peak_allocated_gb": 5.}))
            events = [{"type": "update", "step": step, "elapsed_s": step * .5,
                       "loss": .1, "grad_norm": .1} for step in range(1, 41)]
            (output / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
            result = summarize(output, 16, [{"used_mib": 4000}],
                               {"used_mib": 1500, "total_mib": 8188}, 0, False)
            self.assertFalse(result["safe"])
            self.assertEqual(result["samples_per_second"], 32)

    def test_failure_is_not_a_safe_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            result = summarize(Path(folder), 16, [], {}, 1, False)
            self.assertFalse(result["safe"])


if __name__ == "__main__":
    unittest.main()
