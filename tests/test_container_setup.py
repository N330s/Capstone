from pathlib import Path
import unittest

from scripts.container_preflight import summarize


ROOT = Path(__file__).resolve().parents[1]


class ContainerSetupTests(unittest.TestCase):
    def test_all_selected_checks_must_pass_without_skips(self):
        self.assertTrue(summarize({"cuda": {"status": "passed"}})["passed"])
        skipped = summarize({"cuda": {"status": "skipped"}})
        self.assertFalse(skipped["passed"])
        self.assertTrue(skipped["selected_checks_passed"])
        failed = summarize({"cuda": {"status": "failed"}})
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["selected_checks_passed"])

    def test_build_context_starts_with_deny_all(self):
        lines = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
        self.assertEqual(lines[0], "**")
        for private in ("!data/", "!outputs/", "!test_collection_new_env/", "!third_party/", "!.git/", "!.env"):
            self.assertNotIn(private, lines)

    def test_compose_has_no_external_service_exposure(self):
        # Keep these tests usable in the lightweight base environment without YAML.
        text = (ROOT / "compose.yaml").read_text()
        self.assertNotIn("ports:", text)
        self.assertNotIn("privileged:", text)
        self.assertNotIn("docker.sock", text)
        self.assertIn("read_only: true", text)
        self.assertIn("cap_drop: [ALL]", text)
        self.assertIn("${OPENARM_STORAGE:?", text)
        self.assertIn('user: "${OPENARM_UID:-1000}:${OPENARM_GID:-1000}"', text)
        self.assertEqual(text.count("create_host_path: false"), 7)

    def test_cpu_video_decoder_does_not_replace_cuda_torch(self):
        text = (ROOT / "Dockerfile").read_text()
        self.assertIn("torch==2.10.0 torchvision==0.25.0", text)
        self.assertIn("https://download.pytorch.org/whl/cu128", text)
        self.assertIn("--no-deps torchcodec==0.10.0", text)
        self.assertIn("https://download.pytorch.org/whl/cpu", text)
        self.assertIn("USER 1000:1000", text)


if __name__ == "__main__":
    unittest.main()
