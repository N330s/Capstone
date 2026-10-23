from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import learning.vla.source_archive as archive


class SourceArchiveTests(unittest.TestCase):
    def test_explicit_writable_archive_outside_code_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'code'
            root.mkdir()
            (root/'trainer.py').write_text('version')
            target = Path(folder)/'runs/archives'
            with patch.object(archive, 'ROOT', root), patch.dict('os.environ', {'OPENARM_SOURCE_ARCHIVE': str(target)}):
                record = archive.archive_source('trainer.py')
                self.assertTrue(Path(record['archive']).is_relative_to(target))
                (root/'trainer.py').write_text('new')
                self.assertEqual(archive.verified_source('trainer.py', record['sha256']).read_text(), 'version')

    def test_historical_source_is_verified_after_live_file_changes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(archive, "ROOT", Path(folder)):
            source = Path(folder) / "trainer.py"
            source.write_text("old version")
            record = archive.archive_source("trainer.py")
            source.write_text("corrected version")
            saved = archive.verified_source("trainer.py", record["sha256"])
            self.assertEqual(saved.read_text(), "old version")
            saved.write_text("broken archive")
            with self.assertRaises(ValueError):
                archive.verified_source("trainer.py", record["sha256"])
