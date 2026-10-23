import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from learning.vla.lab_training import load_config, run_folder, train_command, qualification_command, throughput, verify_inputs
from learning.vla.runtime_resources import existing_parent, resources


class LabTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {key: self.root/key for key in
            ('export', 'collection', 'base', 'checkpoint', 'backbone', 'snapshot', 'runs')}
        self.cfg.update(batch_size=8, num_workers=0, additional_steps=6195,
            wall_seconds=14400, save_freq=2000, seed=1000)
        for key in ('export', 'collection', 'base', 'checkpoint', 'backbone', 'snapshot', 'runs'):
            self.cfg[key].mkdir()
        (self.cfg['export']/'openarm_export.json').write_text(json.dumps(
            {'datasets': {'train': {'repo_id': 'local/openarm'}}}))

    def config_file(self, cfg):
        path = self.root/'config.json'
        path.write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in cfg.items()}))
        return path

    def test_portable_absolute_paths_and_bounded_configuration(self):
        self.assertEqual(load_config(self.config_file(self.cfg)), self.cfg)
        for key,value in (('export', 'relative'), ('wall_seconds', 14401),
                          ('batch_size', True), ('num_workers', -1), ('additional_steps', 0)):
            cfg = {**self.cfg, key: value}
            with self.assertRaises(ValueError, msg=key):
                load_config(self.config_file(cfg))

    def test_refuses_existing_run_path_traversal_or_input_overlap(self):
        self.assertEqual(run_folder(self.cfg, 'new_001'), self.cfg['runs']/'new_001')
        for name in (None, '', '../outside', 'a/b', 'a b', 'a.b'):
            with self.assertRaises(ValueError):
                run_folder(self.cfg, name)
        (self.cfg['runs']/'existing').mkdir()
        with self.assertRaises(FileExistsError):
            run_folder(self.cfg, 'existing')
        with self.assertRaises(ValueError):
            run_folder({**self.cfg, 'runs': self.cfg['base']}, 'inside_model')

    def test_commands_use_native_paths_full_split_and_no_upload_or_heuristics(self):
        command = train_command(self.cfg, self.cfg['runs']/'new/training')
        self.assertIn('--native-export', command)
        self.assertIn(str(self.cfg['export']), command)
        self.assertIn(f'--policy.path={self.cfg["checkpoint"]}', command)
        self.assertIn('--steps=6195', command)
        self.assertIn('--batch_size=8', command)
        self.assertIn('--native-validation', command)
        self.assertIn('--policy.push_to_hub=false', command)
        self.assertIn('--wandb.enable=false', command)
        self.assertNotIn('--resume=true', command)
        self.assertFalse(any('episodes=' in a or 'recovery' in a or 'accumulation' in a for a in command))
        smoke = train_command(self.cfg, self.cfg['runs']/'smoke/training', steps=64, smoke=True)
        self.assertIn('--steps=64', smoke)
        self.assertIn('--save_freq=64', smoke)
        self.assertNotIn('--native-validation', smoke)
        reload = qualification_command(self.cfg, self.cfg['runs']/'reload', checkpoint=Path('/new/model'))
        self.assertIn('reload', reload)
        self.assertIn('--backbone', reload)
        self.assertIn(str(self.cfg['backbone']), reload)

    def test_warmed_throughput_uses_elapsed_loader_time_not_compute_metric(self):
        events = [{'type':'update', 'step':i, 'elapsed_s':2*i, 'update_seconds':.1} for i in range(1,21)]
        result = throughput(events, 8)
        self.assertEqual(result['windows_per_second'], 4)
        self.assertEqual(result['warm_updates'], 12)
        with self.assertRaises(ValueError):
            throughput(events[:8], 8)

    def test_frozen_config_hashes_match_reviewed_release(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/lab_smolvla.json').read_text())
        for key in ('export_sha256','stats_sha256','base_sha256','checkpoint_sha256'):
            self.assertEqual(len(cfg[key]),64)
            int(cfg[key],16)
        self.assertEqual(cfg['checkpoint'], '/models/full81_step012243')

    def test_input_binding_mismatch_refused_before_model_import(self):
        cfg = {**self.cfg, 'export_sha256': '0'*64, 'stats_sha256': '0'*64,
               'base_sha256':'0'*64, 'checkpoint_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_inputs(cfg)

    def test_disk_monitor_uses_output_mount_existing_parent(self):
        path = self.root/'new/run'
        self.assertEqual(existing_parent(path), self.root)
        with patch('learning.vla.runtime_resources.subprocess.check_output', return_value='60, 4096\n'), \
             patch('learning.vla.runtime_resources.shutil.disk_usage') as disk:
            disk.return_value.free = 100*2**30
            values = resources(path)
            disk.assert_called_once_with(self.root)
            self.assertEqual(values['disk_free_bytes'], 100*2**30)

    def test_ambiguous_or_low_resource_monitor_refused(self):
        for value in ('60, 4096\n60, 4096\n', '85, 4096\n', '60, 128\n'):
            with patch('learning.vla.runtime_resources.subprocess.check_output', return_value=value):
                with self.assertRaises((ValueError, RuntimeError)):
                    resources(self.root)


if __name__ == '__main__':
    unittest.main()
