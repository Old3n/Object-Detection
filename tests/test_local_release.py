import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import build_release as release


class LocalReleaseTests(unittest.TestCase):
    def test_version_validation(self):
        for value in ('v1.0.0', 'v2.1.3-beta.1'):
            self.assertEqual(release.version_tag(value), value)
        for value in ('main', '--help', '1.0.0', 'v1.0.0/other'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                release.version_tag(value)

    def test_dirty_checkout_is_rejected(self):
        with patch.object(release, 'run', return_value=' M AutoPilot.py'):
            with self.assertRaisesRegex(RuntimeError, 'Commit or stash'):
                release.clean_commit()

    def test_existing_tag_is_rejected_before_building(self):
        responses = ['', 'commit', '', 'WRITE', 'commit', '[{"ref":"refs/tags/v1.0.0"}]']
        with patch.object(release.shutil, 'which', return_value='installed'), patch.object(release, 'run', side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                release.prepare_publish('v1.0.0')

    def test_build_checks_executable_before_creating_release_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dist').mkdir()
            calls = []

            def run(*args, **kwargs):
                calls.append(tuple(str(arg) for arg in args))
                if 'PyInstaller' in args:
                    (root / 'dist' / 'AutoPilot.exe').write_bytes(b'executable fixture')
                if 'tools/package_pwa.py' in args:
                    (root / 'dist' / 'AutoPilot-pwa.zip').write_bytes(b'archive fixture')
                return 'package==1.0' if 'freeze' in args else ''

            with patch.object(release, 'ROOT', root), patch.object(release, 'run', side_effect=run):
                assets = release.build(Path('python.exe'))
            self.assertEqual(len(assets), 4)
            self.assertTrue(all(path.is_file() for path in assets))
            smoke = next(i for i, args in enumerate(calls) if '--self-test' in args)
            package = next(i for i, args in enumerate(calls) if 'tools/package_pwa.py' in args)
            self.assertLess(smoke, package)
            hashes = (root / 'dist' / 'SHA256SUMS.txt').read_text()
            self.assertIn(hashlib.sha256(b'executable fixture').hexdigest(), hashes)
            self.assertIn('AutoPilot.exe', hashes)

    def test_failed_executable_test_stops_packaging(self):
        calls = []

        def run(*args, **kwargs):
            calls.append(args)
            if '--self-test' in args:
                raise subprocess.CalledProcessError(1, args)
            return ''

        with patch.object(release, 'run', side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                release.build(Path('python.exe'))
        self.assertFalse(any('tools/package_pwa.py' in args for args in calls))
        self.assertFalse(any('gh' in args for args in calls))

    def test_changed_commit_prevents_upload(self):
        with patch.object(release, 'clean_commit', return_value='changed'), patch.object(release, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                release.publish('v1.0.0', 'original', [])
            run.assert_not_called()

    def test_publish_targets_built_commit_and_marks_prerelease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dist').mkdir()
            with patch.object(release, 'ROOT', root), patch.object(release, 'clean_commit', return_value='exact-commit'), patch.object(release, 'run') as run, patch('builtins.print'):
                release.publish('v1.0.0-beta.1', 'exact-commit', [root / 'dist' / 'AutoPilot.exe'])
            args = run.call_args.args
            self.assertEqual(args[args.index('--target') + 1], 'exact-commit')
            self.assertIn('--prerelease', args)
            self.assertIn('--notes-file', args)


if __name__ == '__main__':
    unittest.main()
