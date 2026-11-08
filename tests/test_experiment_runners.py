"""Check benchmark launch arguments without importing the simulator or running experiments."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name('fixtures') / 'runner_commands.json'


class ExperimentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # /tmp is mounted noexec on Unity. Keep the executable stub under the ignored output tree.
        (ROOT / 'output').mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'output')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.stub_dir = Path(cls.temporary.name)
        stub = cls.stub_dir / 'python'
        stub.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' "$@"\nexit "${RUNNER_EXIT_CODE:-0}"\n')
        stub.chmod(0o755)
        cls.env = dict(os.environ, PATH=f'{cls.stub_dir}:' + os.environ['PATH'])
        # Cluster shell initialization can override PATH and bypass the stub.
        cls.env.pop('BASH_ENV', None)
        with FIXTURES.open() as stream:
            cls.original_commands = json.load(stream)

    def run_runner(self, mode, method, args=None, cwd=ROOT, env=None):
        script = ROOT / 'sentinel_challenge' / f'{mode}_scripts' / f'run_{method}.sh'
        if args is None:
            args = ['EL_PASO', '3', 'patrol', '5', '0']
        return subprocess.run(['bash', str(script), *args], cwd=cwd, env=env or self.env,
                              capture_output=True, timeout=10)

    def argv(self, result):
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        return result.stdout.decode().rstrip('\0').split('\0')

    def test_all_standard_runners_preserve_original_arguments(self):
        for key, expected in self.original_commands.items():
            with self.subTest(runner=key):
                mode, method = key.split('/')
                self.assertEqual(self.argv(self.run_runner(mode, method)), expected)

    def test_runner_can_be_called_outside_repository(self):
        # The shared runner changes to the repository before invoking Python.
        probe = self.stub_dir / 'python'
        original = probe.read_text()
        try:
            probe.write_text('#!/usr/bin/env bash\nprintf \'%s\\0\' "$PWD" "$PYTHONPATH"\n')
            env = dict(self.env, PYTHONPATH='/extra/modules')
            result = self.run_runner('gt', 'cosar', cwd='/tmp', env=env)
            self.assertEqual(self.argv(result), [str(ROOT), f'{ROOT}:/extra/modules'])
        finally:
            probe.write_text(original)

    def test_explicit_checkpoint_paths_preserve_spaces(self):
        for mode in ['gt', 'no_gt']:
            for method in ['rl', 'mat']:
                with self.subTest(mode=mode, method=method):
                    checkpoint = '/private/checkpoints/best model.pt'
                    result = self.run_runner(mode, method, ['DETROIT', '5', 'stationary', '0', '2', checkpoint])
                    argv = self.argv(result)
                    self.assertEqual(argv[argv.index(f'--{method}_ckpt') + 1], checkpoint)
                    self.assertEqual(argv[argv.index('--sentinel_type') + 1], 'stationary')

    def test_invalid_arguments_fail_without_launching_python(self):
        cases = [[], ['DETROIT', '0', 'patrol', '5', '0'],
                 ['DETROIT', '3', 'unknown', '5', '0'],
                 ['DETROIT', '3', 'patrol', '-1', '0'],
                 ['DETROIT', '3', 'patrol', '5', 'invalid'],
                 ['DETROIT', '3', 'patrol', '5', '0', 'unexpected']]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_runner('gt', 'cosar', args)
                self.assertEqual(result.returncode, 2)
                self.assertIn(b'Usage:', result.stderr)
                self.assertEqual(result.stdout, b'')

    def test_python_exit_status_is_propagated(self):
        result = self.run_runner('gt', 'cosar', env=dict(self.env, RUNNER_EXIT_CODE='17'))
        self.assertEqual(result.returncode, 17)

    def test_shell_syntax_for_all_changed_runners(self):
        scripts = [ROOT/'sentinel_challenge/run_experiment.sh']
        for key in self.original_commands:
            mode, method = key.split('/')
            scripts.append(ROOT/'sentinel_challenge'/f'{mode}_scripts'/f'run_{method}.sh')
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(['bash', '-n', str(script)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
