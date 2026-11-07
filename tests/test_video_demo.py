"""Small replay regressions; no simulator or GPU required."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.make_video_demo import (
    EpisodeStep, build_parser, infer_scene, main, read_step, resolve_paths, timeline,
)


class ReplayTests(unittest.TestCase):
    def test_scene_inference_preserves_underscores(self):
        self.assertEqual(infer_scene(Path('/episodes/EL_PASO/cosar/setting/job_1')), 'EL_PASO')
        self.assertEqual(infer_scene(Path('/output/EL_PASO_agents_num_5/cosar')), 'EL_PASO')

    def test_explicit_paths_work_with_arbitrary_episode_name(self):
        args = build_parser().parse_args([
            '--data_dir', '/records/custom', '--scene', 'EL_PASO',
            '--scene_assets_dir', '/private/scenes/EL_PASO', '--demo_dir', '/renders/new',
        ])
        self.assertEqual(resolve_paths(args), (
            Path('/records/custom'), 'EL_PASO', Path('/private/scenes/EL_PASO'), Path('/renders/new'),
        ))

    def test_legacy_episode_path(self):
        args = build_parser().parse_args(['--output_dir', '/output', '--scene', 'DETROIT'])
        episode, scene, _, demo = resolve_paths(args)
        self.assertEqual(episode, Path('/output/DETROIT_agents_num_15/tour_agent'))
        self.assertEqual(scene, 'DETROIT')
        self.assertEqual(demo, episode / 'demo/district_ego_full')

    def test_unknown_layout_requires_explicit_scene(self):
        with self.assertRaisesRegex(ValueError, '--scene'):
            infer_scene(Path('/records/custom'))

    def test_speech_deduplication_and_clock_use_same_records_as_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            speech = {'type': 'speech', 'subject': 'Alice', 'content': 'Meet here.'}
            for name in ['Alice', 'Bob']:
                folder = root / 'steps' / name
                folder.mkdir(parents=True)
                (folder / '000000.json').write_text(json.dumps({
                    'curr_time': name, 'obs': {'pose': [0, 0, 0], 'events': [speech, speech]},
                }))
            parameters = {'camera_res': [100, 100], 'camera_fov': 60,
                          'camera_extrinsics': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 10]]}
            with patch.object(Path, 'open', autospec=True, side_effect=Path.open) as opened:
                step = read_step(0, root, ['Alice', 'Bob'], parameters, 'Alice')
            self.assertEqual(opened.call_count, 2)
            self.assertEqual(step.events, [('Alice', 'Meet here.')])
            self.assertEqual(step.clock, 'Alice')
            self.assertEqual(step.positions, {'Alice': (50.0, 50.0), 'Bob': (50.0, 50.0)})

    def test_focus_persists_between_steps_and_ignores_unknown_speakers(self):
        steps = [EpisodeStep({}, [('Bob', 'Hello'), ('Sentinel 0', 'Wait')], None),
                 EpisodeStep({}, [], None), EpisodeStep({}, [('Alice', 'Yes')], None),
                 EpisodeStep({}, [], None)]
        self.assertEqual(list(timeline(steps, ['Alice', 'Bob'])),
                         [(0, 'Alice'), (1, 'Bob'), (2, 'Bob'), (3, 'Alice')])

    def test_invalid_limits_fail_before_reading_episode(self):
        for option in ['--threads', '--fps', '--steps']:
            with self.subTest(option=option), patch('tools.make_video_demo.load_context') as load, \
                    patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    main([option, '0'])
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
