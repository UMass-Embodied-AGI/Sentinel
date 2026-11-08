"""Small replay regressions; no simulator or GPU required."""
import io
import json
from pathlib import Path
import tempfile
import subprocess

from PIL import Image
import unittest
from unittest.mock import patch

from tools.make_video_demo import (
    EpisodeStep, RenderContext, build_parser, frame_counts, images_to_video_using_ffmpeg,
    infer_scene, main, process_frame, read_step, resolve_paths, save_frame, timeline,
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


class EncodingValidationTests(unittest.TestCase):
    def test_duration_quantization(self):
        self.assertEqual(frame_counts([('a', 0.1), ('b', 0.2), ('c', 0.001)]), [2, 5, 1])

    def test_invalid_durations(self):
        for duration in [0, -1, float('nan'), float('inf')]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                frame_counts([('a', duration)])
        with self.assertRaises(ValueError):
            frame_counts([])

    def test_failed_encoding_cleans_up_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('tools.make_video_demo.find_ffmpeg', return_value='ffmpeg'), \
                    patch('tools.make_video_demo.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffmpeg')):
                with self.assertRaises(subprocess.CalledProcessError):
                    images_to_video_using_ffmpeg([('frame.png', 0.1)], root/'demo.mp4')
            self.assertEqual(list(root.glob('*.ffconcat')), [])


class FrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        camera = self.root/'ego/Alice/rgb_000000.png'
        camera.parent.mkdir(parents=True)
        Image.new('RGB', (16, 16), 'red').save(camera)
        self.camera = camera
        self.context = RenderContext(
            self.root, self.root, 'AUSTIN', 'ego', 5, False, ['Alice'], [],
            {'Alice': (255, 0, 0)}, Image.new('RGB', (16, 16)), (0, 0, 16, 16),
            [EpisodeStep({'Alice': (8, 8)}, [], 'noon')], 'layout-v1')
        self.renderer = patch('tools.make_video_demo.render_frame',
                              side_effect=lambda **kwargs: Image.new('RGB', (16, 16), 'blue')).start()
        self.addCleanup(patch.stopall)

    def render(self):
        return process_frame((0, 'Alice'), self.context)

    def test_unchanged_frames_are_reused(self):
        self.render()
        self.render()
        self.assertEqual(self.renderer.call_count, 1)

    def test_layout_camera_and_caption_changes_invalidate_cache(self):
        self.render()
        self.context.cache_signature = 'layout-v2'
        self.render()
        Image.new('RGB', (17, 17), 'green').save(self.camera)
        self.render()
        self.context.steps[0].events = [('Alice', 'Meet here.')]
        self.render()
        self.assertEqual(self.renderer.call_count, 4)

    def test_missing_or_corrupted_outputs_are_regenerated(self):
        filename = Path(self.render()[0][0])
        filename.unlink()
        self.render()
        filename.with_suffix('.json').write_text('invalid json')
        self.render()
        filename.write_bytes(b'incomplete image')
        self.render()
        self.assertEqual(self.renderer.call_count, 4)

    def test_overwrite_forces_render(self):
        self.render()
        self.context.overwrite = True
        self.render()
        self.assertEqual(self.renderer.call_count, 2)

    def test_failed_image_save_preserves_existing_output(self):
        filename = Path(self.render()[0][0])
        previous = filename.read_bytes()
        with patch.object(Image.Image, 'save', side_effect=OSError('out of space')):
            with self.assertRaises(OSError):
                save_frame(Image.new('RGB', (16, 16)), filename, 'new-key')
        self.assertEqual(filename.read_bytes(), previous)
        self.assertEqual(sorted(p.name for p in self.root.glob('*.png')), [filename.name])


if __name__ == '__main__':
    unittest.main()
