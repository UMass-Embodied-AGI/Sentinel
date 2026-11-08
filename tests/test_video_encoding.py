"""Opt-in CPU integration checks: RUN_VIDEO_ENCODING_TESTS=1 python -m unittest discover -s tests."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.make_video_demo import find_ffmpeg, frame_counts, write_video


@unittest.skipUnless(os.environ.get('RUN_VIDEO_ENCODING_TESTS') == '1',
                     'Set RUN_VIDEO_ENCODING_TESTS=1 to exercise real encoders.')
class VideoEncodingTests(unittest.TestCase):
    def test_both_writers_preserve_timing_and_frame_order_for_both_codecs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
            durations = [0.1, 0.2, 0.08]
            shots = []
            for index, (color, duration) in enumerate(zip(colors, durations)):
                path = root / f"shot '{index}.png"
                Image.new('RGB', (16, 16), color).save(path)
                shots.append((str(path), duration))
            counts = frame_counts(shots)
            expected = np.repeat(np.array(colors), counts, axis=0)
            run = subprocess.run

            def encode(command, **kwargs):
                result = run(command, capture_output=True, **kwargs)
                self.assertNotIn(b'invalid dropping', result.stderr)
                self.assertNotIn(b'non monoton', result.stderr.lower())
                return result

            for writer in ['default', 'ffmpeg']:
                for codec in ['h264', 'mpeg4']:
                    with self.subTest(writer=writer, codec=codec):
                        output = root / f'{writer}-{codec}.mp4'
                        args = argparse.Namespace(videowriter=writer, codec=codec, threads=1)
                        with patch('tools.make_video_demo.subprocess.run', side_effect=encode):
                            write_video(shots, output, args)
                        metadata = run([find_ffmpeg(), '-hide_banner', '-i', str(output)], capture_output=True)
                        self.assertIn(f'Video: {codec}'.encode(), metadata.stderr)
                        self.assertIn(b'24 fps', metadata.stderr)
                        result = run([find_ffmpeg(), '-v', 'error', '-i', str(output),
                                      '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                                     capture_output=True, check=True)
                        frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 16, 16, 3)
                        self.assertEqual(len(frames), sum(counts))
                        actual = frames.mean(axis=(1, 2))
                        np.testing.assert_allclose(actual, expected, atol=15)
            self.assertEqual(list(root.glob('*.ffconcat')), [])


if __name__ == '__main__':
    unittest.main()
