import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import accumulate
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, __version__ as pillow_version
import tqdm

if __package__:
    from .replay_layout import caption_pages, episode_extent, prepare_aerial, render_frame, font
else:
    from replay_layout import caption_pages, episode_extent, prepare_aerial, render_frame, font


def project_locators(locator_positions, camera_parameters):
    f_x = camera_parameters["camera_res"][0] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
    f_y = camera_parameters["camera_res"][1] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
    intrinsic_K = np.array([[f_x, 0.0, camera_parameters["camera_res"][0]/2.0],
                            [0.0, f_y, camera_parameters["camera_res"][1]/2.0],
                            [0.0, 0.0, 1.0]])
    extrinsic = np.array(camera_parameters["camera_extrinsics"])
    extrinsic = extrinsic[:3, :4]
    pixels = {}
    for name, pos in locator_positions.items():
        P_world = np.append(pos, 1.0)
        P_camera = extrinsic @ P_world
        P_image = intrinsic_K @ P_camera
        if np.isfinite(P_image).all() and P_image[2] > 0:
            pixels[name] = tuple(P_image[:2] / P_image[2])
    return pixels


VIDEO_FPS = 24

def find_ffmpeg():
    """Prefer the installed executable, then MoviePy's bundled imageio binary."""
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError('FFmpeg is unavailable; install ffmpeg or imageio-ffmpeg.') from error


def frame_counts(shots):
    if not shots:
        raise ValueError('Cannot encode an empty replay.')
    counts = []
    for _, duration in shots:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Frame durations must be finite and positive.')
        counts.append(max(1, round(duration * VIDEO_FPS)))
    return counts


def encoder_settings(codec):
    if codec == 'h264':
        return 'libx264', ['-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    if codec == 'mpeg4':
        return 'mpeg4', ['-pix_fmt', 'yuv420p']
    raise ValueError(f'Unsupported codec: {codec}')


def concat_path(path):
    """Escape paths for FFmpeg's concat syntax, which is separate from shell quoting."""
    value = str(Path(path).resolve())
    if '\n' in value or '\r' in value:
        raise ValueError('FFmpeg input paths cannot contain newlines.')
    return "'" + value.replace("'", "'\\''") + "'"


def images_to_video_using_ffmpeg(frame_durations, output_path, threads=8, codec='mpeg4'):
    """Encode timed stills with explicit timestamps, without repeatedly decoding each PNG."""
    counts = frame_counts(frame_durations)
    encoder, options = encoder_settings(codec)
    executable = find_ffmpeg()
    output_path = Path(output_path)
    # Validate paths before creating the manifest so invalid input cannot leak temporary files.
    paths = [concat_path(path) for path, _ in frame_durations]
    with tempfile.NamedTemporaryFile(mode='w', dir=output_path.parent,
                                     suffix='.ffconcat', delete=False) as stream:
        manifest = Path(stream.name)
        stream.write('ffconcat version 1.0\n')
        for path, count in zip(paths, counts):
            stream.write(f'file {path}\noption framerate {VIDEO_FPS}\nduration {count/VIDEO_FPS:.12f}\n')
        # The final repeated image establishes the endpoint for the preceding duration.
        stream.write(f'file {paths[-1]}\noption framerate {VIDEO_FPS}\n')
    command = [executable, '-y', '-f', 'concat', '-safe', '0', '-i', str(manifest),
               '-vf', f'fps={VIDEO_FPS}', '-fps_mode', 'passthrough', '-frames:v', str(sum(counts)),
               '-vcodec', encoder, '-threads', str(threads)]
    if codec == 'h264':
        command += ['-preset', 'slow']
    command += options + [str(output_path)]
    try:
        subprocess.run(command, check=True)
    finally:
        manifest.unlink(missing_ok=True)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EpisodeStep:
    positions: dict
    events: list
    clock: object


@dataclass
class RenderContext:
    episode_dir: Path
    demo_dir: Path
    scene: str
    cam_type: str
    fps: int
    overwrite: bool
    agent_names: list
    sentinel_names: list
    colors: dict
    map_tile: Image.Image
    extent: tuple
    steps: list
    cache_signature: str = ""


def read_step(index, episode_dir, names_order, camera_parameters, clock_agent):
    """Read each agent record once for positions, speech, and the replay clock."""
    poses, events, seen = {}, [], set()
    clock = None
    for name in names_order:
        path = episode_dir / 'steps' / name / f'{index:06d}.json'
        with path.open() as stream:
            record = json.load(stream)
        poses[name] = record['obs']['pose'][:3]
        if name == clock_agent:
            clock = record.get('curr_time')
        for event in record.get('obs', {}).get('events', []) or []:
            if event.get('type') == 'speech':
                key = (event.get('subject', name), event.get('content', ''))
                if key not in seen:
                    seen.add(key)
                    events.append(key)
    return EpisodeStep(project_locators(poses, camera_parameters), events, clock)


def timeline(steps, agent_names):
    focus = agent_names[0] if agent_names else None
    for index, step in enumerate(steps):
        yield index, focus
        for speaker, _ in step.events:
            if speaker in agent_names:
                focus = speaker


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cache_matches(filename, key):
    try:
        with filename.with_suffix('.json').open() as stream:
            metadata = json.load(stream)
        return metadata == {'key': key, 'image': file_signature(filename)}
    except (OSError, ValueError):
        return False


def save_frame(frame, filename, key):
    """Publish complete images and metadata, so interrupted writes are never reused."""
    for suffix in ('.png', '.json'):
        with tempfile.NamedTemporaryFile(dir=filename.parent, suffix=suffix, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            if suffix == '.png':
                frame.save(temporary, compress_level=2)
                temporary.replace(filename)
            else:
                temporary.write_text(json.dumps({'key': key, 'image': file_signature(filename)}))
                temporary.replace(filename.with_suffix('.json'))
        finally:
            temporary.unlink(missing_ok=True)


def process_frame(item, context):
    index, previous_focus = item
    step = context.steps[index]
    camera_signatures = {}
    shots = []
    for subject, message in step.events or [(None, '')]:
        if subject in context.agent_names:
            previous_focus = subject
        pages = caption_pages(message)
        for page, lines in enumerate(pages, 1):
            # 180 wpm for speech; quiet steps retain the original 2x replay speed.
            duration = max(1.6, len(' '.join(lines).split()) / 3) if subject else 1/(2*context.fps)
            filename = context.demo_dir / f'frame_{index:06d}_{len(shots):03d}.png'
            visible = list(context.agent_names[:6])
            if previous_focus and previous_focus not in visible:
                visible[-1:] = [previous_focus]
            cameras = {name: context.episode_dir / context.cam_type / name / f'rgb_{index:06d}.png'
                       for name in visible}
            for name, path in cameras.items():
                if name not in camera_signatures:
                    camera_signatures[name] = file_signature(path)
            key = digest([context.cache_signature, index, step.clock, step.positions,
                          previous_focus, subject, lines, page, len(pages),
                          {name: camera_signatures[name] for name in visible}])
            if context.overwrite or not cache_matches(filename, key):
                frame = render_frame(
                    scene=context.scene, step=index, total=len(context.steps), clock=step.clock,
                    map_image=None, cameras=cameras, colors=context.colors,
                    focus=previous_focus, speaker=subject, lines=lines, page=page, pages=len(pages),
                    cam_type=context.cam_type,
                    aerial=(context.map_tile, context.extent, step.positions, context.sentinel_names))
                save_frame(frame, filename, key)
            shots.append((str(filename), duration))
    return shots


def build_parser():
    parser = argparse.ArgumentParser(description='Render a saved episode without running the simulator.')
    parser.add_argument('--output_dir', '-o', type=Path, default=Path('output'),
                        help='Legacy episode root, used when --data_dir is omitted.')
    parser.add_argument('--scene', help='Scene name; overrides inference from --data_dir.')
    parser.add_argument('--scene_assets_dir', type=Path,
                        help='Scene directory containing global.png and global_cam_parameters.json.')
    parser.add_argument('--demo_dir', type=Path, help='Destination for rendered frames and demo.mp4.')
    parser.add_argument('--config', default='agents_num_15')
    parser.add_argument('--agent_type', default='tour_agent')
    parser.add_argument('--data_dir', '-d', type=Path, help='Episode directory containing curr_sim and steps.')
    parser.add_argument('--fps', type=int, default=5,
                        help='Replay speed parameter; quiet steps last 1/(2*fps) seconds.')
    parser.add_argument('--no_output_video', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Re-render even when the frame cache is valid.')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--cam_type', choices=['ego', 'tp'], default='ego')
    parser.add_argument('--videowriter', choices=['default', 'ffmpeg'], default='default')
    parser.add_argument('--codec', choices=['mpeg4', 'h264'], default='h264')
    parser.add_argument('--threads', type=int, default=16)
    return parser


def infer_scene(episode_dir):
    """Support benchmark SCENE/method/setting/job and legacy SCENE_agents_num_* layouts."""
    if episode_dir.name.startswith('job_') and len(episode_dir.parents) >= 3:
        return episode_dir.parents[2].name
    parent = episode_dir.parent.name
    if '_agents_num_' in parent:
        return parent.split('_agents_num_', 1)[0]
    raise ValueError('Cannot infer the scene from this episode path; supply --scene.')


def resolve_paths(args):
    if args.data_dir is not None:
        episode_dir = args.data_dir.resolve()
        scene = args.scene or infer_scene(episode_dir)
    else:
        scene = args.scene or 'NY'
        episode_dir = (args.output_dir / f'{scene}_{args.config}' / args.agent_type).resolve()
    scene_assets_dir = (args.scene_assets_dir or REPO_ROOT / 'assets' / 'scenes' / scene).resolve()
    demo_dir = (args.demo_dir or episode_dir / 'demo' /
                f'district_{args.cam_type}_{args.steps or "full"}').resolve()
    return episode_dir, scene, scene_assets_dir, demo_dir


def load_context(args):
    episode_dir, scene, assets_dir, demo_dir = resolve_paths(args)
    config_path = episode_dir / 'curr_sim' / 'config.json'
    for path in (config_path, assets_dir / 'global.png', assets_dir / 'global_cam_parameters.json'):
        if not path.is_file():
            raise ValueError(f'Required file not found: {path}')
    with config_path.open() as stream:
        config = json.load(stream)
    num_steps = min(config['step'], args.steps) if args.steps is not None else config['step']
    if num_steps < 1:
        raise ValueError('The replay must have at least one step.')
    names = config['agent_names']
    agents = [name for name in names if not re.fullmatch(r'Sentinel \d+', name)]
    sentinels = [name for name in names if name not in agents]
    if len(config['locator_colors']) != len(names):
        raise ValueError('Each agent must have a locator color.')
    colors = {name: tuple(round(255*c) for c in mcolors.to_rgb(color))
              for name, color in zip(names, config['locator_colors'])}
    with (assets_dir / 'global_cam_parameters.json').open() as stream:
        parameters = json.load(stream)
    reader = partial(read_step, episode_dir=episode_dir, names_order=names,
                     camera_parameters=parameters, clock_agent=agents[0] if agents else None)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        steps = list(tqdm.tqdm(executor.map(reader, range(num_steps)), total=num_steps, desc='Read episode'))
    with Image.open(assets_dir / 'global.png') as base:
        cw, ch = parameters['camera_res']
        if base.size != (cw, ch):
            for step in steps:
                step.positions = {name: (x*base.width/cw, y*base.height/ch)
                                  for name, (x, y) in step.positions.items()}
        extent = episode_extent(base.size, (p for step in steps for p in step.positions.values()))
        tile = prepare_aerial(base, extent)
    demo_dir.mkdir(parents=True, exist_ok=True)
    signature = digest({
        'scene': scene, 'cam_type': args.cam_type, 'fps': args.fps, 'steps': num_steps,
        'agents': agents, 'sentinels': sentinels, 'colors': colors, 'extent': extent,
        'map': [tile.size, hashlib.sha256(tile.tobytes()).hexdigest()],
        'source': [hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   (Path(__file__), Path(__file__).with_name('replay_layout.py'))],
        'pillow': pillow_version,
        'fonts': [file_signature(font(29, bold).path) for bold in (False, True)],
    })
    return RenderContext(episode_dir, demo_dir, scene, args.cam_type, args.fps, args.overwrite,
                         agents, sentinels, colors, tile, extent, steps, signature)


def write_video(shots, output, args):
    if args.videowriter == 'ffmpeg':
        images_to_video_using_ffmpeg(shots, str(output), threads=args.threads, codec=args.codec)
        return
    from moviepy import VideoClip

    ends = list(accumulate(frame_counts(shots)))
    encoder, options = encoder_settings(args.codec)

    @lru_cache(maxsize=1)
    def load(index):
        with Image.open(shots[index][0]) as image:
            return np.array(image)

    with VideoClip(lambda t: load(min(bisect_right(ends, round(t * VIDEO_FPS, 8)), len(ends)-1)),
                   duration=ends[-1]/VIDEO_FPS) as clip:
        clip.write_videofile(str(output), fps=VIDEO_FPS, codec=encoder, threads=args.threads,
                             preset='slow', ffmpeg_params=options)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1 or args.fps < 1 or (args.steps is not None and args.steps < 1):
        parser.error('--threads, --fps, and --steps must be positive.')
    try:
        context = load_context(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    renderer = partial(process_frame, context=context)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        shots = [shot for group in tqdm.tqdm(
            executor.map(renderer, timeline(context.steps, context.agent_names)),
            total=len(context.steps), desc='Render frames') for shot in group]
    if not args.no_output_video:
        write_video(shots, context.demo_dir / 'demo.mp4', args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
