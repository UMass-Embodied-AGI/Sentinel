import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from itertools import accumulate
import json
import os
import re
import subprocess
import tempfile

import matplotlib.colors as mcolors
from moviepy import VideoClip
import numpy as np
from PIL import Image
import tqdm

if __package__:
    from .replay_layout import caption_pages, episode_extent, prepare_aerial, render_frame
else:
    from replay_layout import caption_pages, episode_extent, prepare_aerial, render_frame


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

def images_to_video_using_ffmpeg(frame_durations, output_path, threads=8, codec="mpeg4"):
    """
    frame_durations: list of (image_path, duration_seconds) tuples.
    Each frame is duplicated in the concat list as many times as needed to fill its
    duration at VIDEO_FPS — constant-fps encoding, compatible with all codecs.
    """
    with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(output_path), suffix=".txt", delete=False) as f:
        manifest = f.name
        for path, dur in frame_durations:
            n_copies = max(1, round(dur * VIDEO_FPS))
            for _ in range(n_copies):
                f.write(f"file '{os.path.abspath(path)}'\n")
    if codec == "mpeg4":
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "concat", "-safe", "0",
            "-r", str(VIDEO_FPS),
            "-i", manifest,
            "-vcodec", "mpeg4",
            "-threads", str(threads),
            output_path,
        ]
    elif codec == "h264":
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "concat", "-safe", "0",
            "-r", str(VIDEO_FPS),
            "-i", manifest,
            "-vcodec", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-threads", str(threads),
            output_path,
        ]
    else:
        print(f"Codec {codec} not supported.")
        exit()
    try:
        ffmpeg_cmd[1:1] = ["-y"]
        if codec == "h264":
            ffmpeg_cmd[-1:-1] = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        subprocess.run(ffmpeg_cmd, check=True)
    finally:
        os.remove(manifest)


def collect_speech_events(frame_idx, args, names_order):
    """Return unique (subject, content) speech events observed at this frame across all agents."""
    seen = set()
    events_out = []
    for name in names_order:
        step_file = os.path.join(args.output_dir, 'steps', name, f'{frame_idx:06d}.json')
        if not os.path.exists(step_file):
            continue
        step_data = json.load(open(step_file))
        for event in (step_data.get('obs', {}).get('events', []) or []):
            if event.get('type') == 'speech':
                key = (event.get('subject', name), event.get('content', ''))
                if key not in seen:
                    seen.add(key)
                    events_out.append(key)
    return events_out


def process_frame(item):
    step, events, previous_focus = item
    clock = None
    if non_sentinel_names:
        record = os.path.join(args.output_dir, "steps", non_sentinel_names[0], f"{step:06d}.json")
        with open(record) as f:
            clock = json.load(f).get("curr_time")

    shots = []
    for subject, message in events or [(None, "")]:
        if subject in non_sentinel_names:
            previous_focus = subject
        pages = caption_pages(message)
        for p, lines in enumerate(pages, 1):
            # 180 wpm for speech; quiet steps retain the original 2x replay speed.
            duration = max(1.6, len(" ".join(lines).split()) / 3) if subject else 1/(2*args.fps)
            filename = os.path.join(demo_folder, f"frame_{step:06d}_{len(shots):03d}.png")
            if args.overwrite or not os.path.isfile(filename):
                # Keep up to six feeds legible, including speakers beyond the first six.
                visible = list(non_sentinel_names[:6])
                if previous_focus and previous_focus not in visible:
                    visible[-1:] = [previous_focus]
                cameras = {n: os.path.join(args.output_dir, args.cam_type, n, f"rgb_{step:06d}.png")
                           for n in visible}
                frame = render_frame(scene=args.scene, step=step, total=num_steps, clock=clock,
                    map_image=None, cameras=cameras, colors=colors,
                    focus=previous_focus, speaker=subject, lines=lines, page=p, pages=len(pages),
                    cam_type=args.cam_type, aerial=(map_tile, extent, position_frames[step], sentinel_names))
                frame.save(filename, compress_level=2)
            shots.append((filename, duration))
    return shots


def read_positions(i, args, names_order, camera_parameters):
    poses = {}
    for name in names_order:
        with open(os.path.join(args.output_dir, 'steps', name, f'{i:06d}.json')) as f:
            poses[name] = json.load(f)["obs"]["pose"][:3]
    return project_locators(poses, camera_parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default='output')
    parser.add_argument("--scene", type=str, default='NY')
    parser.add_argument("--config", type=str, default='agents_num_15')
    parser.add_argument("--agent_type", type=str, choices=['tour_agent'], default='tour_agent')
    parser.add_argument("--data_dir", "-d", type=str)
    parser.add_argument("--fps", type=int, default=5,
                        help="Steps shown per second during normal (non-speech) playback.")
    parser.add_argument("--no_output_video", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cam_type", choices=['ego', 'tp'], default='ego')
    parser.add_argument("--videowriter", choices=['default', 'ffmpeg'], default='default')
    parser.add_argument("--codec", choices=['mpeg4', 'h264'], default='h264')
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if args.data_dir is not None:
        args.data_dir = args.data_dir.rstrip('/')
        args.agent_type = args.data_dir.split('/')[-3]
        args.scene = args.data_dir.split('/')[-4].split('_')[0]
        args.output_dir = args.data_dir
    else:
        args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")

    demo_folder = os.path.join(args.output_dir, 'demo', f'district_{args.cam_type}_{args.steps or "full"}')
    os.makedirs(demo_folder, exist_ok=True)

    config_path = os.path.join(args.output_dir, 'curr_sim', "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    names_order = config["agent_names"]
    num_steps = min(config["step"], args.steps) if args.steps else config["step"]
    if num_steps < 1 or args.fps <= 0:
        parser.error("The replay must have at least one step and a positive --fps.")

    non_sentinel_names = [n for n in names_order if not re.fullmatch(r"Sentinel \d+", n)]
    sentinel_names = [n for n in names_order if n not in non_sentinel_names]
    colors = {n: tuple(round(255*c) for c in mcolors.to_rgb(color))
              for n, color in zip(names_order, config["locator_colors"])}
    global_image_path = os.path.join('assets', 'scenes', args.scene, "global.png")

    camera_parameters = json.load(open(os.path.join('assets', 'scenes', args.scene, "global_cam_parameters.json")))
    # Frame the complete run once. The map stays still as speakers change.
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        get_positions = partial(read_positions, args=args, names_order=names_order,
                                camera_parameters=camera_parameters)
        position_frames = list(tqdm.tqdm(executor.map(get_positions, range(num_steps)), total=num_steps))
    with Image.open(global_image_path) as base:
        # Projected coordinates are in calibration pixels. A resized base image
        # must still place the markers on exactly the same streets.
        cw, ch = camera_parameters["camera_res"]
        if base.size != (cw, ch):
            position_frames = [{n: (x*base.width/cw, y*base.height/ch)
                                for n, (x, y) in step.items()} for step in position_frames]
        extent = episode_extent(base.size, (p for step in position_frames for p in step.values()))
        map_tile = prepare_aerial(base, extent)

    def timeline():
        focus = non_sentinel_names[0] if non_sentinel_names else None
        for step in range(num_steps):
            events = collect_speech_events(step, args, names_order)
            yield step, events, focus
            for speaker, _ in events:
                if speaker in non_sentinel_names:
                    focus = speaker

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        shots = [shot for group in tqdm.tqdm(executor.map(process_frame, timeline()), total=num_steps)
                 for shot in group]
    if not args.no_output_video:
        output = os.path.join(demo_folder, "demo.mp4")
        if args.videowriter == "ffmpeg":
            images_to_video_using_ffmpeg(shots, output, threads=args.threads, codec=args.codec)
        else:
            ends = list(accumulate(duration for _, duration in shots))

            @lru_cache(maxsize=1)
            def load(i):
                with Image.open(shots[i][0]) as im:
                    return np.array(im)

            with VideoClip(lambda t: load(min(bisect_right(ends, t), len(ends)-1)),
                           duration=ends[-1]) as clip:
                clip.write_videofile(output, fps=VIDEO_FPS, codec="libx264",
                                     ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
