import os
import re
from collections import defaultdict

import cv2
import sys
import json
import textwrap
import argparse
import subprocess
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors
from moviepy import ImageClip, CompositeVideoClip, concatenate_videoclips, ColorClip

_font_cache = {}

def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype("vico/tools/misc/OpenSans-Regular.ttf", size)
    return _font_cache[size]


# ---------------------------------------------------------------------------
# Styled background
# ---------------------------------------------------------------------------
_BG_NAVY       = (18,  35,  62)   # base dark navy
_BG_STRIP      = (26,  45,  78)   # top avatar strip
_BG_SIDE       = (22,  38,  68)   # side camera columns
_BG_SPEECH     = (14,  26,  50)   # bottom speech panel
_ACCENT_BLUE   = (74, 158, 191)   # bright teal-blue separator lines
_ACCENT_GOLD   = (200, 168,  75)  # warm gold corner accents
_DIVIDER       = ( 38,  58,  92)  # subtle camera-row dividers

_bg_cache = None
_avatar_img_cache = {}

def get_avatar_image(path, width):
    """Load, resize, and cache an avatar as RGBA. Cached per (path, width)."""
    key = (path, width)
    if key not in _avatar_img_cache:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        new_h = int(h * width / w)
        _avatar_img_cache[key] = img.resize((width, new_h), Image.LANCZOS)
    return _avatar_img_cache[key]

def make_background_image():
    """Build a styled static background PIL Image (computed once, then cached)."""
    global _bg_cache
    if _bg_cache is not None:
        return _bg_cache

    img = Image.new("RGB", (VIDEO_W, VIDEO_H), _BG_NAVY)
    draw = ImageDraw.Draw(img)

    # Top avatar strip — vertical gradient from lighter to base
    for y in range(TOP_H):
        t = y / TOP_H
        r = int(_BG_STRIP[0] + t * (_BG_NAVY[0] - _BG_STRIP[0]))
        g = int(_BG_STRIP[1] + t * (_BG_NAVY[1] - _BG_STRIP[1]))
        b = int(_BG_STRIP[2] + t * (_BG_NAVY[2] - _BG_STRIP[2]))
        draw.line([(0, y), (VIDEO_W, y)], fill=(r, g, b))

    # Left and right side camera columns
    draw.rectangle([(0, TOP_H), (SIDE_W, VIDEO_H - BOTTOM_H)], fill=_BG_SIDE)
    draw.rectangle([(VIDEO_W - SIDE_W, TOP_H), (VIDEO_W, VIDEO_H - BOTTOM_H)], fill=_BG_SIDE)

    # Bottom speech panel — vertical gradient from base to darker
    for y in range(BOTTOM_H):
        t = y / BOTTOM_H
        r = int(_BG_NAVY[0] + t * (_BG_SPEECH[0] - _BG_NAVY[0]))
        g = int(_BG_NAVY[1] + t * (_BG_SPEECH[1] - _BG_NAVY[1]))
        b = int(_BG_NAVY[2] + t * (_BG_SPEECH[2] - _BG_NAVY[2]))
        draw.line([(0, VIDEO_H - BOTTOM_H + y), (VIDEO_W, VIDEO_H - BOTTOM_H + y)], fill=(r, g, b))

    # Bright accent separator lines
    draw.rectangle([(0, TOP_H - 3),               (VIDEO_W, TOP_H)],                   fill=_ACCENT_BLUE)
    draw.rectangle([(0, VIDEO_H - BOTTOM_H),       (VIDEO_W, VIDEO_H - BOTTOM_H + 3)],  fill=_ACCENT_BLUE)
    draw.rectangle([(SIDE_W - 3, TOP_H),           (SIDE_W, VIDEO_H - BOTTOM_H)],        fill=_ACCENT_BLUE)
    draw.rectangle([(VIDEO_W - SIDE_W, TOP_H),     (VIDEO_W - SIDE_W + 3, VIDEO_H - BOTTOM_H)], fill=_ACCENT_BLUE)

    # Subtle horizontal dividers between camera rows (side columns only)
    row_h = MID_H // 3
    for r in range(1, 3):
        y = TOP_H + r * row_h
        draw.rectangle([(0, y - 1),           (SIDE_W, y + 1)],           fill=_DIVIDER)
        draw.rectangle([(VIDEO_W - SIDE_W, y - 1), (VIDEO_W, y + 1)],     fill=_DIVIDER)

    # Gold corner accents (top-left and top-right of the whole frame)
    L = 50
    T = 5
    draw.rectangle([(0, 0), (L, T)],  fill=_ACCENT_GOLD)
    draw.rectangle([(0, 0), (T, L)],  fill=_ACCENT_GOLD)
    draw.rectangle([(VIDEO_W - L, 0), (VIDEO_W, T)],  fill=_ACCENT_GOLD)
    draw.rectangle([(VIDEO_W - T, 0), (VIDEO_W, L)],  fill=_ACCENT_GOLD)

    # Thin gold rule just inside the bottom of the top strip
    draw.rectangle([(20, TOP_H - 7), (VIDEO_W - 20, TOP_H - 5)], fill=_ACCENT_GOLD)

    _bg_cache = img
    return _bg_cache

from concurrent.futures import ThreadPoolExecutor
from functools import partial


from vico.tools.utils import *

import time

VIDEO_W, VIDEO_H = 2560, 1920
TOP_H    = 200   # avatar strip
BOTTOM_H = 250   # speech text area
MID_H    = VIDEO_H - TOP_H - BOTTOM_H  # 1470
SIDE_W   = 400   # width of each side camera column
CENTER_W = VIDEO_W - 2 * SIDE_W        # 1760


def safe_image_clip(path, retries=5):
    for _ in range(retries):
        try:
            return ImageClip(path)
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Failed to read image: {path}")


def render_topdown_locators(image, locator_positions, colors, circle_radii, camera_parameters):
    f_x = camera_parameters["camera_res"][0] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
    f_y = camera_parameters["camera_res"][1] / (2.0 * np.tan(np.radians(camera_parameters["camera_fov"] / 2.0)))
    intrinsic_K = np.array([[f_x, 0.0, camera_parameters["camera_res"][0]/2.0],
                            [0.0, f_y, camera_parameters["camera_res"][1]/2.0],
                            [0.0, 0.0, 1.0]])
    extrinsic = np.array(camera_parameters["camera_extrinsics"])
    extrinsic = extrinsic[:3, :4]
    for pos, color, radius in zip(locator_positions, colors, circle_radii):
        P_world = np.append(pos, 1.0)
        P_camera = extrinsic @ P_world
        P_image = intrinsic_K @ P_camera
        pixel_x = int(P_image[0] / P_image[2])
        pixel_y = int(P_image[1] / P_image[2])
        cv2.circle(image, (pixel_x, pixel_y), radius, (color[2]*255, color[1]*255, color[0]*255), -1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)\.(?:png|jpe?g)$', filename)
    return int(match.group(1)) if match else -1


VIDEO_FPS = 24  # output video frame rate

def images_to_video_using_ffmpeg(frame_durations, output_path, threads=8, codec="mpeg4"):
    """
    frame_durations: list of (image_path, duration_seconds) tuples.
    Each frame is duplicated in the concat list as many times as needed to fill its
    duration at VIDEO_FPS — constant-fps encoding, compatible with all codecs.
    """
    with open("images_list.txt", "w") as f:
        for path, dur in frame_durations:
            n_copies = max(1, round(dur * VIDEO_FPS))
            for _ in range(n_copies):
                f.write(f"file '{os.path.abspath(path)}'\n")
    if codec == "mpeg4":
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "concat", "-safe", "0",
            "-r", str(VIDEO_FPS),
            "-i", "images_list.txt",
            "-vcodec", "mpeg4",
            "-threads", str(threads),
            output_path,
        ]
    elif codec == "h264":
        ffmpeg_cmd = [
            "ffmpeg",
            "-f", "concat", "-safe", "0",
            "-r", str(VIDEO_FPS),
            "-i", "images_list.txt",
            "-vcodec", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-threads", str(threads),
            output_path,
        ]
    else:
        print(f"Codec {codec} not supported.")
        exit()
    subprocess.run(ffmpeg_cmd, check=True)
    os.remove("images_list.txt")


def rgb_to_bgr255(color):
    rgb = mcolors.to_rgb(color)
    rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return rgb


def add_colored_dot(frame, position, radius, color):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame)
    x, y = position
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=color, outline=color)
    return frame


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


def process_frame_agents(frame_idx, args, names_order, non_sentinel_names, display_names_order,
                         name_to_color, agent_cam_images, global_images, demo_folder, last_text_actions):

    # --- Speech events and clip duration (2× speed) ---
    speech_events = collect_speech_events(frame_idx, args, names_order)
    if speech_events:
        total_words = sum(len(content.split()) for _, content in speech_events)
        clip_duration = max(1.5, total_words / 6.0) / 2.0
    else:
        clip_duration = 1.0 / (args.fps * 2)

    frame_image_path = os.path.join(demo_folder, f"frame_{frame_idx:06}.png")
    if not args.overwrite and os.path.exists(frame_image_path):
        return ImageClip(frame_image_path, duration=clip_duration)

    # Read current sim time from first available non-sentinel agent
    current_time = None
    for name in non_sentinel_names[:1]:
        step_file = os.path.join(args.output_dir, 'steps', name, f'{frame_idx:06d}.json')
        if os.path.exists(step_file):
            current_time = json.load(open(step_file)).get('curr_time')

    dynamic_clips = []
    # pil_texts: list of (x, y, text, font_size, color_rgb) drawn after composition
    pil_texts = []

    # --- Styled background (full frame, drawn once and cached) ---
    dynamic_clips.append(
        ImageClip(np.array(make_background_image()))
        .with_position((0, 0))
        .with_duration(clip_duration)
    )

    # --- Global view (center of middle area, as large as possible) ---
    global_img_clip = ImageClip(global_images[frame_idx])
    gw, gh = global_img_clip.size
    scale = min(CENTER_W / gw, MID_H / gh)
    new_gw = int(gw * scale)
    gx = SIDE_W + (CENTER_W - new_gw) // 2
    gy = TOP_H + (MID_H - int(gh * scale)) // 2
    dynamic_clips.append(
        global_img_clip.with_position((gx, gy)).with_duration(clip_duration).resized(width=new_gw)
    )
    if current_time:
        pil_texts.append((gx + 12, gy + 12, "Time: " + str(current_time), 40, (255, 255, 255)))

    # --- Top avatar strip (unique names; one Sentinel entry) ---
    # Avatars are pasted via PIL (not moviepy) so we can draw a white backing
    # behind transparent images for contrast against the dark background.
    n_display = len(display_names_order)
    avatar_x_spacing = VIDEO_W / n_display
    avatar_img_w = min(120, int(avatar_x_spacing - 15))
    for i, dname in enumerate(display_names_order):
        ax = int(10 + i * avatar_x_spacing)
        clr = name_color_bgr_255_display[dname]
        pil_texts.append((ax, 20, dname, 17, clr))

    # --- Side camera views: non-sentinel agents only (up to 6) ---
    cam_agents = non_sentinel_names[:6]
    row_h = MID_H // 3
    cam_view_w = SIDE_W - 20
    left_positions  = [(10,                    TOP_H + r * row_h) for r in range(3)]
    right_positions = [(VIDEO_W - SIDE_W + 10, TOP_H + r * row_h) for r in range(3)]

    for i, name in enumerate(cam_agents):
        pos = left_positions[i] if i < 3 else right_positions[i - 3]
        dynamic_clips.append(
            ImageClip(agent_cam_images[name][frame_idx])
            .with_position(pos)
            .with_duration(clip_duration)
            .resized(width=cam_view_w)
        )
        pil_texts.append((pos[0] + 6, pos[1] + 6, name, 18, (220, 220, 220)))

    # --- Bottom speech text ---
    if speech_events:
        y_text = VIDEO_H - BOTTOM_H + 18
        for subject, content in speech_events:
            if y_text > VIDEO_H - 32:
                break
            msg = textwrap.fill(f"{subject}: {content}", width=125)
            pil_texts.append((18, y_text, msg, 32, (255, 255, 180)))
            n_lines = msg.count('\n') + 1
            y_text += 38 * n_lines + 10

    # --- Compose frame (images only) then draw all text/avatars with PIL ---
    frame_clip = CompositeVideoClip(dynamic_clips, size=(VIDEO_W, VIDEO_H))
    frame = frame_clip.get_frame(0)
    frame_img = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(frame_img)

    # Paste avatar images with white rounded-rect backing for contrast
    PAD = 5
    for i, dname in enumerate(display_names_order):
        ax = int(10 + i * avatar_x_spacing)
        ay = 38
        av = get_avatar_image(avatar_images_display[dname], avatar_img_w)
        av_w, av_h = av.size
        draw.rounded_rectangle(
            [ax - PAD, ay - PAD, ax + av_w + PAD, ay + av_h + PAD],
            radius=8, fill=(240, 240, 240)
        )
        frame_img.paste(av, (ax, ay), av)

    for x, y, text, fsize, color in pil_texts:
        draw.text((x, y), text, fill=color, font=get_font(fsize))

    # Colored identity dots on avatar strip
    for i, dname in enumerate(display_names_order):
        ax = int(10 + i * avatar_x_spacing)
        av_h = get_avatar_image(avatar_images_display[dname], avatar_img_w).size[1]
        clr = name_color_bgr_255_display[dname]
        frame_img = add_colored_dot(frame_img, (ax + avatar_img_w // 2, 38 + av_h + PAD + 8), 8, clr)

    frame_img.save(frame_image_path)
    return ImageClip(frame_image_path, duration=clip_duration)


def make_global_image(i, args, names_order, camera_parameters):
    global_img_path = os.path.join(args.output_dir, 'global', f'rgb_{i:06d}.png')
    global_images[i] = global_img_path
    if os.path.exists(global_img_path) and not args.overwrite:
        return True
    agent_poses = []
    for name in names_order:
        step_data = json.load(open(os.path.join(args.output_dir, 'steps', name, f'{i:06d}.json')))
        agent_poses.append(step_data["obs"]["pose"])
    global_image_copy = global_image.copy()
    global_image_with_agents = render_topdown_locators(
        global_image_copy,
        [np.array(pose[:3]) for pose in agent_poses],
        agent_locator_colors,
        circle_radii=[15 for _ in agent_poses],
        camera_parameters=camera_parameters,
    )
    img = Image.fromarray(global_image_with_agents)
    img.save(global_img_path)
    img.close()
    return True


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
    parser.add_argument("--codec", choices=['mpeg4', 'h264'], default='mpeg4')
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    if args.data_dir is not None:
        args.data_dir = args.data_dir.rstrip('/')
        args.agent_type = args.data_dir.split('/')[-3]
        args.scene = args.data_dir.split('/')[-4].split('_')[0]
        args.output_dir = args.data_dir
    else:
        args.output_dir = os.path.join(args.output_dir, f"{args.scene}_{args.config}", f"{args.agent_type}")

    demo_folder = os.path.join(args.output_dir, 'demo')
    os.makedirs(demo_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'global'), exist_ok=True)

    config_path = os.path.join(args.output_dir, 'curr_sim', "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    names_order = config["agent_names"]
    num_agents  = config["num_agents"]
    name_to_color = {}
    last_text_actions = {}
    for agent_name, locator_color in zip(config["agent_names"], config["locator_colors"]):
        name_to_color[agent_name] = locator_color
        last_text_actions[agent_name] = None

    num_steps = config["step"]
    if args.steps:
        num_steps = min(num_steps, args.steps)

    # Separate sentinel vs non-sentinel agents
    SENTINEL_RE = re.compile(r'^Sentinel \d+$')
    non_sentinel_names = [n for n in names_order if not SENTINEL_RE.match(n)]
    sentinel_names     = [n for n in names_order if SENTINEL_RE.match(n)]

    # Display list for avatar strip: unique names, one "Sentinel" entry
    display_names_order = non_sentinel_names[:]
    if sentinel_names:
        display_names_order.append('Sentinel')

    # Build per-agent avatar image paths (keyed by actual agent name)
    avatar_images = {}
    agent_cam_images = defaultdict(dict)
    global_images = {}

    for name in names_order:
        for frame_idx in range(num_steps):
            cam_path = os.path.join(args.output_dir, args.cam_type, name, f'rgb_{frame_idx:06d}.png')
            assert os.path.exists(cam_path), f"Image {cam_path} does not exist."
            agent_cam_images[name][frame_idx] = cam_path
        avatar_name = SENTINEL_RE.sub('Sentinel', name)
        avatar_images[name] = os.path.join('assets', 'imgs', 'avatars', f'{avatar_name}.png')

    # Display avatar images keyed by display name (unique)
    avatar_images_display = {n: avatar_images[n] for n in non_sentinel_names}
    if sentinel_names:
        avatar_images_display['Sentinel'] = os.path.join('assets', 'imgs', 'avatars', 'Sentinel.png')

    # Color lookup tables
    name_color_bgr_255 = {n: rgb_to_bgr255(name_to_color[n]) for n in names_order}
    name_color_bgr_255_display = {n: name_color_bgr_255[n] for n in non_sentinel_names}
    if sentinel_names:
        name_color_bgr_255_display['Sentinel'] = name_color_bgr_255[sentinel_names[0]]

    global_image_path = os.path.join('assets', 'scenes', args.scene, "global.png")
    global_image = cv2.imread(global_image_path)

    camera_parameters = json.load(open(os.path.join('assets', 'scenes', args.scene, "global_cam_parameters.json")))
    agent_locator_colors = map_lang_colors_to_rgb(config["locator_colors"])

    # Generate global top-down images
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        global_image_fn = partial(make_global_image, args=args, names_order=names_order,
                                  camera_parameters=camera_parameters)
        tqdm.tqdm(executor.map(global_image_fn, range(0, num_steps, 1)), total=num_steps)
    for i in range(0, num_steps, 1):
        path = os.path.join(args.output_dir, 'global', f'rgb_{i:06d}.png')
        while not os.path.exists(path) or os.path.getsize(path) == 0:
            time.sleep(0.01)
    print("Finished generating global images.")

    # Render demo frames
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        process_fn = partial(
            process_frame_agents,
            args=args,
            names_order=names_order,
            non_sentinel_names=non_sentinel_names,
            display_names_order=display_names_order,
            name_to_color=name_to_color,
            agent_cam_images=agent_cam_images,
            global_images=global_images,
            demo_folder=demo_folder,
            last_text_actions=last_text_actions,
        )
        clips = list(tqdm.tqdm(executor.map(process_fn, range(0, num_steps, 1)), total=num_steps))

    clips = [c for c in clips if c is not None]

    if not args.no_output_video:
        if args.videowriter == "ffmpeg":
            frame_durations = [
                (os.path.join(demo_folder, f"frame_{i:06}.png"), clips[idx].duration)
                for idx, i in enumerate(range(0, num_steps, 1))
            ]
            images_to_video_using_ffmpeg(
                frame_durations,
                os.path.join(demo_folder, "demo.mp4"),
                threads=args.threads,
                codec=args.codec,
            )
        else:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(os.path.join(demo_folder, "demo.mp4"), fps=VIDEO_FPS)
