"""Pillow-only presentation layer. No simulator, GPU, or video decoder required."""
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


SIZE = (1600, 900)
INK, GREEN, PAPER = "#112d25", "#234439", "#f5f4e9"
LIME, MUTED, WHITE = "#d9ee86", "#b5cbbd", "#f8f8ee"
CAPTION_WIDTH, CAPTION_LINES = 731, 5


@lru_cache(maxsize=32)
def font(size, bold=False):
    style = "Bold" if bold else "Regular"
    for path in (f"vico/tools/misc/OpenSans-{style}.ttf",
                 f"DejaVuSans{'-Bold' if bold else ''}.ttf",
                 f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
                 "vico/tools/misc/OpenSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError("Install DejaVu Sans or supply vico/tools/misc/OpenSans-Regular.ttf")


def caption_pages(message):
    """Pixel-wrap the complete message; split long words and paginate, never elide."""
    face = font(29)
    lines, line = [], ""
    for word in message.split():
        combined = (line + " " + word).strip()
        if face.getlength(combined) <= CAPTION_WIDTH:
            line = combined
            continue
        if line:
            lines.append(line)
        line = ""
        for char in word:
            if face.getlength(line + char) > CAPTION_WIDTH:
                lines.append(line)
                line = ""
            line += char
    if line:
        lines.append(line)
    return [lines[i:i + CAPTION_LINES] for i in range(0, len(lines), CAPTION_LINES)] or [[]]


def _fit_text(text, face, width):
    if face.getlength(text) <= width:
        return text
    while text and face.getlength(text + "…") > width:
        text = text[:-1]
    return text + "…"


def _photo(canvas, source, box):
    """Contain (not crop) observations: keep the original field of view."""
    x, y, w, h = box
    def paste(im):
        tile = ImageOps.contain(im.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x + (w - tile.width)//2, y + (h - tile.height)//2))
    if isinstance(source, (str, Path)):
        with Image.open(source) as im:
            paste(im)
    else:
        paste(source)


def episode_extent(size, points):
    """One fixed square-ish viewport for the run; never chase a moving marker."""
    w, h = size
    xs, ys = [], []
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0, 0, w, h)
    scale = min(1, max(.44, (max(xs)-min(xs))/w*1.32, (max(ys)-min(ys))/h*1.32))
    cw, ch = max(1, round(w*scale)), max(1, round(h*scale))
    left = min(w-cw, max(0, round((max(xs)+min(xs)-cw)/2)))
    top = min(h-ch, max(0, round((max(ys)+min(ys)-ch)/2)))
    return (left, top, left+cw, top+ch)


def prepare_aerial(base, extent):
    """Mute the static photography once, before adding live positional data."""
    crop = base.crop(extent).convert("RGB")
    crop = ImageEnhance.Color(crop).enhance(.32)
    crop = ImageEnhance.Contrast(crop).enhance(.80)
    crop = Image.blend(crop, Image.new("RGB", crop.size, INK), .22)
    return ImageOps.contain(crop, (710, 690), Image.Resampling.LANCZOS)


def aerial_panel(tile, extent, positions, colors, focus, sentinels):
    panel = tile.copy()
    d = ImageDraw.Draw(panel)
    left, top, right, bottom = extent
    w, h = panel.size
    points = {name: (round((x-left)*w/(right-left)), round((y-top)*h/(bottom-top)))
              for name, (x, y) in positions.items() if left <= x < right and top <= y < bottom}
    # Render the focus last, so collocated teammates cannot hide it.
    for name in sorted(points, key=lambda n: n == focus):
        x, y = points[name]
        if name in sentinels:
            d.polygon([(x,y-9),(x+9,y),(x,y+9),(x-9,y)], fill="#ff976c", outline=INK,width=2)
        else:
            d.ellipse((x-8,y-8,x+8,y+8), fill=colors.get(name, WHITE), outline=WHITE,width=2)
        if name == focus:
            d.ellipse((x-15,y-15,x+15,y+15),outline=INK,width=5)
            d.ellipse((x-14,y-14,x+14,y+14),outline=LIME,width=3)
    if focus in points and w >= 200 and h >= 120:
        x,y = points[focus]
        face = font(19, True)
        name = _fit_text(focus, face, w-52)
        bw, bh = round(face.getlength(name))+24, 34
        # Choose a nearby side with the fewest other markers beneath the tag.
        boxes = []
        for dx,dy in ((25,-47),(25,14),(-bw-25,-47),(-bw-25,14)):
            bx,by = max(3,min(w-bw-3,x+dx)),max(3,min(h-bh-3,y+dy))
            hits = sum(bx-10<=px<=bx+bw+10 and by-10<=py<=by+bh+10 for px,py in points.values())
            boxes.append((hits,bx,by))
        _,bx,by=min(boxes,key=lambda b:b[0])
        end=(min(max(x,bx),bx+bw), min(max(y,by),by+bh))
        d.line((x,y,*end),fill=LIME,width=2)
        d.ellipse((x-5,y-5,x+5,y+5),fill=colors.get(focus,WHITE))
        d.rounded_rectangle((bx,by,bx+bw,by+bh),radius=6,fill=INK,outline=LIME)
        d.text((bx+12,by+7),name,font=face,fill=WHITE,anchor="lt")
    return panel


def render_frame(*, scene, step, total, clock, map_image, cameras, colors,
                 focus, speaker=None, lines=(), page=1, pages=1, cam_type="ego",
                 sentinel_color=None, aerial=None):
    frame = Image.new("RGB", SIZE, INK)
    d = ImageDraw.Draw(frame)
    def label(x, y, text, size=19, color=MUTED, bold=False):
        d.text((x, y), text, font=font(size, bold), fill=color, anchor="lt")
    def name_tag(x, y, name, width, size):
        d.ellipse((x, y+3, x+13, y+16), fill=colors.get(name, WHITE), outline=WHITE)
        label(x+23, y, _fit_text(name, font(size, True), width-23), size, WHITE, True)

    label(36, 21, "Sentinel", 43, WHITE, True)
    label(254, 36, "EPISODE REPLAY", 19, LIME, True)
    meta = f"{scene}   /   STEP {step+1} OF {total}"
    d.text((1564, 38), _fit_text(meta, font(21), 680), font=font(21), fill=WHITE, anchor="rt")
    d.line((36, 84, 1564, 84), fill="#577465", width=2)
    label(36, 107, "01   DISTRICT VIEW" if aerial else "01   CITY OVERVIEW", color=LIME, bold=True)
    label(786, 107, "02   AGENT CAMERAS", color=LIME, bold=True)
    if aerial:
        tile, extent, positions, sentinels = aerial
        panel = aerial_panel(tile, extent, positions, colors, focus, sentinels)
        frame.paste(panel, (36+(710-panel.width)//2, 145+(690-panel.height)//2))
        d.ellipse((36,852,50,866),fill=GREEN,outline=WHITE,width=2)
        label(59,850,"Agent",18)
        d.polygon([(151,851),(159,859),(151,867),(143,859)],fill="#ff976c")
        label(169,850,"Sentinel",18)
        d.ellipse((278,851,294,867),outline=LIME,width=3)
        label(304,850,"Featured camera",18)
    else:
        _photo(frame, map_image, (36, 145, 710, 690))
    if not aerial and sentinel_color is not None:
        d.ellipse((37, 853, 51, 867), fill=sentinel_color, outline=WHITE)
        label(62, 851, "Sentinel", 18)
    if clock is not None:
        d.text((746, 851), _fit_text(str(clock), font(18), 235 if aerial else 500), font=font(18),
               fill=MUTED, anchor="rt")

    # Lock the main camera to the speaker; retain that camera between messages.
    if focus in cameras:
        _photo(frame, cameras[focus], (786, 145, 367, 367))
        name_tag(786, 525, focus, 365, 23)
    others = [n for n in cameras if n != focus]
    cols = 2 if len(others) > 1 else 1
    rows = max(1, (len(others) + cols - 1)//cols)
    w, h = 391//cols, 405//rows
    for i, n in enumerate(others):
        x, y = 1173 + i%cols*w, 145 + i//cols*h
        _photo(frame, cameras[n], (x, y, w-8, max(1, h-34)))
        name_tag(x, y+h-29, n, w-8, 15)

    d.rounded_rectangle((786, 571, 1564, 857), radius=14, fill=PAPER)
    label(810, 594, "03   TEAM CHANNEL", 17, GREEN, True)
    if speaker is not None:
        label(810, 630, _fit_text(speaker, font(24, True), 595), 24, INK, True)
        if pages > 1:
            label(1455, 634, f"{page} / {pages}", 17, GREEN)
        for i, line in enumerate(lines):
            label(810, 673 + 35*i, line, 29, INK)
    else:
        label(810, 665, "No new message", 30, GREEN)
        label(810, 717, "First-person cameras" if cam_type == "ego" else "Third-person cameras",
              19, GREEN)
    d.rectangle((0, 892, SIZE[0]-1, 899), fill=GREEN)
    d.rectangle((0, 892, max(0, round(SIZE[0]*(step+1)/max(1,total))-1), 899), fill=LIME)
    return frame
