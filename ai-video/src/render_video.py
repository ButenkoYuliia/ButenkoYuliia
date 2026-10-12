"""Lo-Fi loop "Murchyk & the Rain": procedural pixel-art frames.

World is drawn at 480x270 (pixel art), upscaled 4x to 1920x1080, then a
virtual camera crops it per shot (wide / medium / close-up).
Every animation is periodic in LOOP seconds, so the video loops seamlessly.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 480, 270
OUT_W, OUT_H = 1920, 1080
FPS = 24
LOOP = 32.0
N_FRAMES = int(LOOP * FPS)
TAU = 2 * math.pi

rng = np.random.default_rng(7)

# ---------- palette (fixed for consistency across all shots) ----------
WALL_TOP = np.array([34, 30, 58])
WALL_BOT = np.array([52, 40, 70])
SKY_TOP = (14, 18, 44)
SKY_BOT = (40, 44, 92)
FRAME = (70, 46, 40)
FRAME_HI = (98, 66, 54)
DESK = (110, 72, 56)
DESK_TOP = (140, 94, 70)
DESK_DARK = (78, 50, 42)
CAT = (226, 140, 64)
CAT_DARK = (186, 104, 46)
CAT_LIGHT = (246, 190, 120)
SCARF = (200, 60, 70)
SCARF_DARK = (150, 40, 52)
MUG = (120, 170, 190)
MUG_DARK = (86, 128, 150)
COFFEE = (70, 44, 34)
LEAF = (88, 150, 96)
LEAF_DARK = (58, 110, 72)
POT = (176, 96, 72)
LAMP = (240, 200, 120)
LAMP_DARK = (180, 130, 80)
BOOKS = [(170, 80, 80), (90, 120, 170), (210, 180, 100), (120, 90, 150)]

WIN = (150, 34, 330, 168)  # outside window rect x0,y0,x1,y1

# ---------- static random content ----------
buildings = []
x = WIN[0]
while x < WIN[2]:
    bw = int(rng.integers(14, 30))
    bh = int(rng.integers(30, 90))
    lit = []
    for wy in range(WIN[3] - bh + 6, WIN[3] - 4, 7):
        for wx in range(x + 3, x + bw - 3, 5):
            if rng.random() < 0.45:
                # each window toggles at random times inside the loop
                period = int(rng.choice([1, 2, 4]))
                phase = rng.random()
                duty = rng.uniform(0.6, 1.0)
                col = tuple(int(c) for c in rng.choice(
                    [[255, 210, 120], [255, 180, 100], [180, 220, 255]]))
                lit.append((wx, wy, period, phase, duty, col))
    buildings.append((x, bw, bh, lit))
    x += bw + int(rng.integers(0, 4))

N_RAIN = 140
rain = [(rng.uniform(WIN[0], WIN[2]), rng.uniform(0, 1),
         int(rng.integers(6, 11)), rng.uniform(0.35, 1.0)) for _ in range(N_RAIN)]
# glass drops: slide slowly down, k laps per loop
glass = [(rng.uniform(WIN[0] + 4, WIN[2] - 4), rng.uniform(0, 1),
          int(rng.integers(1, 3))) for _ in range(26)]
stars = [(int(rng.integers(WIN[0], WIN[2])), int(rng.integers(WIN[1], WIN[1] + 40)),
          rng.random()) for _ in range(18)]

# vertical gradients precomputed
yy = np.linspace(0, 1, H)[:, None]
wall = (WALL_TOP[None, None, :] * (1 - yy[..., None]) +
        WALL_BOT[None, None, :] * yy[..., None]).repeat(W, axis=1)

# warm lamp glow (radial), applied additively
gx, gy = np.meshgrid(np.arange(W), np.arange(H))
LAMP_C = (398, 150)
glow_base = np.exp(-(((gx - LAMP_C[0]) ** 2 + (gy - LAMP_C[1]) ** 2) / (2 * 70.0 ** 2)))
glow_col = np.array([255, 170, 90]) / 255.0
# soft vignette
vig = 1 - 0.35 * (((gx - W / 2) / (W / 2)) ** 2 + ((gy - H / 2) / (H / 2)) ** 2)
vig = np.clip(vig, 0.55, 1)[..., None]

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 9)
except OSError:
    FONT = ImageFont.load_default()


def per(t, n=1):
    """phase in [0,1) that repeats n times per loop"""
    return (t * n / LOOP) % 1.0


def draw_world(t):
    img = Image.fromarray(wall.astype(np.uint8))
    d = ImageDraw.Draw(img)

    # --- outside: sky gradient ---
    x0, y0, x1, y1 = WIN
    for y in range(y0, y1):
        k = (y - y0) / (y1 - y0)
        c = tuple(int(SKY_TOP[i] * (1 - k) + SKY_BOT[i] * k) for i in range(3))
        d.line([(x0, y), (x1, y)], fill=c)
    for sx, sy, ph in stars:
        if math.sin(TAU * (per(t, 3) + ph)) > -0.3:
            d.point((sx, sy), fill=(200, 200, 240))
    # moon behind clouds
    d.ellipse([292, 46, 310, 64], fill=(236, 230, 200))
    d.ellipse([297, 43, 315, 61], fill=SKY_TOP)
    # buildings
    for bx, bw, bh, lit in buildings:
        d.rectangle([bx, y1 - bh, bx + bw, y1], fill=(22, 22, 40))
        for wx, wy, period, phase, duty, col in lit:
            if per(t, period) + phase - math.floor(per(t, period) + phase) < duty:
                d.rectangle([wx, wy, wx + 1, wy + 2], fill=col)
    # rain outside (periodic: each drop falls k full lengths per loop)
    span = y1 - y0 + 12
    for rx, ry0, k, a in rain:
        y = y0 - 10 + ((ry0 + per(t, k * 4)) % 1.0) * span
        cx = rx - (y - y0) * 0.12
        c = tuple(int(150 + 80 * a) for _ in range(2)) + (255,)
        d.line([(cx, y), (cx - 1, y + 5)], fill=(int(120 + 90 * a), int(140 + 90 * a), 255))
    # drops on glass
    for gx_, gy0, k in glass:
        y = y0 + ((gy0 + per(t, k)) % 1.0) * (y1 - y0)
        d.point((gx_, y), fill=(200, 220, 255))
        d.point((gx_, y - 1), fill=(150, 170, 220))
        d.point((gx_, y - 2), fill=(110, 130, 190))

    # --- window frame ---
    d.rectangle([x0 - 6, y0 - 6, x1 + 6, y0 - 1], fill=FRAME)
    d.rectangle([x0 - 6, y0 - 6, x0 - 1, y1], fill=FRAME)
    d.rectangle([x1 + 1, y0 - 6, x1 + 6, y1], fill=FRAME)
    d.rectangle([(x0 + x1) // 2 - 2, y0, (x0 + x1) // 2 + 2, y1], fill=FRAME)
    d.rectangle([x0, (y0 + y1) // 2 - 2, x1, (y0 + y1) // 2 + 1], fill=FRAME)
    d.line([(x0 - 6, y0 - 6), (x1 + 6, y0 - 6)], fill=FRAME_HI)
    d.rectangle([x0 - 10, y1, x1 + 10, y1 + 5], fill=FRAME_HI)
    # curtains
    sway = math.sin(TAU * per(t, 2)) * 1.5
    d.polygon([(x0 - 22, y0 - 12), (x0 - 4, y0 - 12), (x0 + 2 + sway, y1 + 4), (x0 - 24, y1 + 4)],
              fill=(120, 62, 86))
    d.polygon([(x1 + 4, y0 - 12), (x1 + 22, y0 - 12), (x1 + 24, y1 + 4), (x1 - 2 + sway, y1 + 4)],
              fill=(120, 62, 86))
    d.rectangle([x0 - 28, y0 - 15, x1 + 28, y0 - 12], fill=(60, 40, 40))

    # --- poster on the wall ---
    d.rectangle([40, 50, 92, 112], fill=(214, 200, 170))
    d.rectangle([44, 54, 88, 92], fill=(80, 110, 150))
    d.ellipse([58, 62, 74, 78], fill=(240, 180, 110))
    d.polygon([(44, 92), (62, 74), (78, 88), (88, 80), (88, 92)], fill=(50, 70, 100))
    d.text((47, 96), "HOME", fill=(70, 50, 50), font=FONT)

    # --- desk ---
    d.rectangle([0, 176, W, 182], fill=DESK_TOP)
    d.rectangle([0, 183, W, H], fill=DESK)
    d.line([(0, 183), (W, 183)], fill=DESK_DARK)
    for yb in range(196, H, 18):
        d.line([(0, yb), (W, yb)], fill=DESK_DARK)

    # --- plant (left) ---
    d.polygon([(80, 176), (106, 176), (102, 150), (84, 150)], fill=POT)
    d.rectangle([82, 148, 104, 152], fill=(150, 80, 60))
    for i, (bx, ang, ln) in enumerate([(93, -1.2, 34), (93, -1.9, 30), (93, -0.7, 28),
                                       (93, -2.4, 26), (93, -1.55, 40)]):
        a = ang + 0.06 * math.sin(TAU * per(t, 2) + i)
        ex = bx + math.cos(a) * ln
        ey = 148 + math.sin(a) * ln
        mx = bx + math.cos(a) * ln * 0.5 + 4
        my = 148 + math.sin(a) * ln * 0.5
        d.polygon([(bx, 148), (mx, my - 3), (ex, ey), (mx - 6, my + 3)],
                  fill=LEAF if i % 2 else LEAF_DARK)

    # --- books (right) ---
    by = 176
    for i, c in enumerate(BOOKS):
        h = 6
        off = (i % 2) * 3
        d.rectangle([330 + off, by - h, 372 + off, by - 1], fill=c)
        d.line([(330 + off, by - 2), (372 + off, by - 2)], fill=(240, 230, 210))
        by -= h

    # --- lamp ---
    d.rectangle([396, 168, 402, 176], fill=LAMP_DARK)
    d.rectangle([388, 174, 410, 177], fill=(70, 60, 60))
    d.line([(399, 168), (399, 140)], fill=(70, 60, 60), width=2)
    d.polygon([(384, 140), (414, 140), (406, 120), (392, 120)], fill=LAMP)
    d.line([(384, 140), (414, 140)], fill=(255, 235, 170))

    # --- mug (with steam) ---
    mx0, my0 = 296, 158
    d.rectangle([mx0, my0, mx0 + 16, 176], fill=MUG)
    d.rectangle([mx0, my0, mx0 + 3, 176], fill=MUG_DARK)
    d.ellipse([mx0 + 13, 162, mx0 + 23, 172], outline=MUG, width=2)
    d.rectangle([mx0 + 1, my0, mx0 + 15, my0 + 2], fill=COFFEE)
    d.text((mx0 + 5, 164), "♥", fill=(230, 240, 250), font=FONT)

    # --- cat Murchyk (sits on desk, looks out of the window) ---
    breathe = math.sin(TAU * per(t, 8))          # ~4s breathing cycle
    cx, cy = 210, 176
    # tail curls along the desk, sways
    tail_ph = math.sin(TAU * per(t, 4))
    pts = []
    for i in range(14):
        s = i / 13
        tx = cx + 22 + s * 22
        ty = cy - 2 - math.sin(s * math.pi) * (6 + 5 * tail_ph) - s * s * 10 * (1 + 0.4 * tail_ph)
        pts.append((tx, ty))
    d.line(pts, fill=CAT_DARK, width=5)
    d.line(pts, fill=CAT, width=3)
    # body
    bw = 26 + breathe * 0.8
    d.ellipse([cx - bw / 2 + 6, cy - 34 - breathe, cx + bw / 2 + 12, cy + 2], fill=CAT)
    d.ellipse([cx - bw / 2 + 4, cy - 26, cx + 8, cy + 2], fill=CAT_DARK)
    # stripes
    for sy in (-26, -20, -14):
        d.line([(cx + 14, cy + sy - breathe), (cx + 20, cy + sy + 2 - breathe)], fill=CAT_DARK)
    # front paws
    d.ellipse([cx - 6, cy - 4, cx + 4, cy + 1], fill=CAT_LIGHT)
    # head (profile, looking left to the window)
    hx, hy = cx - 4, cy - 44 - breathe * 0.6
    ear_tw = 1 if 0.62 < per(t, 1) < 0.64 else 0   # ear twitch once per loop
    d.polygon([(hx + 2, hy - 8), (hx + 6, hy - 20 + ear_tw), (hx + 11, hy - 9)], fill=CAT_DARK)
    d.polygon([(hx + 10, hy - 9), (hx + 16, hy - 19), (hx + 18, hy - 6)], fill=CAT)
    d.polygon([(hx + 4, hy - 9), (hx + 6, hy - 16 + ear_tw), (hx + 9, hy - 9)], fill=(240, 150, 150))
    d.ellipse([hx - 4, hy - 12, hx + 22, hy + 10], fill=CAT)
    d.ellipse([hx - 9, hy - 1, hx + 4, hy + 8], fill=CAT_LIGHT)  # muzzle
    d.point((hx - 9, hy + 1), fill=(200, 100, 110))               # nose
    # eye: blinks twice per loop
    p = per(t, 1)
    blink = any(abs(p - b) < 0.006 for b in (0.18, 0.71))
    if blink:
        d.line([(hx + 1, hy - 2), (hx + 5, hy - 2)], fill=(60, 40, 30))
    else:
        d.ellipse([hx + 1, hy - 5, hx + 5, hy], fill=(120, 200, 110))
        d.line([(hx + 2, hy - 5), (hx + 2, hy)], fill=(20, 20, 20))
        d.point((hx + 4, hy - 4), fill=(255, 255, 255))
    # whiskers
    d.line([(hx - 6, hy + 3), (hx - 16, hy + 1)], fill=(240, 220, 200))
    d.line([(hx - 6, hy + 5), (hx - 15, hy + 7)], fill=(240, 220, 200))
    # scarf (signature detail — same in every shot)
    sh = hy + 9
    d.rounded_rectangle([hx - 1, sh, hx + 22, sh + 6], radius=3, fill=SCARF)
    end_sway = math.sin(TAU * per(t, 4) + 1) * 1.5
    d.polygon([(hx + 16, sh + 4), (hx + 22, sh + 4), (hx + 24 + end_sway, sh + 18), (hx + 18 + end_sway, sh + 18)],
              fill=SCARF_DARK)
    for sx in range(hx + 1, hx + 22, 4):
        d.line([(sx, sh + 1), (sx, sh + 5)], fill=SCARF_DARK)

    arr = np.asarray(img).astype(np.float32) / 255.0

    # --- steam (additive, soft) ---
    steam = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(steam)
    for k in range(3):
        pts = []
        for i in range(37):
            s = i / 36
            sy = 156 - s * 40
            # wisps sway and drift upward; periodic in the loop
            sx = mx0 + 8 + math.sin(TAU * (s * 1.3 - per(t, 8)) + k * 2.1) * (1.5 + s * 6)
            pts.append((sx, sy, (1 - s) * min(s * 6, 1) * (0.55 + 0.45 * math.sin(TAU * (s - per(t, 4)) + k))))
        for (x1, y1, a1), (x2, y2, _) in zip(pts, pts[1:]):
            sd.line([(x1, y1), (x2, y2)], fill=int(max(a1, 0) * 150), width=2)
    steam = np.asarray(steam.filter(ImageFilter.GaussianBlur(1.1))).astype(np.float32) / 255.0
    arr += np.clip(steam, 0, 0.6)[..., None] * np.array([0.85, 0.85, 0.95])

    # --- lamp glow with tiny flicker (periodic) ---
    flick = 0.33 + 0.02 * math.sin(TAU * per(t, 16)) + 0.01 * math.sin(TAU * per(t, 37))
    arr += glow_base[..., None] * glow_col * flick
    arr *= vig
    return np.clip(arr, 0, 1)


# ---------- virtual camera ----------
def ease(x):
    return x * x * (3 - 2 * x)


# (start, end, (cx0, cy0, zoom0), (cx1, cy1, zoom1)) in world px
SHOTS = [
    (0, 8, (240, 120, 1.18), (238, 116, 1.28)),    # 1 wide: establishing room
    (8, 16, (212, 128, 2.40), (232, 124, 2.55)),   # 2 medium: cat at the window
    (16, 24, (300, 150, 3.60), (296, 140, 3.80)),  # 3 close: mug steam + rain
    (24, 32, (234, 116, 1.40), (240, 120, 1.18)),  # 4 wide pull-back -> loops to 1
]
XFADE = 0.5

# vertical 9:16 version: same shots, camera centres moved onto the subject
SHOTS_V = [
    (0, 8, (248, 135, 1.00), (248, 131, 1.06)),
    (8, 16, (208, 128, 2.40), (214, 124, 2.55)),
    (16, 24, (302, 150, 3.60), (302, 140, 3.80)),
    (24, 32, (248, 131, 1.10), (248, 135, 1.00)),
]
FORMATS = {"16x9": (1920, 1080, SHOTS), "9x16": (1080, 1920, SHOTS_V)}
SS = 2  # supersampling: sample 2x2 points per output pixel -> crisp, anti-aliased edges


def cam_at(shots, i, t):
    s0, s1, a, b = shots[i]
    k = ease(min(max((t - s0) / (s1 - s0), 0), 1))
    return tuple(a[j] + (b[j] - a[j]) * k for j in range(3))


def shoot(world, cam, ow, oh):
    """Point-sample the pixel-art world (no bilinear blur), then box-downsample."""
    cx, cy, z = cam
    ch = H / z
    cw = ch * ow / oh
    x0 = min(max(cx - cw / 2, 0), W - cw)
    y0 = min(max(cy - ch / 2, 0), H - ch)
    xs = np.minimum((x0 + (np.arange(ow * SS) + 0.5) * cw / (ow * SS)).astype(int), W - 1)
    ys = np.minimum((y0 + (np.arange(oh * SS) + 0.5) * ch / (oh * SS)).astype(int), H - 1)
    big = world[ys][:, xs]
    return big.reshape(oh, SS, ow, SS, 3).mean(axis=(1, 3)) * 255.0


def frame(fi, fmt="16x9"):
    ow, oh, shots = FORMATS[fmt]
    t = fi / FPS
    world = draw_world(t)
    idx = next(i for i, s in enumerate(shots) if s[0] <= t < s[1])
    out = shoot(world, cam_at(shots, idx, t), ow, oh)
    # crossfade into next shot during the last XFADE seconds of each shot
    s1 = shots[idx][1]
    if t > s1 - XFADE and idx + 1 < len(shots):
        k = (t - (s1 - XFADE)) / XFADE
        nxt = shoot(world, cam_at(shots, idx + 1, shots[idx + 1][0]), ow, oh)
        out = out * (1 - k) + nxt * k
    # light film grain: 6 patterns that cycle (cheap to encode, keeps detail clean)
    g = np.random.default_rng(fi // 4 % 6).normal(0, 1.2, (oh // 2, ow // 2, 1))
    out += np.kron(g, np.ones((2, 2, 1)))
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if idx in (0, 3):
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30 if fmt == "16x9" else 40)
        pos = (60, oh - 90) if fmt == "16x9" else (60, oh - 160)
        d.text(pos, "lofi · rain & purr", fill=(235, 220, 200), font=f)
    return img


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    a, b = int(sys.argv[2]), int(sys.argv[3])
    fmt = sys.argv[4] if len(sys.argv) > 4 else "16x9"
    for fi in range(a, min(b, N_FRAMES)):
        frame(fi, fmt).save(os.path.join(out_dir, f"f{fi:04d}.png"), compress_level=1)
