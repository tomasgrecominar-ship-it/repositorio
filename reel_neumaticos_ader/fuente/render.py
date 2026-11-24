#!/usr/bin/env python3
"""Reel compositor: Neumáticos Ader. Usage: render.py [--stills t1,t2,...] [--out file]"""
import json, math, os, subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

B = os.path.dirname(os.path.abspath(__file__)) + '/'
W, H, FPS = 1080, 1920, 30
TOTAL = 40.0
RED = (232, 48, 40)
YEL = RED
PLATE_IN, PLATE_OUT = (196, 34, 26), (98, 10, 8)
WHITE = (255, 255, 255)
BLACK = (12, 12, 12)

# ---------------------------------------------------------------- fonts
ANTON = B + 'fonts/Anton-Regular.ttf'
MONT = B + 'fonts/Montserrat%5Bwght%5D.ttf'

_fc = {}
def font(path, size, wght=None):
    k = (path, size, wght)
    if k not in _fc:
        f = ImageFont.truetype(path, size)
        if wght:
            f.set_variation_by_axes([wght])
        _fc[k] = f
    return _fc[k]

def anton(s): return font(ANTON, s)
def mont(s, w=800): return font(MONT, s, w)

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out_cubic(p): return 1 - (1 - p) ** 3
def ease_in_out_cubic(p): return 4 * p ** 3 if p < .5 else 1 - (-2 * p + 2) ** 3 / 2
def ease_out_expo(p): return 1 if p >= 1 else 1 - 2 ** (-10 * p)
def ease_out_back(p, c=1.7):
    c3 = c + 1
    return 1 + c3 * (p - 1) ** 3 + c * (p - 1) ** 2

# ---------------------------------------------------------------- text images (RGBA numpy)
def text_width(txt, f, tracking=0):
    if not tracking:
        return f.getlength(txt)
    return sum(f.getlength(c) for c in txt) + tracking * (len(txt) - 1)

def draw_text(d, xy, txt, f, fill, tracking=0, stroke=0, stroke_fill=None):
    x, y = xy
    if not tracking:
        d.text((x, y), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        return
    for c in txt:
        d.text((x, y), c, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        x += f.getlength(c) + tracking

def with_shadow(img, radius=14, opacity=0.55, offset=(0, 6)):
    """img: PIL RGBA. returns new PIL RGBA with soft drop shadow, padded."""
    pad = radius * 2 + max(abs(offset[0]), abs(offset[1]))
    w, h = img.size
    out = Image.new('RGBA', (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    a = img.split()[3]
    sh = Image.new('L', out.size, 0)
    sh.paste(a, (pad + offset[0], pad + offset[1]))
    sh = sh.filter(ImageFilter.GaussianBlur(radius))
    sh = sh.point(lambda v: int(v * opacity))
    shadow = Image.new('RGBA', out.size, (0, 0, 0, 255))
    shadow.putalpha(sh)
    out = Image.alpha_composite(out, shadow)
    out.alpha_composite(img, (pad, pad))
    return out

def extrude(im, depth):
    a = np.array(im)[..., 3]
    out = Image.new('RGBA', (im.size[0] + depth + 2, im.size[1] + depth + 2), (0, 0, 0, 0))
    for k in range(depth, 0, -1):
        sh = Image.new('RGBA', out.size, (14, 8, 12, 255))
        m = np.zeros((out.size[1], out.size[0]), np.uint8)
        m[k:k + a.shape[0], int(k * .6):int(k * .6) + a.shape[1]] = a
        sh.putalpha(Image.fromarray(m))
        out.alpha_composite(sh)
    out.alpha_composite(im, (0, 0))
    return out

def text_img(txt, f, fill=WHITE, tracking=0, stroke=0, stroke_fill=BLACK, shadow=True, d3=0):
    asc, desc = f.getmetrics()
    w = int(text_width(txt, f, tracking)) + 2 * stroke + 4
    h = asc + desc + 2 * stroke + 4
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    draw_text(d, (stroke + 2, stroke + 2), txt, f, fill + (255,), tracking, stroke, stroke_fill)
    bb = im.getbbox()
    if bb:
        im = im.crop((0, max(0, bb[1] - 2), w, min(h, bb[3] + 2)))
    if d3:
        im = extrude(im, d3)
    if shadow:
        im = with_shadow(im)
    return im

def stack(imgs, gap=0, align='center'):
    w = max(i.size[0] for i in imgs)
    h = sum(i.size[1] for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    y = 0
    for i in imgs:
        x = (w - i.size[0]) // 2 if align == 'center' else 0
        out.alpha_composite(i, (x, y))
        y += i.size[1] + gap
    return out

def hstack(imgs, gap=0):
    w = sum(i.size[0] for i in imgs) + gap * (len(imgs) - 1)
    h = max(i.size[1] for i in imgs)
    out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    x = 0
    for i in imgs:
        out.alpha_composite(i, (x, (h - i.size[1]) // 2))
        x += i.size[0] + gap
    return out

def pill(txt, f, bg=YEL, fg=BLACK, padx=34, pady=20, radius=None, outline=None, tracking=0, shadow=True):
    tw = text_width(txt, f, tracking)
    asc, desc = f.getmetrics()
    bbox = f.getbbox('ÁBCDEFGHIJKLMNOPQRSTUVWXYZ')
    th = bbox[3] - bbox[1]
    w, h = int(tw + 2 * padx), int(th + 2 * pady)
    r = radius if radius is not None else h // 2
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if bg is not None:
        d.rounded_rectangle((0, 0, w - 1, h - 1), r, fill=bg + (255,))
    if outline is not None:
        d.rounded_rectangle((1, 1, w - 2, h - 2), r, outline=outline + (255,), width=3)
    draw_text(d, (padx, pady - bbox[1]), txt, f, fg + (255,), tracking)
    return with_shadow(im, 12, .45, (0, 5)) if shadow else im

def to_np(im):
    return np.array(im, dtype=np.uint8)

# ---------------------------------------------------------------- icons
def pin_icon(size=96, color=RED):
    s = size * 4
    im = Image.new('RGBA', (s, int(s * 1.3)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = s // 2
    d.ellipse((0, 0, s - 1, s - 1), fill=color + (255,))
    d.polygon([(s * 0.12, r * 1.35), (s * 0.88, r * 1.35), (r, s * 1.28)], fill=color + (255,))
    d.ellipse((r - s * .17, r - s * .17, r + s * .17, r + s * .17), fill=(0, 0, 0, 0))
    im = im.resize((size, int(size * 1.3)), Image.LANCZOS)
    return im

def bookmark_icon(h=64, color=YEL):
    s = 4
    w = int(h * .72)
    im = Image.new('RGBA', (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 0), (w * s, 0), (w * s, h * s), (w * s / 2, h * s * .7), (0, h * s)], fill=color + (255,))
    return im.resize((w, h), Image.LANCZOS)

# ---------------------------------------------------------------- data: VO
words = json.load(open(B + 'vo/words.json', encoding='utf-8'))
DUR = [w['dur'] for w in words]
GAPS = [0.30, 0.20, 0.30, 0.18, 0.22, 0.22, 0.32, 0.22, 0.28]
LS = [0.10]
for i in range(9):
    LS.append(round(LS[-1] + DUR[i] + GAPS[i], 3))
LE = [LS[i] + DUR[i] for i in range(10)]

def wt(line, idx):
    """absolute (start,end) of spoken word idx in line (1-based line)."""
    w = words[line - 1]['asr'][idx]
    return LS[line - 1] + w[1], LS[line - 1] + w[2]

# ---------------------------------------------------------------- shots
# (clip, src_in, crop_center_x, start, end, transition_in, zoom_dir)
CUTS = [0.0, 1.09, 2.40, 4.97, 7.30, 9.70, 11.95, 14.30, 16.70, 19.85, 23.45, 25.75, 28.00, 30.50, 33.05, TOTAL]
SPEC = [
    ('16983', 8.3, .45, 'start', +1),
    ('16983', 12.9, .50, 'punch', -1),
    ('4331', 4.6, .55, 'whip', +1),
    ('27704', 20.4, .32, 'punch', +1),
    ('33329', 4.5, .45, 'punch', -1),
    ('6393', 0.2, .62, 'whip', +1),
    ('47839', 5.5, .55, 'punch', +1),
    ('47835', 0.8, .45, 'punch', -1),
    ('45540', 0.8, .55, 'punch', +1),
    ('33327', 0.3, .62, 'punch', -1),
    ('47468', 0.5, .62, 'whip', +1),
    ('24391', 12.0, .50, 'punch', -1),
    ('17132', 5.0, .32, 'punch', +1),
    ('23568', 4.0, .55, 'punch', -1),
    ('44651', 1.0, .50, 'whip', +1),
]
PRE = 0.35
POST = 0.4

class Shot:
    def __init__(self, i, clip, src_in, cx, start, end, trans, zdir):
        self.i, self.clip, self.src_in, self.cx = i, clip, src_in, cx
        self.start, self.end, self.trans, self.zdir = start, end, trans, zdir
        self.proc = None
        self.idx = -1
        self.frame = None
        self.done = False

    def _open(self):
        path = B + 'clips/' + self.clip + '.mp4'
        ss = max(0.0, self.src_in - PRE)
        self.offset = self.src_in - ss  # seconds of preroll actually available
        dur = (self.end - self.start) + PRE + POST
        vf = (f"crop=trunc(ih*9/16/2)*2:ih:trunc((iw-ih*9/16)*{self.cx}/2)*2:0,"
              f"scale={W}:{H}:flags=lanczos,fps={FPS},"
              f"unsharp=5:5:0.45:5:5:0,eq=saturation=1.04,format=bgr24")
        cmd = ['ffmpeg', '-loglevel', 'error', '-ss', f'{ss:.3f}', '-i', path, '-t', f'{dur:.3f}',
               '-vf', vf, '-an', '-f', 'rawvideo', '-']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3 * 2)

    def get(self, t):
        if self.proc is None:
            self._open()
        target = int(round((t - self.start + self.offset) * FPS))
        target = max(0, target)
        while self.idx < target and not self.done:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                self.done = True
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.idx += 1
        return self.frame

    def close(self):
        if self.proc:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc = None

    def zoom(self, t):
        u = clamp((t - self.start) / (self.end - self.start), 0, 1.2)
        z = 1.0 + 0.055 * (u if self.zdir > 0 else (1 - u))
        dt = t - self.start
        if self.trans == 'punch' and dt >= 0:
            z *= 1 + 0.085 * (1 - ease_out_cubic(clamp(dt / 0.28)))
        if self.trans == 'start':
            z *= 1 + 0.22 * (1 - ease_out_cubic(clamp(dt / 0.35)))
        return z

SHOTS = [Shot(i, *SPEC[i][:3], CUTS[i], CUTS[i + 1], SPEC[i][3], SPEC[i][4]) for i in range(len(SPEC))]

def zoomed(img, z):
    if abs(z - 1) < 1e-4:
        return img
    M = np.array([[z, 0, W / 2 * (1 - z)], [0, z, H / 2 * (1 - z)]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

WHIP = 0.30  # seconds total

def background(t):
    # whip transition?
    for s in SHOTS[1:]:
        if s.trans == 'whip' and s.start - WHIP / 2 <= t < s.start + WHIP / 2:
            prev = SHOTS[s.i - 1]
            p = (t - (s.start - WHIP / 2)) / WHIP
            e = ease_in_out_cubic(p)
            A = zoomed(prev.get(t), prev.zoom(t))
            Bf = zoomed(s.get(t), s.zoom(max(t, s.start)))
            oy = int(round(e * H))
            canvas = np.empty_like(A)
            if oy < H:
                canvas[:H - oy] = A[oy:]
            if oy > 0:
                canvas[H - oy:] = Bf[:oy]
            # motion blur proportional to speed
            dp = 1e-3
            v = (ease_in_out_cubic(min(1, p + dp)) - ease_in_out_cubic(max(0, p - dp))) / (2 * dp)
            k = int(clamp(v * 70, 1, 181))
            if k > 2:
                canvas = cv2.blur(canvas, (1, k))
            return canvas
    for s in SHOTS:
        if s.start <= t < s.end:
            return zoomed(s.get(t), s.zoom(t))
    s = SHOTS[-1]
    return zoomed(s.get(t), s.zoom(t))

# ---------------------------------------------------------------- grade, vignette, grain
x = np.arange(256) / 255.0
curve = np.clip(0.5 + (x - 0.5) * 1.07, 0, 1)
curve = 0.78 * curve + 0.22 * (curve * curve * (3 - 2 * curve))
lut_b = np.clip(curve * 0.985 * 255, 0, 255).astype(np.uint8)
lut_g = np.clip(curve * 255, 0, 255).astype(np.uint8)
lut_r = np.clip((curve + 0.018 * np.sin(np.pi * curve)) * 255, 0, 255).astype(np.uint8)
LUT = np.dstack([lut_b, lut_g, lut_r]).reshape(256, 1, 3)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W * .75)) ** 2 + ((yy - H / 2) / (H * .72)) ** 2)
vig = 1 - 0.30 * np.clip(r - 0.35, 0, 1) ** 1.6
# legibility gradients: bottom (subtitles) and top (cards)
yb = np.clip((yy - 950) / 500, 0, 1)
bottom = 1 - 0.42 * (yb ** 1.2) * (yy < 1650) - 0.42 * (yy >= 1650)
yt = np.clip((700 - yy) / 600, 0, 1)
top = 1 - 0.30 * yt ** 1.3
MASK = (vig * bottom * top)[..., None].astype(np.float32)
rng = np.random.default_rng(7)
GRAIN = [cv2.GaussianBlur(rng.normal(0, 4.2, (H, W)).astype(np.float32), (0, 0), 0.6) for _ in range(6)]

_pr = np.sqrt(((xx - W / 2) / (W * .8)) ** 2 + ((yy - H * .42) / (H * .6)) ** 2).clip(0, 1)[..., None]
PLATE = (np.array(PLATE_IN[::-1], np.float32) * (1 - _pr) + np.array(PLATE_OUT[::-1], np.float32) * _pr)

def red_plate(fr, k):
    lum = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255.0
    tint = np.clip(PLATE * (0.55 + 0.75 * lum), 0, 255)
    return np.clip(fr.astype(np.float32) * (1 - k) + tint * k, 0, 255).astype(np.uint8)

def grade(img, fi, extra_dark=0.0):
    img = cv2.LUT(img, LUT)
    f = img.astype(np.float32) * (MASK * (1 - extra_dark))
    f += GRAIN[fi % len(GRAIN)][..., None]
    return np.clip(f, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- overlay elements
class El:
    def __init__(self, img, x, y, t0, t1, anim='pop', din=.30, aout='fade', dout=.22, anchor='c', pulse=None):
        self.im = to_np(img) if not isinstance(img, np.ndarray) else img
        self.x, self.y, self.t0, self.t1 = x, y, t0, t1
        self.anim, self.din, self.aout, self.dout, self.anchor, self.pulse = anim, din, aout, dout, anchor, pulse

    def state(self, t):
        if t < self.t0 or t > self.t1 + self.dout:
            return None
        p = clamp((t - self.t0) / self.din)
        q = clamp((t - self.t1) / self.dout) if t > self.t1 else 0
        s, a, dx, dy = 1.0, 1.0, 0.0, 0.0
        if self.anim == 'pop':
            s = 0.55 + 0.45 * ease_out_back(p, 2.2) if p < 1 else 1
            a = ease_out_cubic(clamp(p * 1.8))
        elif self.anim == 'slam':
            s = 1.6 - 0.6 * ease_out_cubic(p)
            a = ease_out_cubic(clamp(p * 2.5))
        elif self.anim == 'left':
            e = ease_out_expo(p)
            dx = -110 * (1 - e); a = e
        elif self.anim == 'rise':
            e = ease_out_expo(p)
            dy = 70 * (1 - e); a = e
        elif self.anim == 'fade':
            a = ease_out_cubic(p)
        if q > 0:
            if self.aout == 'left':
                dx -= 120 * ease_in_out_cubic(q)
            elif self.aout == 'up':
                dy -= 60 * ease_in_out_cubic(q)
            elif self.aout == 'shrink':
                s *= 1 - 0.25 * q
            a *= 1 - q
        if self.pulse and p >= 1 and q == 0:
            period, t_ref = self.pulse
            ph = ((t - t_ref) % period) / period
            s *= 1 + 0.035 * math.exp(-ph * 6)
        return s, a, dx, dy

def blend(frame, el, t):
    st = el.state(t)
    if st is None:
        return
    s, a, dx, dy = st
    if a <= 0.003:
        return
    im = el.im
    sh0 = getattr(el, 'shine', None)
    if sh0 is not None and sh0 <= t <= sh0 + 0.75:
        p = (t - sh0) / 0.75
        h_, w_ = im.shape[:2]
        xs = np.arange(w_)[None, :] + 0.35 * np.arange(h_)[:, None]
        cx = -0.2 * w_ + p * 1.6 * w_
        band = np.exp(-((xs - cx) / (0.06 * w_)) ** 2)[..., None] * 0.55
        im = im.copy().astype(np.float32)
        im[..., :3] = im[..., :3] + (255 - im[..., :3]) * band * (im[..., 3:4] / 255)
        im = np.clip(im, 0, 255).astype(np.uint8)
    if abs(s - 1) > 1e-3:
        h, w = im.shape[:2]
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = im.shape[:2]
    if el.anchor == 'c':
        x0, y0 = int(round(el.x - w / 2 + dx)), int(round(el.y - h / 2 + dy))
    else:  # left-center
        x0, y0 = int(round(el.x + dx)), int(round(el.y - h / 2 + dy))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = im[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32)
    al = sub[..., 3:4] / 255.0 * a
    rgb = sub[..., 2::-1]  # RGBA->BGR
    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (rgb * al + roi * (1 - al)).astype(np.uint8)

ELS = []        # always-on-top elements
DIM = []        # (t0,t1,amount) full-frame darkening windows

# ---- HOOK (line 1): kinetic headline synced to words
w1 = words[0]['asr']
rows = [('¿TUS CUBIERTAS', 0, 150, WHITE), ('AGUANTAN', 2, 205, WHITE), ('UN BACHE', 3, 205, WHITE), ('MÁS?', 5, 300, YEL)]
row_imgs = []
for txt, wi, size, col in rows:
    f = anton(size)
    while text_width(txt, f) > 900:
        size -= 6; f = anton(size)
    row_imgs.append(text_img(txt, f, col, shadow=True, d3=max(6, size // 22)))
gap = -18
total_h = sum(i.size[1] for i in row_imgs) + gap * 3
y = 860 - total_h / 2
for (txt, wi, size, col), im in zip(rows, row_imgs):
    t0 = 0.0 if wi == 0 else LS[0] + w1[wi][1] - 0.04
    ELS.append(El(im, W / 2, y + im.size[1] / 2, t0, CUTS[2] - 0.12, anim='slam' if wi == 5 else 'pop', din=.22, aout='up', dout=.16))
    y += im.size[1] + gap

# ---- Line 2 card: MENOS AGARRE / FRENA PEOR
c1 = text_img('MENOS AGARRE', anton(128), WHITE, d3=6)
c2 = text_img('FRENA PEOR', anton(150), RED, d3=7)
ELS.append(El(c1, W / 2, 470, wt(2, 3)[0] - .05, CUTS[4] - .2, anim='rise', din=.35, aout='up'))
ELS.append(El(c2, W / 2, 640, wt(2, 10)[0] - .05, CUTS[4] - .2, anim='slam', din=.25, aout='up'))
chip_wet = pill('CON PISO MOJADO', mont(34, 800), RED, WHITE, 26, 14, tracking=2)
ELS.append(El(chip_wet, W / 2, 330, wt(2, 8)[0] - .05, CUTS[4] - .2, anim='pop', din=.25, aout='up'))

# ---- Line 3 card: small warning chip
chip3 = pill('NO TE QUEDES A PIE', mont(36, 800), None, WHITE, 28, 16, outline=WHITE, tracking=3)
ELS.append(El(chip3, W / 2, 420, wt(3, 0)[0], CUTS[5] - .2, anim='pop', din=.3, aout='up'))

# ---- Line 4 title card: 3 SEÑALES
t3 = text_img('3', anton(520), RED, d3=16)
ts = text_img('SEÑALES', anton(190), WHITE, d3=8)
tsub = text_img('PARA CAMBIAR TUS CUBIERTAS', mont(42, 800), WHITE, tracking=4)
DIM.append((CUTS[5], CUTS[6], 0.45))
ELS.append(El(t3, W / 2, 640, wt(4, 3)[0] - .08, CUTS[6] - .12, anim='slam', din=.28, aout='shrink', dout=.14))
ELS.append(El(ts, W / 2, 1000, wt(4, 4)[0] - .06, CUTS[6] - .12, anim='rise', din=.3, aout='shrink', dout=.14))
ELS.append(El(tsub, W / 2, 1150, wt(4, 4)[0] + .12, CUTS[6] - .12, anim='fade', din=.3, aout='fade', dout=.14))

# ---- Lines 5-7: numbered cards
def num_card(n, title, sub, active):
    sq = 150
    im = Image.new('RGBA', (sq, sq), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, sq - 1, sq - 1), 26, fill=RED + (255,))
    f = anton(104)
    tw = f.getlength(n)
    bb = f.getbbox(n)
    d.text(((sq - tw) / 2, (sq - (bb[3] - bb[1])) / 2 - bb[1]), n, font=f, fill=WHITE + (255,))
    ft = anton(92)
    size = 92
    while text_width(title, ft) > 640:
        size -= 4; ft = anton(size)
    tt = text_img(title, ft, WHITE, shadow=False)
    st = text_img(sub, mont(40, 700), (238, 238, 238), shadow=False)
    # progress bars
    pb = Image.new('RGBA', (3 * 70 + 2 * 12, 10), (0, 0, 0, 0))
    dp = ImageDraw.Draw(pb)
    for k in range(3):
        col = YEL + (255,) if k <= active else (255, 255, 255, 90)
        dp.rounded_rectangle((k * 82, 0, k * 82 + 70, 9), 5, fill=col)
    right = stack([tt, st, pb], gap=12, align='left')
    card = Image.new('RGBA', (sq + 28 + right.size[0], max(sq, right.size[1])), (0, 0, 0, 0))
    card.alpha_composite(im, (0, (card.size[1] - sq) // 2))
    card.alpha_composite(right, (sq + 28, (card.size[1] - right.size[1]) // 2))
    return with_shadow(card, 16, .6, (0, 6))

cards = [
    (5, '01', 'DIBUJO GASTADO', 'Mínimo legal: 1,6 mm', 6, 8),
    (6, '02', 'CORTES O GLOBOS', 'En el flanco de la cubierta', 8, 9),
    (7, '03', 'VIBRACIONES', 'o el auto tira hacia un lado', 9, 10),
]
for k, (ln, n, title, sub, ci, ce) in enumerate(cards):
    im = num_card(n, title, sub, k)
    ELS.append(El(im, 70, 400, wt(ln, 0)[0] - .1, CUTS[ce] - .22, anim='left', din=.45, aout='left', dout=.2, anchor='l'))

# ---- Line 8: brand lockup
LOGO = Image.open(B + 'logo/logo_neumaticosader.png').convert('RGBA')
def lockup(scale=1.0, width=900):
    w = int(width * scale)
    lg = LOGO.resize((w, int(LOGO.size[1] * w / LOGO.size[0])), Image.LANCZOS)
    return with_shadow(lg, 18, .55, (0, 8))

brand = lockup()
t_ader = wt(8, 1)[0]
ELS.append(El(brand, W / 2, 560, t_ader - .08, CUTS[12] - .15, anim='slam', din=.3, aout='up', dout=.2))
chipA = pill('ASESORAMIENTO', mont(34, 800), None, WHITE, 26, 14, outline=WHITE, tracking=2)
chipB = pill('LA MEDIDA JUSTA', mont(34, 800), RED, WHITE, 26, 14, tracking=2)
ELS.append(El(chipA, W / 2, 790, wt(8, 4)[0] - .05, CUTS[12] - .15, anim='pop', din=.28, aout='up', dout=.2))
ELS.append(El(chipB, W / 2, 890, wt(8, 9)[0] - .05, CUTS[12] - .15, anim='pop', din=.28, aout='up', dout=.2))

# ---- Line 9: location + trust
chipT = pill('ATENCIÓN DE CONFIANZA', mont(34, 800), None, WHITE, 26, 14, outline=WHITE, tracking=2)
ELS.append(El(chipT, W / 2, 330, wt(9, 0)[0] - .05, CUTS[14] - .15, anim='pop', din=.28, aout='up', dout=.2))
loc_a = text_img('AV. DE MAYO 210', anton(120), WHITE, shadow=False, d3=6)
loc_b = text_img('VILLA ADELINA', mont(46, 800), WHITE, tracking=8, shadow=False)
loc = with_shadow(hstack([pin_icon(92), stack([loc_a, loc_b], gap=26, align='left')], gap=26), 16, .6, (0, 6))
ELS.append(El(loc, W / 2, 540, wt(9, 4)[0] - .08, CUTS[14] - .15, anim='rise', din=.4, aout='up', dout=.2))

# ---- End card (line 10 + hold)
T_END = CUTS[14]
PLATE_T = T_END + 0.02
brand2 = lockup(1.0, 1000)
LOGO_EL = El(brand2, W / 2, 730, T_END + .30, TOTAL + 5, anim='slam', din=.34)
LOGO_EL.shine = T_END + 0.85
ELS.append(LOGO_EL)
addr = text_img('AV. DE MAYO 210  ·  VILLA ADELINA', mont(38, 700), WHITE, tracking=2)
ELS.append(El(addr, W / 2, 880, T_END + .60, TOTAL + 5, anim='fade', din=.4))
cta = pill('ESCRIBINOS POR DM', mont(52, 900), WHITE, RED, 52, 30, tracking=2)
t_cta = wt(10, 0)[0]
ELS.append(El(cta, W / 2, 1040, t_cta - .05, TOTAL + 5, anim='pop', din=.32, pulse=(60 / 112.35 * 2, t_cta)))
handle = text_img('@neumaticosader_avdemayo', mont(44, 700), WHITE)
ELS.append(El(handle, W / 2, 1170, t_cta + .25, TOTAL + 5, anim='rise', din=.4))
save = with_shadow(hstack([bookmark_icon(70, WHITE), text_img('GUARDÁ ESTE VIDEO', anton(96), WHITE, shadow=False, d3=6)], gap=24), 16, .6, (0, 6))
T_SAVE = wt(10, 5)[0]
ELS.append(El(save, W / 2, 1380, T_SAVE, TOTAL + 5, anim='pop', din=.3))

# ---------------------------------------------------------------- subtitles
MERGE = {5: {(8, 12): '1,6 MM'}, 9: {(7, 9): '210'}}
SUB_LINES = [2, 3, 5, 6, 7, 8, 9]
SUB_Y = 1265
SUB_F = mont(66, 900)

def line_tokens(ln):
    raw = words[ln - 1]['text'].split()
    asr = words[ln - 1]['asr']
    assert len(raw) == len(asr), (ln, raw, asr)
    toks = []
    i = 0
    merges = MERGE.get(ln, {})
    while i < len(raw):
        m = next(((a, b, txt) for (a, b), txt in merges.items() if a == i), None)
        if m:
            a, b, txt = m
            toks.append([txt, LS[ln - 1] + asr[a][1], LS[ln - 1] + asr[b - 1][2], raw[b - 1][-1] in ',.:?'])
            i = b
            continue
        w = raw[i]
        punct = w[-1] in ',.:?'
        disp = w.rstrip(',.:').upper()
        toks.append([disp, LS[ln - 1] + asr[i][1], LS[ln - 1] + asr[i][2], punct])
        i += 1
    return toks

SUBS = []  # (t0, t1, [tokens], line)
for ln in SUB_LINES:
    toks = line_tokens(ln)
    chunks, cur = [], []
    for tk in toks:
        cand = cur + [tk]
        chars = len(' '.join(x[0] for x in cand))
        if cur and (chars > 17 or len(cand) > 3):
            chunks.append(cur); cur = [tk]
        else:
            cur = cand
        if tk[3] and cur:
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    # merge single short trailing chunk into previous if it fits
    for ci, ch in enumerate(chunks):
        t0 = ch[0][1] - 0.06
        t1 = chunks[ci + 1][0][1] - 0.06 if ci + 1 < len(chunks) else LE[ln - 1] + 0.12
        if ln == 10:
            t1 = min(t1, T_SAVE - 0.02)
        SUBS.append((t0, t1, ch, ln))

_sub_cache = {}
def sub_img(ci, active):
    k = (ci, active)
    if k in _sub_cache:
        return _sub_cache[k]
    t0, t1, ch, ln = SUBS[ci]
    f = SUB_F
    stroke = 7
    words_ = [x[0] for x in ch]
    space = f.getlength(' ')
    ws = [f.getlength(w) for w in words_]
    tot = sum(ws) + space * (len(ws) - 1)
    scale_f = f
    if tot > 880:
        sz = int(66 * 880 / tot)
        scale_f = mont(sz, 900)
        ws = [scale_f.getlength(w) for w in words_]
        space = scale_f.getlength(' ')
        tot = sum(ws) + space * (len(ws) - 1)
    asc, desc = scale_f.getmetrics()
    im = Image.new('RGBA', (int(tot) + 2 * stroke + 32, asc + desc + 2 * stroke + 24), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = stroke + 16
    capb = scale_f.getbbox('ÁBCDEFGHIJKLMNOPQRSTUVWXYZ')
    for j, w in enumerate(words_):
        if j == active:
            d.rounded_rectangle((x - 12, stroke + 12 + capb[1] - 12, x + ws[j] + 12, stroke + 12 + capb[3] + 12), 12, fill=RED + (255,))
            d.text((x, stroke + 12), w, font=scale_f, fill=WHITE + (255,))
        else:
            d.text((x, stroke + 12), w, font=scale_f, fill=WHITE + (255,), stroke_width=stroke, stroke_fill=(10, 10, 10, 255))
        x += ws[j] + space
    im = with_shadow(im, 12, .5, (0, 5))
    arr = to_np(im)
    _sub_cache[k] = arr
    return arr

def draw_subs(frame, t):
    for ci, (t0, t1, ch, ln) in enumerate(SUBS):
        if t0 <= t < t1:
            active = 0
            for j, tk in enumerate(ch):
                if t >= tk[1] - 0.05:
                    active = j
            el = El(sub_img(ci, active), W / 2, SUB_Y, t0, t1, anim='none', dout=0.0001)
            p = clamp((t - t0) / .12)
            el.state = (lambda tt, p=p: (0.9 + 0.1 * ease_out_back(p, 2.0), ease_out_cubic(clamp(p * 2)), 0, 0))
            blend(frame, el, t)

# ---------------------------------------------------------------- main render
def render_frame(fi):
    t = fi / FPS
    bg = background(t)
    dark = 0.0
    for a, b, amt in DIM:
        if a - .2 <= t <= b:
            dark = max(dark, amt * ease_out_cubic(clamp((t - a + .2) / .35)))
    fr = grade(bg, fi, dark)
    if t >= PLATE_T - 0.1:
        fr = red_plate(fr, 0.9 * ease_out_cubic(clamp((t - PLATE_T + 0.1) / 0.4)))
    for el in ELS:
        blend(fr, el, t)
    draw_subs(fr, t)
    # close finished readers
    for s in SHOTS:
        if s.proc and t > s.end + WHIP:
            s.close()
    return fr

def main():
    args = sys.argv[1:]
    if args and args[0] == '--stills':
        ts = [float(v) for v in args[1].split(',')]
        os.makedirs(B + 'stills', exist_ok=True)
        for tt in ts:
            for s in SHOTS:
                s.close(); s.idx = -1; s.done = False
            fr = render_frame(int(round(tt * FPS)))
            cv2.imwrite(B + f'stills/s_{tt:05.2f}.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    out = args[1] if len(args) > 1 and args[0] == '--out' else B + 'work/video_only.mp4'
    n = int(round(TOTAL * FPS))
    enc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
                            '-profile:v', 'high', '-level', '4.2',
                            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                            out], stdin=subprocess.PIPE)
    for fi in range(n):
        fr = render_frame(fi)
        enc.stdin.write(np.ascontiguousarray(fr).tobytes())
        if fi % 60 == 0:
            print(f'frame {fi}/{n}', flush=True)
    enc.stdin.close(); enc.wait()
    json.dump({'LS': LS, 'LE': LE, 'CUTS': CUTS, 'T_SAVE': T_SAVE,
               'shots': [(s.clip, s.start, s.end, s.trans) for s in SHOTS],
               'subs': [(a, b, [x[0] for x in c]) for a, b, c, _ in SUBS]},
              open(B + 'work/timeline.json', 'w'), indent=1, ensure_ascii=False)
    print('done', out)

if __name__ == '__main__':
    main()
