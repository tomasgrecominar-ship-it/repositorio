import numpy as np, sys, math, subprocess, os
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')
TOTAL = 29.0

# ---------------------------------------------------------------- helpers
def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def sm(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)

_disc = {}
def disc(r):
    r = max(2, int(r))
    if r in _disc: return _disc[r]
    n = 2 * r + 6
    y, x = np.mgrid[0:n, 0:n] - (n - 1) / 2
    d = np.sqrt(x * x + y * y)
    soft = max(1.2, r * 0.06)
    a = np.clip((r - d) / soft + 0.5, 0, 1)
    rim = 1 + 0.45 * smooth(r * 0.72, r * 0.97, d)
    tex = 0.82 + 0.18 * (1 - d / r)
    s = (a * rim * tex).astype(np.float32)
    _disc[r] = s
    return s

_glow = {}
def glow(r, p=2.0):
    key = (int(r), p)
    if key in _glow: return _glow[key]
    R = int(r)
    n = 2 * R * 3 + 1
    y, x = np.mgrid[0:n, 0:n] - (n - 1) / 2
    d = np.sqrt(x * x + y * y) / R
    s = (1 / (1 + d ** p)) * np.clip(1 - d / 3, 0, 1) ** 2
    _glow[key] = s.astype(np.float32)
    return _glow[key]

def blit(f, spr, cx, cy, col, k=1.0):
    h, w = spr.shape[:2]
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c: return
    sub = spr[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    if sub.ndim == 2:
        f[y0c:y1c, x0c:x1c] += sub[..., None] * (np.asarray(col, np.float32) * k)
    else:
        f[y0c:y1c, x0c:x1c] += sub * k

def over(f, rgba, x0, y0, k=1.0):
    """alpha composite premultiplied rgba (h,w,4 float) at x0,y0"""
    h, w = rgba.shape[:2]
    x0 = int(round(x0)); y0 = int(round(y0))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x0 + w), min(H, y0 + h)
    if x1c <= x0c or y1c <= y0c: return
    sub = rgba[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    a = sub[..., 3:4] * k
    f[y0c:y1c, x0c:x1c] = f[y0c:y1c, x0c:x1c] * (1 - a) + sub[..., :3] * k

GOLDS = np.array([[1.0, 0.66, 0.24], [1.0, 0.55, 0.16], [1.0, 0.76, 0.38], [0.95, 0.48, 0.14], [1.0, 0.84, 0.55]], np.float32)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def background(c_in=(0.10, 0.05, 0.02), c_out=(0.012, 0.006, 0.004), cx=0.5, cy=0.45, rad=1.0):
    d = np.sqrt(((xx / W - cx) * 1.0) ** 2 + ((yy / H - cy) * 0.56) ** 2) / (0.55 * rad)
    t = np.clip(d, 0, 1)[..., None] ** 1.3
    return (np.array(c_in, np.float32) * (1 - t) + np.array(c_out, np.float32) * t).astype(np.float32)

class Bokeh:
    def __init__(self, seed, n, rmin, rmax, inten, vel=(0, -12), area=1.3, twinkle=0.3, cols=GOLDS, ybias=None):
        g = np.random.default_rng(seed)
        self.x = g.uniform(-W * (area - 1) / 2, W * (1 + (area - 1) / 2), n)
        if ybias is None:
            self.y = g.uniform(-H * (area - 1) / 2, H * (1 + (area - 1) / 2), n)
        else:
            self.y = np.clip(g.normal(H * ybias[0], H * ybias[1], n), -200, H + 200)
        self.r = g.uniform(rmin, rmax, n) ** 1.0
        self.c = cols[g.integers(0, len(cols), n)]
        self.k = g.uniform(0.35, 1.0, n) * inten
        self.ph = g.uniform(0, 6.28, n); self.fr = g.uniform(0.3, 1.2, n)
        self.vx = vel[0] * g.uniform(0.6, 1.4, n); self.vy = vel[1] * g.uniform(0.6, 1.4, n)
        self.tw = twinkle
    def draw(self, f, t, dx=0, dy=0, rscale=1.0, k=1.0):
        for i in range(len(self.x)):
            x = (self.x[i] + self.vx[i] * t + dx); y = (self.y[i] + self.vy[i] * t + dy)
            r = self.r[i] * rscale
            kk = self.k[i] * (1 - self.tw + self.tw * (0.5 + 0.5 * math.sin(self.ph[i] + self.fr[i] * t * 2.4))) * k
            blit(f, disc(r), x, y, self.c[i], kk / (1 + r / 120))

class Dust:
    def __init__(self, seed, n, vel=(4, -25), size=(1.5, 4), inten=1.2, fall=False):
        g = np.random.default_rng(seed)
        self.x = g.uniform(0, W, n); self.y = g.uniform(0, H, n)
        self.s = g.uniform(*size, n); self.k = g.uniform(0.3, 1, n) * inten
        self.vx = vel[0] * g.uniform(-1, 1, n); self.vy = vel[1] * g.uniform(0.5, 1.5, n)
        self.ph = g.uniform(0, 6.28, n)
    def draw(self, f, t, k=1.0):
        for i in range(len(self.x)):
            x = (self.x[i] + self.vx[i] * t + 18 * math.sin(self.ph[i] + t * 0.8)) % W
            y = (self.y[i] + self.vy[i] * t) % H
            tw = 0.55 + 0.45 * math.sin(self.ph[i] * 3 + t * 3.1)
            blit(f, glow(self.s[i], 2.2), x, y, (1.0, 0.82, 0.5), self.k[i] * tw * k)

# ---------------------------------------------------------------- candle
_body = {}
def candle_body(w, h, blur=0, tone=(0.95, 0.88, 0.76)):
    key = (w, h, blur, tone)
    if key in _body: return _body[key]
    pad = 40 + blur * 3
    Wc, Hc = w + 2 * pad, h + 2 * pad
    y, x = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    u = (x - Wc / 2) / (w / 2)
    ey = w * 0.09
    top = pad + ey
    inside = (np.abs(u) <= 1) & (y >= top - ey * np.sqrt(np.clip(1 - u ** 2, 0, 1))) & (y <= pad + h)
    z = np.sqrt(np.clip(1 - u ** 2, 0, 1))
    vy = (y - top) / h
    lit = 0.10 + 0.55 * z ** 1.5 * (1 - 0.6 * vy) + 0.12 * np.clip(-u, 0, 1) * z
    sss = np.exp(-np.clip(vy, 0, None) * 5.0) * (0.7 + 0.3 * z)
    col = np.array(tone, np.float32)
    rgb = col * lit[..., None] + np.array([1.0, 0.55, 0.2], np.float32) * (sss * 0.85)[..., None]
    # top wax pool
    pool = ((u ** 2 + ((y - top) / ey) ** 2) < 0.92) & (y < top + ey)
    rgb[pool] = np.array([1.0, 0.78, 0.45]) * 0.9
    a = inside.astype(np.float32)
    rgbA = np.dstack([rgb * a[..., None], a])
    if blur:
        im = Image.fromarray(np.clip(rgbA * 255, 0, 255).astype(np.uint8), 'RGBA')
        im = im.filter(ImageFilter.GaussianBlur(blur))
        rgbA = np.asarray(im).astype(np.float32) / 255
    else:
        im = Image.fromarray(np.clip(rgbA * 255, 0, 255).astype(np.uint8), 'RGBA').filter(ImageFilter.GaussianBlur(1.0))
        rgbA = np.asarray(im).astype(np.float32) / 255
    _body[key] = (rgbA, pad, top)
    return _body[key]

def flame(f, fx, fy, fh, t, seed=0, k=1.0):
    """fx,fy = wick top; fh = flame height in px"""
    n1 = math.sin(t * 7.1 + seed) * 0.5 + math.sin(t * 12.7 + seed * 2) * 0.3 + math.sin(t * 3.3 + seed * 5) * 0.4
    n2 = math.sin(t * 9.3 + seed * 3) * 0.6 + math.sin(t * 15.1 + seed) * 0.4
    hh = fh * (1 + 0.05 * n2)
    wd = hh * 0.2
    x0, x1 = int(fx - wd * 2.5), int(fx + wd * 2.5)
    y0, y1 = int(fy - hh * 1.15), int(fy + hh * 0.15)
    x0c, x1c, y0c, y1c = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    # glow first
    gk = k * (0.85 + 0.08 * n1)
    blit(f, glow(fh * 0.55, 1.6), fx, fy - hh * 0.4, (1.0, 0.55, 0.18), 0.55 * gk)
    blit(f, glow(fh * 2.2, 1.8), fx, fy - hh * 0.4, (1.0, 0.5, 0.15), 0.22 * gk)
    if x1c <= x0c or y1c <= y0c: return
    Y, X = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float32)
    v = (fy - Y) / hh
    sway = (0.06 * n1) * hh * np.clip(v, 0, 1) ** 2
    u = (X - fx - sway) / wd
    vv = np.clip(v + 0.1, 1e-3, None)
    r = 1.25 * np.sqrt(vv) * np.clip(1.02 - v, 0, None) ** 0.95 * 1.6
    d = np.abs(u) / np.maximum(r, 1e-3)
    tipm = smooth(1.0, 0.85, v)
    body = smooth(1.0, 0.45, d) * (v > -0.1) * smooth(-0.1, 0.03, v) * tipm
    core = smooth(0.75, 0.0, d) * smooth(0.65, 0.15, v) * smooth(-0.05, 0.12, v)
    blue = smooth(1.0, 0.4, d) * smooth(0.2, 0.0, v) * (v > -0.1) * tipm
    col = (body[..., None] * np.array([1.0, 0.6, 0.18], np.float32) * 1.6
           + core[..., None] * np.array([1.0, 0.93, 0.75], np.float32) * 2.2
           + blue[..., None] * np.array([0.12, 0.18, 0.6], np.float32) * 0.8)
    f[y0c:y1c, x0c:x1c] += col * k
    # wick
    wx, wy = int(fx), int(fy)
    ww = max(1, int(fh * 0.025))
    f[max(0, wy - int(hh * 0.12)):min(H, wy + int(fh * 0.06)), max(0, wx - ww):min(W, wx + ww)] *= 0.25

def candle(f, cx, base_y, w, h, t, fh, seed=0, blur=0, k=1.0, tone=(0.95, 0.88, 0.76)):
    rgbA, pad, top = candle_body(int(w), int(h), blur, tone)
    fl = 0.9 + 0.1 * math.sin(t * 8 + seed)
    over(f, rgbA * np.array([fl, fl, fl, 1], np.float32), cx - rgbA.shape[1] / 2, base_y - h - pad, k)
    fy = base_y - h + (top - pad) * 0.2 - fh * 0.02
    if blur:
        bl = 1 + 0.06 * math.sin(t * 7 + seed)
        blit(f, glow(fh * 0.35 + blur, 2.0), cx, fy - fh * 0.45, (1.0, 0.75, 0.4), 1.2 * k * bl)
        blit(f, glow(fh * 1.8, 1.8), cx, fy - fh * 0.4, (1.0, 0.5, 0.15), 0.25 * k * bl)
    else:
        flame(f, cx, fy, fh, t, seed, k)
    return fy

# ---------------------------------------------------------------- pine
def make_pine(seed, length=1400, scale=1.0, blur=0, light=(1.0, 0.75, 0.4), bend=0.18):
    g = np.random.default_rng(seed)
    S = 2
    Wd, Hd = int(length * 1.3), int(length * 1.05)
    im = Image.new('RGBA', (Wd * S, Hd * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    def bez(p0, p1, p2, n):
        ts = np.linspace(0, 1, n)[:, None]
        return (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * p1 + ts ** 2 * p2
    def branch(p0, p2, bendv, nl, needle_len, width, depth):
        p0 = np.array(p0, float); p2 = np.array(p2, float)
        dvec = p2 - p0; L = np.linalg.norm(dvec); nrm = np.array([-dvec[1], dvec[0]]) / L
        p1 = (p0 + p2) / 2 + nrm * L * bendv
        pts = bez(p0, p1, p2, nl)
        # needles: back layer darker then front
        for layer in (0, 1):
            for i in range(nl):
                q = i / (nl - 1)
                tan = pts[min(i + 1, nl - 1)] - pts[max(i - 1, 0)]
                ang = math.atan2(tan[1], tan[0])
                ln = needle_len * (0.55 + 0.45 * math.sin(math.pi * min(1, q * 1.15 + 0.1))) * g.uniform(0.85, 1.15)
                for side in (-1, 1):
                    for _ in range(3):
                        a = ang + side * g.uniform(0.5, 1.1) - 0.3
                        e = pts[i] + np.array([math.cos(a), math.sin(a)]) * ln
                        if layer == 0:
                            c = g.uniform(0.04, 0.09); col = (int(255 * c * .55), int(255 * c * 1.45), int(255 * c * 0.8), 255)
                        else:
                            c = g.uniform(0.09, 0.2); hl = g.uniform(0, 1) < 0.18
                            col = (int(255 * c * .65), int(255 * c * 1.55), int(255 * c * 0.85), 255)
                            if hl: col = (int(255 * light[0] * .55), int(255 * light[1] * .55), int(255 * light[2] * .35), 255)
                        dr.line([tuple(pts[i] * S), tuple(e * S)], fill=col, width=int(3.2 * S * scale))
        for i in range(nl - 1):
            wv = width * (1 - i / nl) + 1
            dr.line([tuple(pts[i] * S), tuple(pts[i + 1] * S)], fill=(60, 38, 22, 255), width=int(wv * S))
        if depth > 0:
            for j in range(5):
                q = g.uniform(0.15, 0.8)
                i = int(q * (nl - 1))
                tan = pts[i + 1] - pts[i - 1]; ang = math.atan2(tan[1], tan[0]) + g.choice([-1, 1]) * g.uniform(0.5, 0.8)
                sl = L * g.uniform(0.25, 0.4)
                branch(pts[i], pts[i] + np.array([math.cos(ang), math.sin(ang)]) * sl, g.uniform(-.15, .15), int(nl * 0.45), needle_len * 0.8, width * 0.5, depth - 1)
    branch((40, Hd * 0.55), (length * 1.12, Hd * 0.45), bend, int(120 * scale), 92 * scale, 10 * scale, 1)
    im = im.resize((Wd, Hd), Image.LANCZOS)
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(im).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    return a

def place(rgba, angle_deg, s=1.0, pivot=(0, 0.5)):
    """rotate/scale an rgba array around pivot (fractions). returns array and pivot pixel"""
    im = Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), 'RGBA')
    if s != 1.0: im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
    px, py = pivot[0] * im.width, pivot[1] * im.height
    im = im.rotate(angle_deg, resample=Image.BILINEAR, center=(px, py), expand=False)
    return np.asarray(im).astype(np.float32) / 255, (px, py)

class Pine:
    def __init__(self, seed, length, scale, blur, s=1.0, bend=0.18, flip=False):
        self.a = make_pine(seed, length, scale, blur, bend=bend)
        if flip: self.a = np.ascontiguousarray(self.a[:, ::-1])
        self.flip = flip
        if s != 1.0:
            im = Image.fromarray(np.clip(self.a * 255, 0, 255).astype(np.uint8), 'RGBA')
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
            self.a = np.asarray(im).astype(np.float32) / 255
        self.cache = {}
    def draw(self, f, ax, ay, angle, k=1.0, pivot=None):
        if pivot is None: pivot = (0.97, 0.55) if self.flip else (0.03, 0.55)
        key = round(angle, 2)
        if key not in self.cache:
            if len(self.cache) > 12: self.cache.clear()
            self.cache[key] = place(self.a, angle, 1.0, pivot)
        arr, (px, py) = self.cache[key]
        over(f, arr, ax - px, ay - py, k)

# ---------------------------------------------------------------- ornament
def make_ornament(R=150):
    n = int(R * 2.6)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    cx, cy = n / 2, n / 2 + R * 0.15
    u = (x - cx) / R; v = (y - cy) / R
    d2 = u * u + v * v
    ins = smooth(1.0, 0.985, np.sqrt(d2))
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array([-0.5, -0.6, 0.62]); L /= np.linalg.norm(L)
    nd = np.clip(u * L[0] + v * L[1] + z * L[2], 0, 1)
    base = np.array([0.78, 0.52, 0.18], np.float32)
    rgb = base * (0.08 + 0.75 * nd ** 1.3)[..., None]
    fres = (1 - z) ** 3
    rgb += np.array([1.0, 0.7, 0.35]) * (fres * 0.6)[..., None]
    spec = np.exp(-(((u + 0.35) ** 2 + (v + 0.42) ** 2) / 0.012))
    rgb += np.array([1.0, 0.95, 0.85]) * (spec * 1.8)[..., None]
    g = np.random.default_rng(3)
    for _ in range(14):  # reflected bokeh
        a0 = g.uniform(0, 6.28); rr = g.uniform(0.3, 0.9)
        pu, pv = rr * math.cos(a0) * 0.9, rr * math.sin(a0) * 0.9
        sp = np.exp(-(((u - pu) ** 2 + (v - pv) ** 2) / g.uniform(0.0008, 0.003)))
        rgb += np.array([1.0, 0.8, 0.45]) * (sp * g.uniform(0.3, 0.9))[..., None]
    # cap
    capm = (np.abs(x - cx) < R * 0.17) & (y > cy - R * 1.13) & (y < cy - R * 0.93)
    capcol = np.array([0.75, 0.6, 0.3]) * (0.5 + 0.5 * np.cos((x - cx) / (R * 0.17) * 1.4))[..., None]
    rgb = np.where(capm[..., None], capcol, rgb)
    a = np.maximum(ins, capm.astype(np.float32))
    rgbA = np.dstack([rgb * a[..., None], a]).astype(np.float32)
    im = Image.fromarray(np.clip(rgbA * 255, 0, 255).astype(np.uint8), 'RGBA').filter(ImageFilter.GaussianBlur(0.8))
    return np.asarray(im).astype(np.float32) / 255, (cx, cy - R * 1.13)

# ---------------------------------------------------------------- text
def font(name, size): return ImageFont.truetype(os.path.join(FONTS, name), size)

def text_layer(items):
    """items: list of (text, fontfile, size, y_center, tracking, kind) -> premult rgba with gold gradient + glow"""
    mask = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(mask)
    for text, ff, size, yc, track, kind in items:
        fnt = font(ff, size)
        if track:
            ws = [d.textlength(ch, font=fnt) for ch in text]
            tw = sum(ws) + track * (len(text) - 1)
            x = (W - tw) / 2
            for ch, w_ in zip(text, ws):
                d.text((x, yc), ch, font=fnt, fill=255, anchor='lm'); x += w_ + track
        else:
            d.text((W / 2, yc), text, font=fnt, fill=255, anchor='mm')
    m = np.asarray(mask).astype(np.float32) / 255
    gy = (yy / H)
    grad = np.dstack([0.98 + 0 * gy, 0.80 - 0.1 * np.sin(gy * 40), 0.48 - 0.08 * np.sin(gy * 40)]).astype(np.float32)
    gl = np.asarray(mask.filter(ImageFilter.GaussianBlur(18))).astype(np.float32) / 255
    sh = np.asarray(mask.filter(ImageFilter.GaussianBlur(6))).astype(np.float32) / 255
    return m, grad, gl, sh

def draw_text(f, tl, k, dy=0):
    m, grad, gl, sh = tl
    f *= (1 - 0.5 * k * TEXTDIM)[..., None]
    if dy:
        s = int(dy)
        m = np.roll(m, s, 0); gl = np.roll(gl, s, 0); sh = np.roll(sh, s, 0)
    f *= (1 - 0.55 * sh * k)[..., None]
    f += gl[..., None] * np.array([1.0, 0.6, 0.25], np.float32) * (0.35 * k)
    a = (m * k)[..., None]
    f[:] = f * (1 - a) + grad * a

# ---------------------------------------------------------------- shots
class S1:  # intro bokeh
    def __init__(s):
        s.bg = background((0.09, 0.045, 0.02))
        s.b1 = Bokeh(1, 18, 90, 190, 0.4, (6, -18)); s.b2 = Bokeh(2, 30, 30, 80, 0.6, (10, -30)); s.d = Dust(3, 60)
    def __call__(s, t, d):
        f = s.bg.copy(); rs = 1.25 - 0.25 * sm(0, 2.5, t)
        s.b1.draw(f, t, rscale=rs); s.b2.draw(f, t, rscale=rs * 1.1); s.d.draw(f, t)
        return f, 1.0 + 0.03 * t / d

class S2:  # fairy lights
    def __init__(s):
        s.bg = background((0.06, 0.03, 0.015)); s.b = Bokeh(4, 25, 60, 150, 0.45, (-20, 0))
        g = np.random.default_rng(5)
        s.bulbs = []
        for row, (y0, amp, sag, z) in enumerate([(700, 120, 260, 1.0), (1250, 90, 200, 0.6)]):
            for i in range(14):
                q = i / 13
                x = -150 + q * 1500
                y = y0 + sag * 4 * q * (1 - q) + amp * (q - 0.5)
                s.bulbs.append((x, y, z, g.uniform(0, 6.28), g.choice(len(GOLDS))))
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-40 * t)
        pan = -70 * t
        for x, y, z, ph, ci in s.bulbs:
            tw = 0.85 + 0.15 * math.sin(ph + t * 3)
            X = x + pan * z
            blit(f, glow(70 * z, 2.0), X, y, GOLDS[ci], 0.5 * tw)
            blit(f, glow(14 * z, 2.4), X, y, (1.0, 0.9, 0.7), 2.4 * tw)
            blit(f, disc(int(22 * z)), X, y, (1.0, 0.8, 0.5), 0.5 * tw)
        return f, 1.02 + 0.03 * t / d

class S3:  # single candle close
    def __init__(s):
        s.bg = background((0.07, 0.035, 0.018), cy=0.4); s.b = Bokeh(6, 22, 70, 170, 0.34, (8, -10)); s.d = Dust(7, 30, inten=0.8)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t)
        candle(f, 560 - 15 * t, 2150, 470, 1250, t, 300, seed=1)
        s.d.draw(f, t)
        return f, 1.0 + 0.05 * t / d

class S4:  # pine branch
    def __init__(s):
        s.bg = background((0.08, 0.04, 0.02), cy=0.42); s.b = Bokeh(8, 27, 70, 180, 0.45, (5, -12))
        s.p = Pine(11, 1300, 1.25, 0); s.fg = Pine(12, 1100, 1.6, 22, bend=-0.1, flip=True); s.d = Dust(9, 50)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-25 * t)
        s.p.draw(f, -120, 1350, 24 + 1.2 * math.sin(t * 1.1), k=1.0)
        s.d.draw(f, t)
        s.fg.draw(f, 1230, 330, 22 + 1.5 * math.sin(t * 0.9 + 1), k=0.95)
        return f, 1.0 + 0.04 * t / d

class S5:  # candle macro, calm
    def __init__(s):
        s.bg = background((0.06, 0.03, 0.015), cx=0.6, cy=0.4); s.b = Bokeh(13, 19, 80, 200, 0.3, (4, -6))
        s.fg = Pine(14, 1200, 1.6, 26, bend=0.1)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-12 * t)
        candle(f, 220 + 10 * t, 1650, 300, 900, t, 160, seed=4, blur=22, k=0.75)
        candle(f, 700 - 18 * t, 2250, 520, 1150, t, 330, seed=2)
        s.fg.draw(f, -200, 1930, 7 + 1.0 * math.sin(t * 0.8), k=1.0)
        return f, 1.0 + 0.05 * t / d

class S6:  # title
    def __init__(s):
        s.bg = background((0.085, 0.042, 0.02), cy=0.42); s.b = Bokeh(15, 26, 50, 160, 0.33, (4, -14)); s.d = Dust(16, 70)
        s.top = Pine(17, 1050, 1.1, 5, bend=0.12, flip=True); s.bot = Pine(18, 1050, 1.1, 6, bend=0.12)
        s.tl = text_layer([('RESERVA TU SESIÓN', 'Montserrat-SemiBold.ttf', 40, 660, 9, 0),
                           ('Navidad', 'Cormorant-MediumItalic.ttf', 190, 800, 0, 1),
                           ('con Rosa', 'Cormorant-MediumItalic.ttf', 120, 960, 0, 1)])
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, rscale=1.0 + 0.1 * math.sin(t * 0.5)); s.d.draw(f, t)
        s.top.draw(f, 1200, 170, 14 + 1.2 * math.sin(t), k=1.0)
        s.bot.draw(f, -130, 1830, 12 + 1.2 * math.sin(t * 1.1 + 2), k=1.0)
        k = sm(0.2, 1.3, t)
        draw_text(f, s.tl, k, dy=int(30 * (1 - k)))
        return f, 1.0 + 0.035 * t / d

class S7:  # ornament on pine, falling gold flakes
    def __init__(s):
        s.bg = background((0.075, 0.038, 0.02), cy=0.5); s.b = Bokeh(19, 25, 70, 180, 0.41, (0, -8))
        s.p = Pine(20, 1400, 1.25, 0, bend=-0.08); s.orn, s.ocap = make_ornament(200)
        s.fl = Dust(21, 80, vel=(10, 70), size=(2, 6), inten=1.4)
        s.bfl = Bokeh(22, 12, 20, 45, 0.8, (5, 90))
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dy=10 * t)
        s.bfl.draw(f, t)
        ang = 9 * math.sin(t * 1.3)
        ax, ay = 560, 470
        L = 360
        ox = ax + L * math.sin(math.radians(ang)); oy = ay + L * math.cos(math.radians(ang))
        # thread
        for i in range(60):
            q = i / 59
            px, py = ax + (ox - ax) * q, ay + (oy - ay) * q
            f[int(py), int(px) - 1:int(px) + 1] = f[int(py), int(px) - 1:int(px) + 1] * 0.3 + np.array([0.9, 0.7, 0.4]) * 0.6
        arr, _ = place(s.orn, -ang, 1.0, (s.ocap[0] / s.orn.shape[1], s.ocap[1] / s.orn.shape[0]))
        over(f, arr, ox - s.ocap[0], oy - s.ocap[1])
        s.p.draw(f, -150, 520, -8 + 0.8 * math.sin(t * 1.3), k=1.0)
        s.fl.draw(f, t)
        return f, 1.0 + 0.04 * t / d

class S8:  # row of candles
    def __init__(s):
        s.bg = background((0.07, 0.035, 0.018), cy=0.45); s.b = Bokeh(23, 22, 70, 170, 0.34, (6, -6))
        s.fg = Pine(24, 1300, 1.4, 4, bend=0.05); s.d = Dust(25, 40, inten=0.9)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-30 * t)
        pan = -45 * t
        candle(f, 950 + pan * 0.6, 1450, 260, 800, t, 120, seed=7, blur=14, k=0.8)
        candle(f, 160 + pan * 0.8, 1650, 300, 700, t, 150, seed=8, blur=8, k=0.9)
        candle(f, 560 + pan, 1900, 360, 950, t, 210, seed=9)
        candle(f, 1020 + pan * 1.1, 2000, 330, 650, t, 200, seed=5)
        s.fg.draw(f, -200, 1840, -6 + math.sin(t), k=1.0)
        s.d.draw(f, t)
        return f, 1.0 + 0.04 * t / d

class S9:  # end card
    def __init__(s):
        s.bg = background((0.09, 0.045, 0.02), cy=0.42); s.b = Bokeh(26, 28, 50, 170, 0.33, (4, -16)); s.d = Dust(27, 80)
        s.top = Pine(28, 1050, 1.1, 4, bend=0.12, flip=True); s.bot = Pine(29, 1050, 1.1, 5, bend=0.12)
        s.t1 = text_layer([('Feliz', 'Cormorant-MediumItalic.ttf', 170, 700, 0, 1),
                           ('Navidad', 'Cormorant-MediumItalic.ttf', 210, 870, 0, 1)])
        s.t2 = text_layer([('RESERVA TU SESIÓN CON ROSA', 'Montserrat-SemiBold.ttf', 40, 1090, 7, 0),
                           ('Escríbele por mensaje directo', 'Cormorant-MediumItalic.ttf', 66, 1175, 0, 1)])
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t); s.d.draw(f, t)
        s.top.draw(f, 1200, 170, 14 + 1.2 * math.sin(t), k=1.0)
        s.bot.draw(f, -130, 1830, 12 + 1.2 * math.sin(t * 1.1 + 2), k=1.0)
        k1 = sm(0.2, 1.2, t); draw_text(f, s.t1, k1, dy=int(30 * (1 - k1)))
        k2 = sm(1.1, 2.0, t); draw_text(f, s.t2, k2, dy=int(20 * (1 - k2)))
        return f, 1.0 + 0.025 * t / d

# timeline: (class, start, end) ; crossfade XF centered on boundaries
CUTS = [float(x) for x in os.environ.get('CUTS', '0,2.75,3.85,4.85,7.6,11.6,15.8,20.3,24.1,29').split(',')]
SHOTS = [S1, S2, S3, S4, S5, S6, S7, S8, S9]
XF = [0.6, 0.35, 0.35, 0.5, 0.6, 0.6, 0.6, 0.6]

TEXTDIM = np.exp(-(((xx - W / 2) / 520) ** 2 + ((yy - 880) / 330) ** 2)).astype(np.float32)
vign = (1 - 0.55 * np.clip(np.sqrt(((xx / W - 0.5) * 1.1) ** 2 + ((yy / H - 0.47) * 0.8) ** 2) / 0.62, 0, 1) ** 2.2)[..., None].astype(np.float32)

def zoom(f, z):
    if z <= 1.0005: return f
    cw, ch = W / z, H / z
    x0, y0 = (W - cw) / 2, (H - ch) / 2
    im = Image.fromarray(f)
    return np.asarray(im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch)))

def finish(f, fi):
    f = f * vign
    f = 1 - np.exp(-f * 1.25)
    f = f ** 0.95
    f = f * np.array([1.0, 0.97, 0.9], np.float32)
    g = np.random.default_rng(fi).normal(0, 0.012, (H // 2, W // 2)).astype(np.float32)
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)
    f = f + g[..., None]
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

def shot_frame(inst, i, t):
    s0, s1 = CUTS[i], CUTS[i + 1]
    lead = XF[i - 1] / 2 if i > 0 else 0
    tl = t - s0 + lead
    d = (s1 - s0) + lead + (XF[i] / 2 if i < len(XF) else 0)
    f, z = inst[i](tl, d)
    return zoom(finish(f, int(t * FPS) + i * 1000), z)

def main(f0, f1, out):
    inst = {}
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        t = fi / FPS
        # which shots are active
        act = []
        for i in range(len(SHOTS)):
            a = CUTS[i] - (XF[i - 1] / 2 if i > 0 else 0)
            b = CUTS[i + 1] + (XF[i] / 2 if i < len(XF) else 0)
            if a <= t < b: act.append(i)
        for i in list(inst):
            if i not in act and i < min(act): del inst[i]
        for i in act:
            if i not in inst: inst[i] = SHOTS[i]()
        if len(act) == 1:
            img = shot_frame(inst, act[0], t)
        else:
            i, j = act[0], act[1]
            c = CUTS[j]; x = XF[i]
            w = sm(c - x / 2, c + x / 2, t)
            A = shot_frame(inst, i, t).astype(np.float32); B = shot_frame(inst, j, t).astype(np.float32)
            img = (A * (1 - w) + B * w).astype(np.uint8)
        if fi < 6: img = (img.astype(np.float32) * (fi / 6)).astype(np.uint8)
        p.stdin.write(img.tobytes())
        if fi % 30 == 0: print(out, fi, flush=True)
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
