"""Romantic Christmas reel — shots built on the helpers in render.py"""
import numpy as np, sys, math, subprocess, os
from PIL import Image, ImageDraw, ImageFilter
import render as R
from render import W, H, FPS, smooth, sm, disc, glow, blit, over, Pine, place, text_layer, draw_text, candle, xx, yy

ROSEGOLD = np.array([[1.0, 0.55, 0.42], [1.0, 0.42, 0.45], [1.0, 0.70, 0.36], [0.95, 0.28, 0.32], [1.0, 0.78, 0.6]], np.float32)

def bg(c_in=(0.12, 0.025, 0.04), c_out=(0.014, 0.004, 0.007), cx=0.5, cy=0.45):
    return R.background(c_in, c_out, cx, cy)

def to_np(im):
    a = np.asarray(im).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    return a

def to_im(a):
    b = a.copy()
    al = np.maximum(b[..., 3:4], 1e-4)
    b[..., :3] = np.where(b[..., 3:4] > 0, b[..., :3] / al, 0)
    return Image.fromarray(np.clip(b * 255, 0, 255).astype(np.uint8), 'RGBA')

# ------------------------------------------------------------ heart bokeh
_heart = {}
def heart(r):
    r = max(4, int(r))
    if r in _heart: return _heart[r]
    n = 2 * r + 8
    ss = 3
    y, x = np.mgrid[0:n * ss, 0:n * ss].astype(np.float32) / ss - (n - 1) / 2
    X = x / r * 1.25; Y = -(y / r * 1.25) + 0.15
    m = (((X * X + Y * Y - 1) ** 3 - X * X * Y ** 3) <= 0).astype(np.float32)
    m = m.reshape(n, ss, n, ss).mean((1, 3))
    im = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(0.8, r * 0.04)))
    m = np.asarray(im).astype(np.float32) / 255
    inner = np.asarray(im.filter(ImageFilter.GaussianBlur(max(1, r * 0.18)))).astype(np.float32) / 255
    s = m * (0.8 + 0.6 * np.clip(1 - inner, 0, 1))
    _heart[r] = s.astype(np.float32)
    return _heart[r]

class HBokeh:
    """mix of circles and hearts"""
    def __init__(self, seed, n, rmin, rmax, inten, vel=(0, -12), hearts=0.5, cols=ROSEGOLD, twinkle=0.3, area=1.3):
        g = np.random.default_rng(seed)
        self.x = g.uniform(-W * (area - 1) / 2, W * (1 + (area - 1) / 2), n)
        self.y = g.uniform(-H * (area - 1) / 2, H * (1 + (area - 1) / 2), n)
        self.r = g.uniform(rmin, rmax, n); self.c = cols[g.integers(0, len(cols), n)]
        self.k = g.uniform(0.35, 1.0, n) * inten; self.h = g.uniform(0, 1, n) < hearts
        self.ph = g.uniform(0, 6.28, n); self.fr = g.uniform(0.3, 1.2, n)
        self.vx = vel[0] * g.uniform(0.6, 1.4, n); self.vy = vel[1] * g.uniform(0.6, 1.4, n); self.tw = twinkle
    def draw(self, f, t, dx=0, dy=0, rscale=1.0, k=1.0):
        for i in range(len(self.x)):
            x = self.x[i] + self.vx[i] * t + dx; y = self.y[i] + self.vy[i] * t + dy
            r = self.r[i] * rscale
            kk = self.k[i] * (1 - self.tw + self.tw * (0.5 + 0.5 * math.sin(self.ph[i] + self.fr[i] * t * 2.4))) * k
            spr = heart(r) if self.h[i] else disc(r)
            blit(f, spr, x, y, self.c[i], kk / (1 + r / 120))

class Dust(R.Dust):
    def draw(self, f, t, k=1.0, col=(1.0, 0.78, 0.6)):
        for i in range(len(self.x)):
            x = (self.x[i] + self.vx[i] * t + 18 * math.sin(self.ph[i] + t * 0.8)) % W
            y = (self.y[i] + self.vy[i] * t) % H
            tw = 0.55 + 0.45 * math.sin(self.ph[i] * 3 + t * 3.1)
            blit(f, glow(self.s[i], 2.2), x, y, col, self.k[i] * tw * k)

# ------------------------------------------------------------ petals
def make_petal(size, seed, blur=0):
    g = np.random.default_rng(seed)
    n = int(size * 1.4)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    u = (x - n / 2) / (size / 2); v = 1 - (y - (n - size) / 2) / size  # v: 0 base -> 1 tip
    wv = np.clip(np.sin(np.pi * np.clip(v, 0, 1) ** 0.65), 0, 1) * (1 - 0.18 * smooth(0.85, 1.0, v) * smooth(0.25, 0.0, np.abs(u)))
    m = smooth(1.0, 0.9, np.abs(u) / np.maximum(wv, 1e-3)) * (v > 0) * (v < 1)
    base = np.array([0.62, 0.05, 0.10]) * g.uniform(0.85, 1.15)
    shade = 0.55 + 0.45 * np.cos(np.clip(u, -1, 1) * 1.3) + 0.25 * v
    col = base * shade[..., None] + np.array([0.35, 0.12, 0.12]) * (smooth(0.4, 0.0, v) * 0.6)[..., None]
    vein = 1 - 0.12 * (np.abs(np.sin(u * 9)) < 0.15)
    col *= vein[..., None]
    a = np.dstack([col * m[..., None], m]).astype(np.float32)
    im = to_im(a)
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur))
    else: im = im.filter(ImageFilter.GaussianBlur(0.6))
    return im

class Petals:
    def __init__(self, seed, n, size=(50, 90), vel=(15, 70), blurfrac=0.35, inten=1.0):
        g = np.random.default_rng(seed)
        self.items = []
        for i in range(n):
            bl = g.uniform() < blurfrac
            sz = g.uniform(*size) * (1.8 if bl else 1.0)
            im = make_petal(int(sz), seed * 100 + i, blur=int(sz * 0.12) if bl else 0)
            self.items.append(dict(im=im, x=g.uniform(-50, W + 50), y=g.uniform(-200, H), vx=vel[0] * g.uniform(-1, 1),
                                   vy=vel[1] * g.uniform(0.6, 1.4) * (1.4 if bl else 1), rot=g.uniform(0, 360), vr=g.uniform(-40, 40),
                                   fp=g.uniform(0, 6.28), fs=g.uniform(0.8, 2.0), k=inten * (0.8 if bl else 1.0)))
    def draw(self, f, t, k=1.0):
        for p in self.items:
            y = (p['y'] + p['vy'] * t) % (H + 300) - 150
            x = p['x'] + p['vx'] * t + 40 * math.sin(p['fp'] + t * 0.9)
            flip = math.cos(p['fp'] + t * p['fs'])
            sx = max(0.18, abs(flip))
            im = p['im']
            im2 = im.resize((max(2, int(im.width * sx)), im.height), Image.BILINEAR).rotate(p['rot'] + p['vr'] * t, Image.BILINEAR, expand=True)
            a = to_np(im2)
            a[..., :3] *= (0.75 + 0.25 * (flip > 0))
            over(f, a, x - a.shape[1] / 2, y - a.shape[0] / 2, k * p['k'])

# ------------------------------------------------------------ rose bloom (top / 3-4 view)
def make_rose(R0=360, seed=1, tilt=0.78):
    g = np.random.default_rng(seed)
    n = int(R0 * 2.6)
    ss = 1
    cx, cy = n / 2, n / 2
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    X = (x - cx); Y = (y - cy) / tilt
    rr = np.sqrt(X * X + Y * Y); th = np.arctan2(Y, X)
    out = np.zeros((n, n, 4), np.float32)
    npet = 46
    for i in range(npet):
        q = i / (npet - 1)                          # 0 outer -> 1 inner
        rad = R0 * (1 - q) ** 0.85 * 0.98 + R0 * 0.03
        ang = i * 2.39996 + g.normal(0, 0.1)
        wid = (1.25 - 0.45 * q)                     # angular half-width
        inner = rad * (0.15 + 0.35 * q)
        d_ang = np.angle(np.exp(1j * (th - ang)))
        rel = (rr - inner) / max(rad - inner, 1)
        edge = 1 - (d_ang / wid) ** 2                # 1 at centerline
        shape = (rel < np.sqrt(np.clip(edge, 0, 1)) * (1 + 0.06 * np.cos(d_ang * 5))) & (rr > inner * 0.3) & (np.abs(d_ang) < wid)
        if not shape.any(): continue
        m = shape.astype(np.float32)
        cup = np.clip(rel, 0, 1)
        light = 0.35 + 0.65 * cup ** 1.4 * (0.75 + 0.25 * np.cos(d_ang * 1.5))
        light *= 0.85 + 0.25 * np.cos(th - ang - 2.4) * (1 - q)  # key light from top-left
        base = np.array([0.66, 0.05, 0.11]) * (0.85 + 0.3 * q)
        col = base * light[..., None]
        col += np.array([0.55, 0.22, 0.22]) * (smooth(0.88, 1.0, cup) * 0.35 * (1 - q))[..., None]  # rim sheen
        # shadow onto what is below
        sh = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10 + 18 * (1 - q)))).astype(np.float32) / 255
        out[..., :3] *= (1 - 0.65 * sh * (1 - m))[..., None]
        out[..., :3] = out[..., :3] * (1 - m[..., None]) + col * m[..., None]
        out[..., 3] = np.maximum(out[..., 3], m)
    out[..., :3] *= (0.25 + 0.75 * smooth(0, R0 * 0.35, rr))[..., None]  # dark heart of the bloom
    out[..., :3] *= out[..., 3:4]
    im = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), 'RGBA').filter(ImageFilter.GaussianBlur(1.0))
    return to_np(im)

# ------------------------------------------------------------ heart ornament
def make_heart_orn(Rr=180):
    n = int(Rr * 2.8)
    hm = heart(int(Rr))
    pad = (n - hm.shape[0]) // 2
    m = np.zeros((n, n), np.float32); m[pad:pad + hm.shape[0], pad:pad + hm.shape[1]] = np.clip(hm / hm.max() * 1.6, 0, 1)
    mi = Image.fromarray((m * 255).astype(np.uint8))
    h1 = np.asarray(mi.filter(ImageFilter.GaussianBlur(Rr * 0.22))).astype(np.float32) / 255
    h = np.sqrt(np.clip(h1, 0, 1))
    gy, gx = np.gradient(h * Rr * 0.9)
    N = np.dstack([-gx, -gy, np.ones_like(h)]); N /= np.linalg.norm(N, axis=2, keepdims=True)
    L = np.array([-0.45, -0.6, 0.66]); L /= np.linalg.norm(L)
    nd = np.clip((N * L).sum(2), 0, 1)
    Hv = L + np.array([0, 0, 1]); Hv /= np.linalg.norm(Hv)
    spec = np.clip((N * Hv).sum(2), 0, 1) ** 60
    base = np.array([0.62, 0.03, 0.08])
    col = base * (0.1 + 0.9 * nd ** 1.5)[..., None] + np.array([1, 0.92, 0.88]) * (spec * 1.6)[..., None]
    fres = (1 - N[..., 2]) ** 2
    col += np.array([0.9, 0.45, 0.3]) * (fres * 0.5)[..., None]
    g = np.random.default_rng(9)
    for _ in range(40):  # gold glitter
        px, py = g.integers(0, n, 2)
        if m[py, px] > 0.9:
            col[max(0, py - 2):py + 2, max(0, px - 2):px + 2] += np.array([1.0, 0.8, 0.45]) * g.uniform(0.3, 0.9)
    a = smooth(0.25, 0.6, m)
    # gold cap at the top notch
    cy_top = pad + int(hm.shape[0] * 0.22); cxm = n // 2
    cap = (np.abs(xx[:n, :n] - cxm) < Rr * 0.12) & (yy[:n, :n] > cy_top - Rr * 0.22) & (yy[:n, :n] < cy_top + Rr * 0.02)
    capc = np.array([0.85, 0.65, 0.32]) * (0.45 + 0.55 * np.cos((xx[:n, :n] - cxm) / (Rr * 0.12) * 1.4))[..., None]
    col = np.where(cap[..., None], capc, col); a = np.maximum(a, cap.astype(np.float32))
    rgba = np.dstack([col * a[..., None], a]).astype(np.float32)
    return rgba, (cxm, cy_top - Rr * 0.22)

# ------------------------------------------------------------ gift box
def make_gift(w=560, h=430, d=200):
    S = 2
    Wc, Hc = int((w + d + 200) * S), int((h + d * 0.6 + 520) * S)
    im = Image.new('RGBA', (Wc, Hc), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    ox, oy = 100 * S, int((d * 0.55 + 420) * S)
    dx, dy = d * S, -d * 0.55 * S
    W2, H2 = w * S, h * S
    paper = np.array([0.95, 0.82, 0.62])
    def c(v, k): return tuple(int(255 * min(1, x * k)) for x in v) + (255,)
    front = [(ox, oy), (ox + W2, oy), (ox + W2, oy + H2), (ox, oy + H2)]
    side = [(ox + W2, oy), (ox + W2 + dx, oy + dy), (ox + W2 + dx, oy + H2 + dy), (ox + W2, oy + H2)]
    top = [(ox, oy), (ox + dx, oy + dy), (ox + W2 + dx, oy + dy), (ox + W2, oy)]
    dr.polygon(front, fill=c(paper, 0.78)); dr.polygon(side, fill=c(paper, 0.48)); dr.polygon(top, fill=c(paper, 0.98))
    red = np.array([0.72, 0.05, 0.1]); rw = 46 * S
    mx = ox + W2 / 2
    dr.polygon([(mx - rw, oy), (mx + rw, oy), (mx + rw, oy + H2), (mx - rw, oy + H2)], fill=c(red, 0.85))
    my = oy + H2 / 2
    dr.polygon([(ox, my - rw), (ox + W2, my - rw), (ox + W2, my + rw), (ox, my + rw)], fill=c(red, 0.85))
    dr.polygon([(ox + W2, my - rw), (ox + W2 + dx, my - rw + dy), (ox + W2 + dx, my + rw + dy), (ox + W2, my + rw)], fill=c(red, 0.5))
    dr.polygon([(mx - rw, oy), (mx - rw + dx, oy + dy), (mx + rw + dx, oy + dy), (mx + rw, oy)], fill=c(red, 1.05))
    tm = (ox + W2 / 2 + dx / 2, oy + dy / 2)
    dr.polygon([(ox + dx / 2, tm[1] - rw * 0.5), (ox + dx / 2 + W2, tm[1] - rw * 0.5), (ox + dx / 2 + W2, tm[1] + rw * 0.5), (ox + dx / 2, tm[1] + rw * 0.5)], fill=c(red, 1.0))
    # bow loops
    for sgn in (-1, 1):
        lay = Image.new('RGBA', im.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        bx, by = tm[0] + sgn * 120 * S, tm[1] - 85 * S
        ld.ellipse([bx - 140 * S, by - 80 * S, bx + 140 * S, by + 80 * S], fill=c(red, 0.95))
        ld.ellipse([bx - 70 * S + sgn * 25 * S, by - 28 * S, bx + 60 * S + sgn * 25 * S, by + 34 * S], fill=c(red, 0.55))
        ld.arc([bx - 132 * S, by - 74 * S, bx + 132 * S, by + 74 * S], 200, 320, fill=c(np.array([1.0, 0.55, 0.55]), 0.9), width=int(10 * S))
        lay = lay.rotate(sgn * 20, center=(tm[0], tm[1]), resample=Image.BICUBIC)
        im.alpha_composite(lay)
    # tails
    for sgn in (-1, 1):
        dr.polygon([(tm[0], tm[1]), (tm[0] + sgn * 150 * S, tm[1] + 160 * S), (tm[0] + sgn * 95 * S, tm[1] + 175 * S), (tm[0] - sgn * 10 * S, tm[1] + 20 * S)], fill=c(red, 0.7))
    dr.ellipse([tm[0] - 45 * S, tm[1] - 60 * S, tm[0] + 45 * S, tm[1] + 10 * S], fill=c(red, 1.05))
    im = im.resize((Wc // S, Hc // S), Image.LANCZOS)
    a = to_np(im)
    # soft light gradient + sheen
    gy = np.linspace(1.15, 0.75, a.shape[0])[:, None, None]
    gx = np.linspace(1.1, 0.85, a.shape[1])[None, :, None]
    a[..., :3] *= gy * gx
    im = to_im(a).filter(ImageFilter.GaussianBlur(1.2))
    im = im.crop(im.getbbox())
    im = im.resize((int(im.width * 1.25), int(im.height * 1.25)), Image.LANCZOS)
    return to_np(im), None

# ------------------------------------------------------------ shots
class S1:  # hearts bokeh intro
    def __init__(s):
        s.bg = bg(); s.b1 = HBokeh(1, 18, 90, 180, 0.5, (5, -16), hearts=0.5); s.b2 = HBokeh(2, 30, 30, 75, 0.7, (8, -26), hearts=0.6)
        s.p = Petals(3, 10, vel=(12, 60)); s.d = Dust(4, 50)
    def __call__(s, t, d):
        f = s.bg.copy(); rs = 1.2 - 0.2 * sm(0, 2.8, t)
        s.b1.draw(f, t, rscale=rs); s.b2.draw(f, t, rscale=rs); s.d.draw(f, t); s.p.draw(f, t + 2)
        return f, 1.0 + 0.03 * t / d

class S2:  # rose bloom macro
    def __init__(s):
        s.bg = bg((0.10, 0.02, 0.03), cy=0.5); s.b = HBokeh(5, 18, 70, 160, 0.35, (0, -8), hearts=0.4)
        s.rose = make_rose(430, 2); s.d = Dust(6, 30, inten=0.8)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t)
        blit(f, glow(420, 2.0), 540, 980, (1.0, 0.45, 0.35), 0.25)
        arr, (px, py) = place(s.rose, 6 * t, 1.0, (0.5, 0.5))
        over(f, arr, 540 - px, 990 - py)
        s.d.draw(f, t)
        return f, 1.02 + 0.06 * t / d

class S3:  # two candles together ("miradas cómplices")
    def __init__(s):
        s.bg = bg((0.10, 0.03, 0.03), cy=0.42); s.b = HBokeh(7, 20, 60, 150, 0.35, (4, -8), hearts=0.5); s.p = Petals(8, 6, vel=(8, 45), blurfrac=0.6)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-15 * t)
        candle(f, 360 + 8 * t, 1950, 330, 900, t, 230, seed=11, tone=(0.96, 0.87, 0.82))
        candle(f, 730 - 8 * t, 2080, 330, 820, t, 230, seed=11.4, tone=(0.96, 0.87, 0.82))
        s.p.draw(f, t)
        return f, 1.0 + 0.05 * t / d

class S4:  # candles + petals on surface
    def __init__(s):
        s.bg = bg((0.10, 0.03, 0.025), cy=0.45); s.b = HBokeh(9, 22, 70, 170, 0.35, (6, -6), hearts=0.45)
        s.p = Petals(10, 12, vel=(10, 55)); s.fg = Pine(24, 1300, 1.4, 4, bend=0.05)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-30 * t)
        pan = -45 * t
        candle(f, 950 + pan * 0.6, 1450, 260, 800, t, 120, seed=7, blur=14, k=0.8)
        candle(f, 160 + pan * 0.8, 1650, 300, 700, t, 150, seed=8, blur=8, k=0.9)
        candle(f, 560 + pan, 1900, 360, 950, t, 210, seed=9)
        candle(f, 1020 + pan * 1.1, 2000, 330, 650, t, 200, seed=5)
        s.fg.draw(f, -200, 1840, -6 + math.sin(t), k=1.0)
        s.p.draw(f, t)
        return f, 1.0 + 0.04 * t / d

class S5:  # two heart ornaments together
    def __init__(s):
        s.bg = bg((0.11, 0.03, 0.035), cy=0.5); s.b = HBokeh(11, 24, 60, 170, 0.4, (3, -10), hearts=0.5)
        s.p = Pine(30, 1450, 1.25, 0, bend=-0.06); s.o1, s.c1 = make_heart_orn(150); s.o2, s.c2 = make_heart_orn(125)
        s.pt = Petals(12, 10, vel=(12, 55)); s.d = Dust(13, 60)
    def hang(s, f, orn, cap, ax, ay, L, ang):
        ox = ax + L * math.sin(math.radians(ang)); oy = ay + L * math.cos(math.radians(ang))
        for i in range(90):
            q = i / 89; px, py = ax + (ox - ax) * q, ay + (oy - ay) * q
            f[int(py), int(px) - 1:int(px) + 1] = f[int(py), int(px) - 1:int(px) + 1] * 0.3 + np.array([0.9, 0.7, 0.4]) * 0.6
        arr, _ = place(orn, -ang, 1.0, (cap[0] / orn.shape[1], cap[1] / orn.shape[0]))
        over(f, arr, ox - cap[0], oy - cap[1])
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-10 * t)
        blit(f, glow(380, 2.0), 540, 1050, (1.0, 0.5, 0.4), 0.2)
        s.hang(f, s.o1, s.c1, 400, 560, 430, 5 * math.sin(t * 1.2))
        s.hang(f, s.o2, s.c2, 700, 520, 560, -5 * math.sin(t * 1.2 + 0.4))
        s.p.draw(f, -160, 600, -6 + 0.8 * math.sin(t * 1.2), k=1.0)
        s.d.draw(f, t); s.pt.draw(f, t)
        return f, 1.0 + 0.045 * t / d

class S6:  # title
    def __init__(s):
        s.bg = bg((0.12, 0.03, 0.04), cy=0.44); s.b = HBokeh(15, 24, 50, 150, 0.3, (4, -14), hearts=0.6); s.d = Dust(16, 60)
        s.rose = make_rose(260, 5); s.rose2 = make_rose(220, 6)
        s.top = Pine(17, 1050, 1.1, 5, bend=0.12, flip=True)
        s.p = Petals(18, 9, vel=(10, 50), blurfrac=0.5)
        s.tl = text_layer([('RESERVA TU SESIÓN', 'Montserrat-SemiBold.ttf', 40, 660, 9, 0),
                           ('Navidad', 'Cormorant-MediumItalic.ttf', 190, 800, 0, 1),
                           ('con Rosa', 'Cormorant-MediumItalic.ttf', 120, 960, 0, 1)])
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t); s.d.draw(f, t)
        s.top.draw(f, 1200, 170, 14 + 1.2 * math.sin(t), k=1.0)
        over(f, s.rose, 130 - s.rose.shape[1] / 2, 1830 - s.rose.shape[0] / 2)
        over(f, s.rose2, 400 - s.rose2.shape[1] / 2, 1960 - s.rose2.shape[0] / 2)
        k = sm(0.2, 1.3, t); draw_text(f, s.tl, k, dy=int(30 * (1 - k)))
        s.p.draw(f, t)
        return f, 1.0 + 0.035 * t / d

class S7:  # petals flying over golden lights
    def __init__(s):
        s.bg = bg((0.10, 0.025, 0.035), cy=0.5); s.b = HBokeh(19, 22, 70, 170, 0.38, (-30, -4), hearts=0.5)
        g = np.random.default_rng(31); s.bulbs = []
        for (y0, sag, z) in [(560, 240, 1.0), (1320, 180, 0.65)]:
            for i in range(14):
                q = i / 13; s.bulbs.append((-150 + q * 1500, y0 + sag * 4 * q * (1 - q), z, g.uniform(0, 6.28)))
        s.pt = Petals(22, 22, size=(55, 100), vel=(-10, 40), blurfrac=0.45); s.wind = (-260, 120)
        s.fl = Dust(21, 60, vel=(10, 40), size=(2, 5), inten=1.2)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t)
        for x, y, z, ph in s.bulbs:
            X = x - 50 * t * z; tw = 0.85 + 0.15 * math.sin(ph + t * 3)
            blit(f, glow(70 * z, 2.0), X, y, (1.0, 0.6, 0.35), 0.5 * tw)
            blit(f, glow(14 * z, 2.4), X, y, (1.0, 0.88, 0.7), 2.2 * tw)
        s.fl.draw(f, t)
        s.pt.draw(f, t, k=1.0) if False else None
        for p in s.pt.items:
            p.setdefault('x0', p['x'])
        sh = s.wind[0] * t
        for p in s.pt.items: p['x'] = p['x0'] + sh * (1.4 if p['vy'] > 60 else 1.0)
        for p in s.pt.items: p['x'] = (p['x'] + 200) % (W + 400) - 200
        s.pt.draw(f, t)
        return f, 1.0 + 0.04 * t / d

class S8:  # candle + rose
    def __init__(s):
        s.bg = bg((0.09, 0.025, 0.025), cx=0.6, cy=0.4); s.b = HBokeh(23, 18, 80, 190, 0.32, (4, -6), hearts=0.5)
        s.rose = make_rose(330, 8); s.p = Petals(25, 8, vel=(8, 45), blurfrac=0.5)
        rb = to_im(s.rose).filter(ImageFilter.GaussianBlur(5)); s.roseb = to_np(rb)
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t, dx=-12 * t)
        candle(f, 230 + 10 * t, 1600, 300, 900, t, 160, seed=4, blur=22, k=0.75, tone=(0.96, 0.87, 0.82))
        candle(f, 700 - 18 * t, 2150, 480, 1100, t, 300, seed=2, tone=(0.96, 0.87, 0.82))
        over(f, s.roseb, 250 - 14 * t - s.roseb.shape[1] / 2, 1720 - s.roseb.shape[0] / 2)
        s.p.draw(f, t)
        return f, 1.0 + 0.05 * t / d

class S9:  # end card
    def __init__(s):
        s.bg = bg((0.12, 0.03, 0.04), cy=0.42); s.b = HBokeh(26, 26, 50, 160, 0.3, (4, -16), hearts=0.6); s.d = Dust(27, 70)
        s.rose = make_rose(250, 9); s.rose2 = make_rose(200, 10)
        s.top = Pine(28, 1050, 1.1, 4, bend=0.12, flip=True)
        s.p = Petals(29, 10, vel=(10, 50), blurfrac=0.5)
        s.t0 = text_layer([('con todo el cariño', 'Cormorant-MediumItalic.ttf', 64, 600, 0, 1)])
        s.t1 = text_layer([('Feliz', 'Cormorant-MediumItalic.ttf', 170, 730, 0, 1),
                           ('Navidad', 'Cormorant-MediumItalic.ttf', 210, 900, 0, 1)])
        s.t2 = text_layer([('RESERVA TU SESIÓN CON ROSA', 'Montserrat-SemiBold.ttf', 40, 1120, 7, 0),
                           ('Escríbele por mensaje directo', 'Cormorant-MediumItalic.ttf', 66, 1205, 0, 1)])
    def __call__(s, t, d):
        f = s.bg.copy(); s.b.draw(f, t); s.d.draw(f, t)
        s.top.draw(f, 1200, 170, 14 + 1.2 * math.sin(t), k=1.0)
        over(f, s.rose, 940 - s.rose.shape[1] / 2, 1820 - s.rose.shape[0] / 2)
        over(f, s.rose2, 680 - s.rose2.shape[1] / 2, 1960 - s.rose2.shape[0] / 2)
        k0 = sm(0.1, 1.0, t); draw_text(f, s.t0, k0, dy=int(20 * (1 - k0)))
        k1 = sm(0.6, 1.6, t); draw_text(f, s.t1, k1, dy=int(30 * (1 - k1)))
        k2 = sm(1.6, 2.5, t); draw_text(f, s.t2, k2, dy=int(20 * (1 - k2)))
        s.p.draw(f, t)
        return f, 1.0 + 0.025 * t / d

CUTS = [0, 3.0, 4.45, 5.8, 8.25, 13.0, 17.4, 21.5, 26.0, 32.0]
SHOTS = [S1, S2, S3, S4, S5, S6, S7, S8, S9]
XF = [0.6, 0.4, 0.45, 0.6, 0.6, 0.6, 0.6, 0.7]

def finish(f, fi):
    f = f * R.vign
    f = 1 - np.exp(-f * 1.25)
    f = f ** 0.95 * np.array([1.0, 0.95, 0.93], np.float32)
    g = np.random.default_rng(fi).normal(0, 0.008, (H // 2, W // 2)).astype(np.float32)
    f = f + np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

def shot_frame(inst, i, t):
    s0, s1 = CUTS[i], CUTS[i + 1]
    lead = XF[i - 1] / 2 if i > 0 else 0
    d = (s1 - s0) + lead + (XF[i] / 2 if i < len(XF) else 0)
    f, z = inst[i](t - s0 + lead, d)
    return R.zoom(finish(f, int(t * FPS) + i * 1000), z)

def main(f0, f1, out):
    inst = {}
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-preset', 'medium', '-crf', '14', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        t = fi / FPS
        act = [i for i in range(len(SHOTS)) if CUTS[i] - (XF[i - 1] / 2 if i > 0 else 0) <= t < CUTS[i + 1] + (XF[i] / 2 if i < len(XF) else 0)]
        for i in list(inst):
            if i not in act: del inst[i]
        for i in act:
            if i not in inst: inst[i] = SHOTS[i]()
        if len(act) == 1:
            img = shot_frame(inst, act[0], t)
        else:
            i, j = act[0], act[1]
            w = sm(CUTS[j] - XF[i] / 2, CUTS[j] + XF[i] / 2, t)
            img = (shot_frame(inst, i, t).astype(np.float32) * (1 - w) + shot_frame(inst, j, t).astype(np.float32) * w).astype(np.uint8)
        if fi < 6: img = (img.astype(np.float32) * (fi / 6)).astype(np.uint8)
        p.stdin.write(img.tobytes())
        if fi % 30 == 0: print(out, fi, flush=True)
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
