import numpy as np, subprocess, sys, os
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
TOTAL = 32.4
NF = int(TOTAL * FPS)
F32 = np.float32
yy, xx = np.mgrid[0:H, 0:W].astype(F32)

# scene starts (synced to 96 BPM: beat = 0.625 s) ; drop at 17.5
T = [0.0, 5.0, 8.75, 12.5, 17.5, 22.5, 26.25, TOTAL]
YELLOW = (247, 200, 96)
WHITE = (247, 244, 236)

def ease(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def eout(x):
    x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3
def lerp(a, b, k): return a + (b - a) * k

# ---------------------------------------------------------------- noise
def fbm(w, h, octaves, seed, base=3, gain=0.55):
    g = np.random.default_rng(seed); acc = np.zeros((h, w), F32); amp = 1.0
    for o in range(octaves):
        rx = int(base * 2 ** o) + 2; ry = int(base * 2 ** o * h / w) + 2
        a = g.random((ry, rx)).astype(F32)
        acc += amp * np.asarray(Image.fromarray(a, "F").resize((w, h), Image.BICUBIC)); amp *= gain
    return (acc - acc.mean()) / acc.std()

def fbm1d(n, seed, base=4, octaves=7, gain=0.5):
    g = np.random.default_rng(seed); x = np.linspace(0, 1, n); acc = np.zeros(n); amp = 1
    for o in range(octaves):
        k = base * 2 ** o; pts = g.uniform(-1, 1, k + 2)
        acc += amp * np.interp(x * (k + 1), np.arange(k + 2), pts); amp *= gain
    return acc

# ---------------------------------------------------------------- stars
def stars(n, seed, density=None, bmax=1.0):
    g = np.random.default_rng(seed)
    A = np.zeros((H, W), F32); B = np.zeros((H, W), F32)
    xs, ys = [], []
    while len(xs) < n:
        x, y = g.uniform(0, W), g.uniform(0, H)
        if density is None or g.random() < density[int(y), int(x)]:
            xs.append(x); ys.append(y)
    for x, y in zip(xs, ys):
        b = bmax * (0.08 + 0.92 * g.power(4)); s = 0.55 + 1.5 * g.power(6)
        L = A if g.random() < 0.5 else B
        r = int(np.ceil(s * 3)); x0, x1 = max(0, int(x) - r), min(W, int(x) + r + 1); y0, y1 = max(0, int(y) - r), min(H, int(y) + r + 1)
        gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        L[y0:y1, x0:x1] += 255 * b * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * (s * 0.55) ** 2))
    return A, B
def star_at(SA, t, k=1.0):
    return (SA[0] * (0.7 + 0.3 * np.sin(2.3 * t)) + SA[1] * (0.7 + 0.3 * np.sin(1.7 * t + 2))) * k

STARS = stars(1100, 1)
# Milky Way for the new moon scene
mw_d = np.abs((yy - 0.95 * xx) - 300) / np.sqrt(1 + 0.95 ** 2)
mw_band = np.exp(-(mw_d / 260) ** 2) * (0.55 + 0.45 * np.clip(fbm(W, H, 6, 21) * 0.5 + 0.5, 0, 1))
mw_dust = np.clip(fbm(W, H, 6, 22, base=5) - 0.4, 0, 2) * np.exp(-(mw_d / 120) ** 2)
mw_band = np.clip(mw_band - 0.25 * np.clip(mw_dust, 0, 0.8), 0, 1)
STARS_MW = stars(3200, 2, density=0.12 + 0.88 * mw_band, bmax=0.9)

# ---------------------------------------------------------------- moon
MS = 1024
def make_albedo(n):
    a = 0.78 + 0.22 * np.clip(fbm(n, n, 7, 3) * 0.3, -1, 1)
    v, u = np.mgrid[0:n, 0:n].astype(F32) / n * 2 - 1
    maria = [(-0.35, -0.45, 0.30), (-0.05, -0.38, 0.22), (0.25, -0.25, 0.20), (0.45, -0.05, 0.14),
             (0.30, 0.15, 0.17), (-0.55, 0.05, 0.36), (-0.25, 0.30, 0.18), (0.05, 0.05, 0.12),
             (-0.60, -0.25, 0.18), (0.55, 0.30, 0.10)]
    m = np.zeros_like(a); noise = fbm(n, n, 5, 11) * 0.15 + 0.5
    for cx, cy, rr in maria:
        d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2) / rr
        m = np.maximum(m, np.clip(1.25 - d - 0.5 * (noise - 0.5), 0, 1))
    a = a * (1 - 0.38 * np.clip(m * 1.6, 0, 1))
    g = np.random.default_rng(5)
    for _ in range(90):
        cx, cy = g.uniform(-0.95, 0.95, 2); rr = 0.012 + 0.07 * g.power(6)
        d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2) / rr
        a += np.exp(-((d - 1) / 0.25) ** 2) * 0.035 - 0.035 * np.exp(-(d / 0.8) ** 4)
    cx, cy = -0.12, 0.62
    d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2); ang = np.arctan2(v - cy, u - cx)
    rays = (np.abs(np.sin(ang * 9 + 0.3)) ** 30 + np.abs(np.sin(ang * 13)) ** 40) * np.exp(-d / 0.6)
    a += 0.18 * np.exp(-(d / 0.03) ** 2) + 0.03 * rays
    a = np.clip(a, 0, 1.2)
    from scipy.ndimage import gaussian_filter; a = gaussian_filter(a.astype(F32), 1.8)
    return a[::-1, ::-1].copy()   # southern hemisphere view (Argentina): rotated 180°
ALB = make_albedo(MS)

def render_moon(kind, size, warm=False):
    n = MS
    v, u = np.mgrid[0:n, 0:n].astype(F32) / (n - 1) * 2 - 1
    rr = u ** 2 + v ** 2; rad = np.sqrt(rr); un = u / np.maximum(1, rad)
    zz = np.sqrt(np.clip(1 - np.minimum(rr, 1), 0, 1))
    # southern hemisphere: waxing (creciente) lit on the LEFT, waning (menguante) on the RIGHT
    th, sg = {"full": (0.0, 1), "gib": (0.28, 1), "fq": (np.pi / 2, -1), "lq": (np.pi / 2, 1), "new": (np.pi, 1)}[kind]
    lam = un * sg * np.sin(th) + zz * np.cos(th)
    ls = np.clip(2 * lam / (lam + zz + 0.002), 0, 1)
    lam2 = un * sg * np.sin(th) + np.maximum(zz, 0.3) * np.cos(th)
    term = np.clip(lam2 * 7 + 0.15, 0, 1); term = term * term * (3 - 2 * term)
    limb = 0.92 + 0.08 * zz
    es = 0.05 if kind != "new" else 0.09
    val = ALB * (ls * term * limb + es * (1 - term) * limb)
    tint = np.array([1.0, 0.90, 0.74]) if warm else np.array([0.97, 0.975, 1.0])
    rgb = np.clip(val[..., None] * tint * 248, 0, 255)
    if kind == "new":
        rim = np.exp(-((1 - np.minimum(rad, 1.5)) * n / 2 / 6) ** 2)
        rgb = rgb + rim[..., None] * np.array([80, 95, 140])
    alpha = np.clip((1 - rad) * n / 2 + 0.5, 0, 1)
    im = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8), "RGBA").convert("RGBa").resize((size, size), Image.LANCZOS).convert("RGBA")
    arr = np.asarray(im, F32)
    return arr[..., :3], arr[..., 3] / 255

def paste(c, mo, cx, cy, op=1.0, scale=1.0):
    rgb, a = mo
    if scale != 1.0:
        s = max(2, int(rgb.shape[0] * scale))
        im = Image.fromarray(np.dstack([rgb, a * 255]).astype(np.uint8), "RGBA").convert("RGBa").resize((s, s), Image.BICUBIC).convert("RGBA")
        arr = np.asarray(im, F32); rgb, a = arr[..., :3], arr[..., 3] / 255
    s = rgb.shape[0]; x0 = int(round(cx - s / 2)); y0 = int(round(cy - s / 2))
    X0, Y0 = max(0, x0), max(0, y0); X1, Y1 = min(W, x0 + s), min(H, y0 + s)
    if X1 <= X0 or Y1 <= Y0: return
    sa = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None] * op
    c[Y0:Y1, X0:X1] = c[Y0:Y1, X0:X1] * (1 - sa) + rgb[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * sa

def glow(c, cx, cy, R, col, s):
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = 0.65 * np.exp(-d2 / R ** 2) + 0.35 / (1 + d2 / (0.6 * R) ** 2)
    c += g[..., None] * (np.array(col, F32) * s)

def vgrad(stops):
    """stops: list of (y, (r,g,b))"""
    ys = np.array([s[0] for s in stops], F32); out = np.zeros((H, 3), F32)
    for ch in range(3):
        out[:, ch] = np.interp(np.arange(H), ys, [s[1][ch] for s in stops])
    return np.repeat(out[:, None, :], W, axis=1)

def ridge(seed, base, amp, color, rim=None, rim_s=0.0, rough=0.5, oct_=8, kbase=3):
    h = base - amp * (fbm1d(W, seed, base=kbase, octaves=oct_, gain=rough) * 0.5 + 0.5)
    a = np.clip(yy - h[None, :] + 0.5, 0, 1)
    col = np.broadcast_to(np.array(color, F32), (H, W, 3)).copy()
    if rim is not None:
        e = np.exp(-np.clip(yy - h[None, :], 0, None) / 5)
        col += e[..., None] * np.array(rim, F32) * rim_s
    return a, col
def put_ridge(c, R, dy=0):
    a, col = R; k = int(round(dy))
    if k > 0:
        a = np.vstack([np.zeros((k, W), F32), a[:-k]]); col = np.vstack([col[:1].repeat(k, 0), col[:-k]])
    c[:] = c * (1 - a[..., None]) + col * a[..., None]

def clouds_on(c, n, thr, soft, light, base_col, light_col, op=1.0, ox=0, oy=0, relief=0.0):
    h, w = n.shape; ox = int(ox) % (w - W - 12) if w > W + 12 else 0
    nn = n[oy:oy + H, ox:ox + W]
    a = np.clip((nn - thr) / soft, 0, 1) * op
    shade = 0.75 + 0.25 * (1 - a)
    if relief:
        emb = np.clip(nn - n[oy:oy + H, ox + 10:ox + 10 + W], -1.2, 1.2)
        shade = shade * (0.68 + 0.32 * np.tanh(nn * 0.7)) + relief * emb
    col = np.array(base_col, F32) * shade[..., None] + \
          np.array(light_col, F32) * (light * (1.1 - 0.45 * a))[..., None]
    c[:] = c * (1 - a[..., None]) + col * a[..., None]
    return a

def lightmap(cx, cy, R):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / R ** 2).astype(F32)

# ---------------------------------------------------------------- assets
M_GIB = render_moon("gib", 780)
M_LQ = render_moon("lq", 340)
M_FQ = render_moon("fq", 320)
M_SUPER = render_moon("full", 900, warm=True)
M_LQ2 = render_moon("lq", 430)
MINI = [render_moon(k, 132, warm=(k == "full")) for k in ("lq", "new", "fq", "full", "lq")]

CL_A = fbm(2 * W, H, 7, 11, base=2)
CL_B = fbm(2 * W, H, 7, 12, base=3)
CL_RUSH = fbm(W + 1900, H, 7, 13, base=3)
CL_WISP = fbm(2 * W, H, 6, 14, base=2)
CL_WISP = np.asarray(Image.fromarray(CL_WISP, "F").resize((2 * W, H // 3)).resize((2 * W, H)))  # stretched horizontal wisps
CL_WISP = (CL_WISP - CL_WISP.mean()) / CL_WISP.std()

# scene 0 (intro)
S0_SKY = vgrad([(0, (5, 8, 22)), (H, (16, 22, 48))]); glow(S0_SKY, 540, 740, 700, (40, 52, 96), 1.0)
S0_L = lightmap(540, 740, 640)
# scene 1 (mountains)
S1_SKY = vgrad([(0, (6, 10, 28)), (900, (22, 32, 68)), (1250, (52, 66, 112)), (H, (52, 66, 112))])
glow(S1_SKY, 700, 560, 420, (80, 100, 160), 0.8)
S1_R = [ridge(31, 1150, 260, (38, 50, 88), (120, 140, 190), 0.5, kbase=2),
        ridge(32, 1330, 240, (20, 27, 54), (110, 130, 180), 0.6, kbase=3),
        ridge(33, 1560, 300, (6, 8, 18), (90, 110, 160), 0.5, kbase=2)]
S1_MIST = [np.exp(-((yy - y) / 60) ** 2)[..., None] * np.array([60, 72, 110], F32) * 0.35 for y in (1170, 1350)]
S1_L = lightmap(700, 560, 400)
# scene 2 (new moon / dark starry sky)
S2_SKY = vgrad([(0, (2, 3, 10)), (H, (9, 11, 26))]) + mw_band[..., None] * np.array([46, 46, 70], F32)
S2_SKY = np.clip(S2_SKY - np.clip(mw_dust, 0, 0.6)[..., None] * 12, 0, 255)
S2_HILL = ridge(41, 1760, 120, (2, 2, 6), kbase=2, rough=0.45)
# scene 3 (water reflection)
HZ = 1060
S3_SKY = vgrad([(0, (6, 10, 30)), (700, (20, 30, 66)), (HZ, (44, 58, 104)), (H, (44, 58, 104))])
glow(S3_SKY, 430, 560, 380, (80, 100, 160), 0.8)
S3_SHORE = ridge(51, HZ + 2, 70, (8, 10, 22), (60, 70, 110), 0.4, rough=0.62, kbase=6)
S3_SHORE = (S3_SHORE[0] * np.clip(HZ + 3 - yy, 0, 1), S3_SHORE[1])
def _gl(seed):
    r = np.random.default_rng(seed).random(((H - HZ) // 5, (W + 400) // 14)).astype(F32)
    return np.asarray(Image.fromarray(r, "F").resize((W + 400, H - HZ), Image.BILINEAR))
S3_GL1 = _gl(52); S3_GL2 = _gl(53)
S3_L = lightmap(430, 560, 380)
# scene 4 (superluna)
S4_SKY = vgrad([(0, (8, 8, 26)), (700, (34, 26, 66)), (1150, (110, 70, 72)), (1320, (170, 108, 76)), (H, (170, 108, 76))])
S4_RIDGE = ridge(61, 1420, 200, (10, 6, 14), (255, 190, 130), 0.9, kbase=2, rough=0.5)
S4_RIDGE2 = ridge(62, 1300, 120, (44, 26, 40), (255, 180, 120), 0.5, kbase=3)
# scene 5 (clouds revealing moon)
S5_SKY = vgrad([(0, (6, 9, 26)), (H, (26, 32, 62))]); glow(S5_SKY, 540, 700, 480, (70, 86, 140), 0.8)
S5_L = lightmap(540, 700, 520)
S5_HOLE = np.exp(-(((xx - 540) ** 2 + (yy - 700) ** 2) / 420 ** 2)).astype(F32)
# scene 6 (outro)
S6_SKY = vgrad([(0, (3, 5, 16)), (H, (12, 16, 36))]); glow(S6_SKY, 540, 760, 650, (30, 36, 70), 1.0)

# vignette + bottom darkening for legibility
VIG = np.clip(1 - 0.32 * (((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H / 2) / (H * 0.68)) ** 2), 0.45, 1)
VIG *= 1 - 0.38 * ease((yy - 1180) / 520)
VIG = VIG[..., None].astype(F32)

# ---------------------------------------------------------------- text
FD = os.path.join(HERE, "fonts")
def font(name, size, wght=None):
    f = ImageFont.truetype(os.path.join(FD, name), size)
    if wght:
        try: f.set_variation_by_axes([wght])
        except Exception: pass
    return f
REG = "CormorantGaramond%5Bwght%5D.ttf"; ITA = "CormorantGaramond-Italic%5Bwght%5D.ttf"
MEAS = ImageDraw.Draw(Image.new("RGB", (8, 8)))

def text_line(runs, baseline, track=0):
    width = sum(MEAS.textlength(t, font=f) for t, f, _ in runs) + track * (sum(len(t) for t, _, _ in runs) - 1)
    x = (W - width) / 2; y0 = baseline - 230; hh = 320
    im = Image.new("RGBA", (W, hh), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    x0 = x
    for t, f, col in runs:
        if track:
            for ch in t:
                d.text((x, 230), ch, font=f, fill=col + (255,), anchor="ls"); x += MEAS.textlength(ch, font=f) + track
        else:
            d.text((x, 230), t, font=f, fill=col + (255,), anchor="ls"); x += MEAS.textlength(t, font=f)
    arr = np.asarray(im, F32)
    sh = np.asarray(im.getchannel("A").filter(ImageFilter.GaussianBlur(14)), F32) / 255
    return dict(y0=y0, rgb=arr[..., :3], a=arr[..., 3] / 255, sh=sh, x0=x0, x1=x0 + width)

XS = np.arange(W, dtype=F32)
def draw_line(c, L, p, op):
    if p <= 0 or op <= 0: return
    soft = 140
    front = L["x0"] - soft + p * (L["x1"] - L["x0"] + 2 * soft)
    m = np.clip((front - XS) / soft, 0, 1) * op
    y0 = L["y0"]; y1 = y0 + L["a"].shape[0]
    Y0, Y1 = max(0, y0), min(H, y1)
    reg = c[Y0:Y1]; s = slice(Y0 - y0, Y1 - y0)
    sh = L["sh"][s] * m * 0.7
    reg *= (1 - sh[..., None])
    a = (L["a"][s] * m)[..., None]
    c[Y0:Y1] = reg * (1 - a) + L["rgb"][s] * a

def date_phase(day, phase):
    fday = font(REG, 104, 700); fde = font(ITA, 84, 500)
    size = 122
    while True:
        fph = font(REG, size, 600)
        if MEAS.textlength(phase, font=fph) <= 880: break
        size -= 2
    l1 = text_line([(day, fday, YELLOW), (" de diciembre", fde, WHITE)], 1415)
    l2 = text_line([(phase, fph, WHITE)], 1555)
    return l1, l2

TX = {1: date_phase("1", "Cuarto menguante"), 2: date_phase("8", "Luna nueva"),
      3: date_phase("17", "Cuarto creciente"), 4: date_phase("23", "Luna llena · Superluna"),
      5: date_phase("30", "Cuarto menguante")}
TITLE = (text_line([("Diciembre", font(REG, 168, 500), WHITE)], 1440, track=4),
         text_line([("bajo la ", font(ITA, 108, 500), WHITE), ("Luna", font(ITA, 108, 600), YELLOW)], 1580))
OUT1 = text_line([("Guardá este", font(REG, 118, 500), WHITE)], 1120)
OUT2 = text_line([("calendario lunar", font(REG, 118, 500), WHITE)], 1250)
SIGN = text_line([("Rosa", font("GreatVibes-Regular.ttf", 210), YELLOW)], 1530)
MINI_NUM = [text_line([(d, font(REG, 60, 700), YELLOW)], 860) for d in ("1", "8", "17", "23", "30")]

# ---------------------------------------------------------------- camera
def camera(c, z, fx, fy, ang=0.0):
    img = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8))
    if ang: img = img.rotate(ang, resample=Image.BILINEAR, center=(fx, fy))
    cw, ch = W / z, H / z
    x0 = float(np.clip(fx - cw / 2, 0, W - cw)); y0 = float(np.clip(fy - ch / 2, 0, H - ch))
    return np.asarray(img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch)), F32).copy()

# ---------------------------------------------------------------- scenes
def sc0(u):
    c = S0_SKY + star_at(STARS, u, 0.5)[..., None]
    glow(c, 540, 740, 460, (120, 135, 190), 0.35 * ease(u / 2.5))
    paste(c, M_GIB, 540, 740)
    thr = lerp(-0.35, 1.2, ease((u - 0.1) / 3.0))
    clouds_on(c, CL_A, thr, 0.9, S0_L, (34, 40, 70), (215, 222, 245), op=0.92, ox=200 + u * 70)
    clouds_on(c, CL_B, thr + 0.55, 0.8, S0_L, (38, 44, 74), (225, 230, 250), op=0.7, ox=600 + u * 150)
    return camera(c, lerp(1.0, 1.12, ease(u / 5.0)), 540, 800)

def sc1(u):
    c = S1_SKY + star_at(STARS, u)[..., None]
    paste(c, M_LQ, 700, 560 + u * 8)
    clouds_on(c, CL_WISP, 1.1, 0.9, S1_L, (40, 50, 84), (160, 175, 215), op=0.45, ox=300 + u * 40)
    put_ridge(c, S1_R[0], u * 5); c += S1_MIST[0]
    put_ridge(c, S1_R[1], u * 12); c += S1_MIST[1]
    put_ridge(c, S1_R[2], u * 24)
    return camera(c, lerp(1.0, 1.09, ease(u / 3.75)), 640, 820)

def sc2(u):
    c = S2_SKY + star_at(STARS_MW, u)[..., None] + star_at(STARS, u + 3, 0.6)[..., None]
    # shooting star on the beat
    st = u - 1.875
    if 0 < st < 0.6:
        p = st / 0.45
        hx, hy = lerp(260, 760, min(p, 1)), lerp(330, 640, min(p, 1))
        L = 260; dx, dy = 500 / np.hypot(500, 310), 310 / np.hypot(500, 310)
        along = (xx - hx) * dx + (yy - hy) * dy; perp = (xx - hx) * -dy + (yy - hy) * dx
        tail = np.exp(-(perp / 1.6) ** 2) * np.clip(1 + along / L, 0, 1) * (along <= 0)
        k = (1 - ease((st - 0.35) / 0.25)) * 255
        c += (tail * k)[..., None] * np.array([0.9, 0.95, 1.0], F32)
    put_ridge(c, S2_HILL)
    return camera(c, lerp(1.07, 1.16, ease(u / 3.75)), 540, 900, ang=lerp(0, 1.6, u / 3.75))

def sc3(u):
    c = S3_SKY + star_at(STARS, u)[..., None]
    paste(c, M_FQ, 430, 560)
    clouds_on(c, CL_WISP, 1.3, 0.9, S3_L, (40, 50, 84), (160, 175, 215), op=0.35, ox=500 + u * 35)
    # water: rippled reflection of the sky
    ys = np.arange(HZ, H, dtype=F32); dep = (ys - HZ) / (H - HZ)
    sy = np.clip(HZ - 1 - (ys - HZ) * 0.85 + 2.5 * np.sin(ys * 0.21 - u * 3), 0, HZ - 1).astype(int)
    amp = 1 + 10 * dep
    dx = amp * np.sin(0.045 * (ys - HZ) ** 1.05 - 2.2 * u) + 0.5 * amp * np.sin(0.017 * ys + 1.2 * u + 1)
    sx = np.clip(XS[None, :] + dx[:, None], 0, W - 1).astype(int)
    refl = c[sy[:, None], sx] * 0.6 + np.array([3, 7, 18], F32)
    # moon glitter path
    o = int(u * 50) % 400
    g = S3_GL1[:, o:o + W] * np.cos(u * 4) ** 2 + S3_GL2[:, o:o + W] * np.sin(u * 4) ** 2
    wid = 30 + 190 * dep
    path = np.exp(-((XS[None, :] - 430) / wid[:, None]) ** 2)
    sp = np.clip((g - 0.80) / 0.12, 0, 1) * path * (0.35 + 0.65 * dep[:, None])
    refl += sp[..., None] * np.array([200, 210, 240], F32)
    c[HZ:] = refl
    put_ridge(c, S3_SHORE)
    # build: clouds roll in before the drop
    if u > 3.9:
        k = ease((u - 3.9) / 1.1)
        clouds_on(c, CL_RUSH, lerp(3.6, -3.8, k) + 2.2 * (XS / W - 0.5), 0.8, S3_L, (44, 46, 74), (220, 200, 180), ox=200 + u * 260, relief=0.9)
        c += k ** 3 * np.array([90, 70, 50], F32)
    return camera(c, lerp(1.0, 1.18, (u / 5.0) ** 1.6), 480, 900)

def sc4(u):
    c = S4_SKY + star_at(STARS, u, 0.4)[..., None]
    my = lerp(1230, 860, eout(u / 4.6)); mx = 540
    glow(c, mx, my, 820, (255, 190, 120), 0.45)
    # light rays
    d = np.sqrt((xx - mx) ** 2 + (yy - my) ** 2); ang = np.arctan2(yy - my, xx - mx)
    rays = ((0.5 + 0.5 * np.sin(ang * 23 + u * 0.15)) ** 8 + (0.5 + 0.5 * np.sin(ang * 37 + 1.3 - u * 0.1)) ** 12)
    rays *= 0.5 + 0.5 * np.sin(ang * 5 + 0.7) ** 2
    c += (rays * np.exp(-d / 600) * np.clip((d - 430) / 80, 0, 1) * 24)[..., None] * np.array([1.0, 0.8, 0.55], F32)
    paste(c, M_SUPER, mx, my)
    L = lightmap(mx, my, 620)
    clouds_on(c, CL_WISP, 1.0, 0.8, L, (70, 44, 54), (255, 205, 150), op=0.55, ox=900 + u * 60)
    put_ridge(c, S4_RIDGE2, u * 4)
    put_ridge(c, S4_RIDGE, u * 10)
    # clouds part on the drop
    if u < 1.0:
        clouds_on(c, CL_RUSH, lerp(-3.8, 3.6, eout(u / 1.0)) - 2.2 * (XS / W - 0.5), 0.8, L, (60, 46, 66), (255, 215, 170), ox=1500 + u * 300, relief=0.9)
    c = camera(c, lerp(1.32, 1.0, eout(u / 3.2)), 540, 1000)
    c += np.exp(-u * 4.5) * np.array([255, 238, 212], F32) * 0.95   # flash on the drop
    return c

def sc5(u):
    c = S5_SKY + star_at(STARS, u, 0.7)[..., None]
    paste(c, M_LQ2, 540, 700)
    open_ = eout((u - 0.2) / 2.4)
    thr_back = 0.2 + 2.2 * S5_HOLE * open_
    n = CL_B[:, int(400 + u * 40):int(400 + u * 40) + W]
    a = np.clip((n - thr_back) / 0.7, 0, 1)
    col = np.array([28, 34, 60], F32) * (0.75 + 0.25 * (1 - a))[..., None] + np.array([200, 210, 240], F32) * (S5_L * (1.2 - 0.75 * a))[..., None]
    c[:] = c * (1 - a[..., None]) + col * a[..., None]
    clouds_on(c, CL_A, 0.6 + 1.6 * open_, 0.6, S5_L, (22, 26, 48), (190, 200, 235), op=0.9, ox=800 + u * 160)
    return camera(c, lerp(1.0, 1.09, ease(u / 3.75)), 540, 760)

def sc6(u):
    c = S6_SKY + star_at(STARS, u)[..., None]
    for i, mo in enumerate(MINI):
        k = eout((u - 0.35 - i * 0.3125) / 0.5)
        if k > 0:
            cx = 540 + (i - 2) * 196
            glow(c, cx, 700, 90, (110, 120, 170) if i != 3 else (255, 200, 130), 0.25 * k)
            paste(c, mo, cx, 700 + (1 - k) * 30, op=k, scale=0.9 + 0.1 * k)
    c = camera(c, lerp(1.0, 1.04, u / 6.0), 540, 900)
    for i, L in enumerate(MINI_NUM):
        L2 = dict(L); off = (i - 2) * 196
        L2["x0"] += off; L2["x1"] += off
        sl = L["rgb"]; a = L["a"]; sh = L["sh"]
        L2["rgb"] = np.roll(sl, off, axis=1); L2["a"] = np.roll(a, off, axis=1); L2["sh"] = np.roll(sh, off, axis=1)
        draw_line(c, L2, eout((u - 0.55 - i * 0.3125) / 0.4), 1.0)
    return c

SCENES = [sc0, sc1, sc2, sc3, sc4, sc5, sc6]

def overlays(c, t):
    # cloud-rush transitions
    for tc in (5.0, 22.5):
        dt = t - tc
        if abs(dt) < 0.7:
            k = np.clip(1.15 * (1 - abs(dt) / 0.7), 0, 1)
            clouds_on(c, CL_RUSH, lerp(3.6, -3.0, ease(k)) + 2.2 * (XS / W - 0.5), 0.8, 0.22, (40, 46, 78), (170, 180, 220), ox=950 + dt * 1300, relief=0.9)
    # darkness dips
    for tc, hw in ((8.75, 0.55), (12.5, 0.5)):
        dt = abs(t - tc)
        if dt < hw: c *= ease(dt / hw) ** 1.3
    # clouds + darkness into the outro
    dt = t - 26.25
    if abs(dt) < 0.8:
        k = np.clip(1.3 * (1 - abs(dt) / 0.8), 0, 1)
        clouds_on(c, CL_RUSH, lerp(3.6, -3.8, ease(k)) - 2.2 * (XS / W - 0.5), 0.8, 0.15, (30, 34, 60), (140, 150, 190), ox=300 + dt * 900, relief=0.9)
        c *= 1 - 0.85 * ease(k)
    if t < 0.25: c *= ease(t / 0.25)          # quick open
    if t > TOTAL - 1.0: c *= 1 - ease((t - (TOTAL - 1.0)) / 0.9)

def frame(fi):
    t = fi / FPS
    s = max(i for i in range(7) if T[i] <= t)
    u = t - T[s]; D = T[s + 1] - T[s]
    c = SCENES[s](u)
    c *= VIG
    # text
    out_op = 1 - ease((u - (D - 0.5)) / 0.38) if s < 6 else 1.0
    if s == 0:
        draw_line(c, TITLE[0], eout((u - 0.65) / 0.9), out_op)
        draw_line(c, TITLE[1], eout((u - 1.25) / 0.8), out_op)
    elif s == 4:
        draw_line(c, TX[4][0], eout((u - 0.55) / 0.6), out_op)
        draw_line(c, TX[4][1], eout((u - 1.1) / 0.8), out_op)
    elif s < 6:
        draw_line(c, TX[s][0], eout((u - 0.3) / 0.6), out_op)
        draw_line(c, TX[s][1], eout((u - 0.65) / 0.8), out_op)
    else:
        draw_line(c, OUT1, eout((u - 2.0) / 0.8), 1)
        draw_line(c, OUT2, eout((u - 2.45) / 0.8), 1)
        draw_line(c, SIGN, eout((u - 3.1) / 1.3), 1)
    overlays(c, t)
    return np.clip(c, 0, 255).astype(np.uint8).tobytes()

if __name__ == "__main__":
    if os.environ.get("PREVIEW"):
        ts = [float(x) for x in os.environ["PREVIEW"].split(",")]
        for t in ts:
            Image.frombytes("RGB", (W, H), frame(int(t * FPS))).save(os.path.join(HERE, f"p_{t:05.2f}.png"))
        sys.exit()
    out, audio = sys.argv[1], sys.argv[2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio,
                           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for b in pool.imap(frame, range(NF), chunksize=4):
            ff.stdin.write(b)
    ff.stdin.close(); ff.wait()
    print("done", out)
