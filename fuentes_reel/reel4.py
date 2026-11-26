"""Reel «Diciembre bajo la Luna» with real footage (Mixkit) + real Moon images (NASA SVS Dial-a-Moon)."""
import numpy as np, subprocess, sys, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clips import SPEC, MK, CW, CH, probe

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
TOTAL = 45.15
NF = int(TOTAL * FPS)
F32 = np.float32
T = [0.0, 5.0, 11.25, 17.5, 25.0, 32.5, 38.75, TOTAL]     # beat-synced cuts (96 BPM); drop at 25.0
NAMES = ["intro", "d01", "d08", "d17", "d23", "d30", "outro"]
YELLOW = (247, 200, 96); WHITE = (247, 244, 236)

def ease(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def eout(x):
    x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3
def lerp(a, b, k): return a + (b - a) * k

# ---------------------------------------------------------------- footage streaming
class Clip:
    def __init__(self, name, dur, size=(CW, CH)):
        s = SPEC[name]; path = os.path.join(MK, s["f"]); sw, sh = probe(path)
        cw = int(round(sh * 9 / 16)) // 2 * 2; x0 = int(np.clip(s["cx"] - cw / 2, 0, sw - cw))
        self.w, self.h = size
        vf = (f"setpts=PTS/{s['speed']},fps={FPS},crop={cw}:{sh}:{x0}:0,"
              f"scale={self.w}:{self.h}:flags=lanczos,unsharp=5:5:0.5:5:5:0")
        self.p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", str(s["t0"]), "-i", path,
                                   "-t", str(dur * s["speed"] + 0.5), "-vf", vf, "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.last = None
    def next(self):
        b = self.p.stdout.read(self.w * self.h * 3)
        if len(b) == self.w * self.h * 3:
            self.last = np.frombuffer(b, np.uint8).reshape(self.h, self.w, 3)
        return self.last

def load_clip(name, dur, size):
    c = Clip(name, dur, size); return np.stack([c.next().copy() for _ in range(int(dur * FPS) + 1)])

# ---------------------------------------------------------------- real moons (NASA, south-up)
def moon_img(name, d, tint=(1, 1, 1)):
    im = Image.open(os.path.join(HERE, "real", "moon", name + ".png")).convert("RGBA")
    im = im.convert("RGBa").resize((d, d), Image.LANCZOS).convert("RGBA")
    a = np.asarray(im, F32); rgb = a[..., :3] * np.array(tint, F32); al = a[..., 3] / 255
    return rgb, al

def region(c, cx, cy, s):
    x0 = int(round(cx - s / 2)); y0 = int(round(cy - s / 2))
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(c.shape[1], x0 + s), min(c.shape[0], y0 + s)
    return (slice(Y0, Y1), slice(X0, X1)), (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0))

def moon_screen(c, mo, cx, cy, gain=1.0):
    """Screen blend: the lit side shows, the unlit side disappears into the real sky."""
    rgb, al = mo; s = rgb.shape[0]
    (cs, ms) = region(c, cx, cy, s)
    lum = rgb[ms].mean(2, keepdims=True)
    m = np.clip(rgb[ms] * al[ms][..., None] * gain * np.clip((lum - 22) / 35, 0, 1), 0, 255)
    c[cs] = 255 - (255 - c[cs]) * (255 - m) / 255

def moon_over(c, mo, cx, cy, op=1.0):
    rgb, al = mo; s = rgb.shape[0]; (cs, ms) = region(c, cx, cy, s)
    a = al[ms][..., None] * op
    c[cs] = c[cs] * (1 - a) + rgb[ms] * a

def halo(c, cx, cy, R, col, s):
    pad = int(R * 3); (cs, _) = region(c, cx, cy, 2 * pad)
    ys = np.arange(cs[0].start, cs[0].stop, dtype=F32)[:, None]; xs = np.arange(cs[1].start, cs[1].stop, dtype=F32)[None, :]
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    g = 0.6 * np.exp(-d2 / R ** 2) + 0.4 / (1 + d2 / (0.55 * R) ** 2)
    c[cs] += g[..., None] * np.array(col, F32) * s

M01 = moon_img("d01", 330); M17 = moon_img("d17", 300); M30 = moon_img("d30", 300, (0.93, 0.97, 1.05))
M23 = moon_img("d23", 1044, (1.13, 1.06, 0.95))
M23 = (M23[0], np.asarray(Image.fromarray((M23[1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)), F32) / 255)
MINI = [moon_img(n, 132) for n in ("d01", "d08", "d17", "d23", "d30")]

# reflection of the first-quarter moon on the lake (flipped, stretched, blurred)
_r = Image.fromarray(np.dstack([np.clip(M17[0] * M17[1][..., None], 0, 255), M17[1] * 255]).astype(np.uint8), "RGBA")
_r = _r.transpose(Image.FLIP_TOP_BOTTOM).resize((300, 470), Image.BICUBIC).filter(ImageFilter.GaussianBlur(7))
_fade = np.clip(1 - np.abs(np.linspace(-1, 1, 470)) ** 2, 0, 1)[:, None, None]
REFL = np.asarray(_r, F32)[..., :3] * 0.42 * _fade

# ---------------------------------------------------------------- text (same typography as before)
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
    x = (W - width) / 2; y0 = baseline - 230
    im = Image.new("RGBA", (W, 320), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x0 = x
    for t, f, col in runs:
        for ch in (t if track else [t]):
            d.text((x, 230), ch, font=f, fill=col + (255,), anchor="ls"); x += MEAS.textlength(ch, font=f) + (track if track else 0)
    arr = np.asarray(im, F32)
    sh = np.asarray(im.getchannel("A").filter(ImageFilter.GaussianBlur(16)), F32) / 255
    return dict(y0=y0, rgb=arr[..., :3], a=arr[..., 3] / 255, sh=sh, x0=x0, x1=x0 + width)

XS = np.arange(W, dtype=F32)
def draw_line(c, L, p, op):
    if p <= 0 or op <= 0: return
    soft = 140; front = L["x0"] - soft + p * (L["x1"] - L["x0"] + 2 * soft)
    m = np.clip((front - XS) / soft, 0, 1) * op
    y0 = L["y0"]; y1 = y0 + L["a"].shape[0]; Y0, Y1 = max(0, y0), min(H, y1); s = slice(Y0 - y0, Y1 - y0)
    reg = c[Y0:Y1]; reg *= (1 - (L["sh"][s] * m * 0.8)[..., None])
    a = (L["a"][s] * m)[..., None]; c[Y0:Y1] = reg * (1 - a) + L["rgb"][s] * a

def date_phase(day, phase):
    fday = font(REG, 104, 700); fde = font(ITA, 84, 500); size = 122
    while MEAS.textlength(phase, font=font(REG, size, 600)) > 880: size -= 2
    return (text_line([(day, fday, YELLOW), (" de diciembre", fde, WHITE)], 1415),
            text_line([(phase, font(REG, size, 600), WHITE)], 1555))
TX = {1: date_phase("1", "Cuarto menguante"), 2: date_phase("8", "Luna nueva"),
      3: date_phase("17", "Cuarto creciente"), 4: date_phase("23", "Luna llena · Superluna"),
      5: date_phase("30", "Cuarto menguante")}
FPHR = font(ITA, 80, 500)
def phrase(l1, l2):
    size = 80
    while max(MEAS.textlength(l, font=font(ITA, size, 500)) for l in (l1, l2)) > 920: size -= 2
    f = font(ITA, size, 500)
    return (text_line([(l1, f, WHITE)], 1548), text_line([(l2, f, WHITE)], 1640))
PH = {1: phrase("Momento de soltar y ordenar", "antes de que termine el año"),
      2: phrase("La última luna nueva del año:", "ideal para sembrar intenciones"),
      3: phrase("El último cuarto creciente del año:", "lo que empezaste toma impulso"),
      4: phrase("La Superluna más grande del año", "ilumina la previa de Navidad"),
      5: phrase("El último cuarto menguante del año:", "súper poderoso para cerrar ciclos")}
TITLE = (text_line([("Diciembre", font(REG, 168, 500), WHITE)], 1440, track=4),
         text_line([("bajo la ", font(ITA, 108, 500), WHITE), ("Luna", font(ITA, 108, 600), YELLOW)], 1580))
OUT1 = text_line([("Guardá este", font(REG, 118, 500), WHITE)], 1120)
OUT2 = text_line([("calendario lunar", font(REG, 118, 500), WHITE)], 1250)
SIGN = text_line([("Rosa", font("GreatVibes-Regular.ttf", 210), YELLOW)], 1530)
MINI_NUM = []
for i, d in enumerate(("1", "8", "17", "23", "30")):
    L = text_line([(d, font(REG, 60, 700), YELLOW)], 860); off = (i - 2) * 196
    for k in ("rgb", "a", "sh"): L[k] = np.roll(L[k], off, axis=1)
    L["x0"] += off; L["x1"] += off; MINI_NUM.append(L)

yy, xx = np.mgrid[0:H, 0:W].astype(F32)
VIG = np.clip(1 - 0.3 * (((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H / 2) / (H * 0.68)) ** 2), 0.5, 1)
VIG *= 1 - 0.42 * ease((yy - 1150) / 560)
VIG = VIG[..., None].astype(F32)
del yy, xx

def camera(c, z, fx, fy):
    """z=1 shows the whole 1296x2304 canvas; larger z pushes in around (fx, fy)."""
    img = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8))
    z = max(1.0, z); cw, ch = CW / z, CH / z
    x0 = float(np.clip(fx - cw / 2, 0, CW - cw)); y0 = float(np.clip(fy - ch / 2, 0, CH - ch))
    return np.asarray(img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + cw, y0 + ch)), F32).copy()

# ---------------------------------------------------------------- scenes (real footage + real moons)
def sc_intro(c, u):
    return camera(c, lerp(1.0, 1.22, ease(u / 5.0)), 670, 1150)

def sc_d01(c, u):
    halo(c, 860, 640, 230, (70, 90, 140), 0.55)
    moon_screen(c, M01, 860, 640 + u * 6)
    return camera(c, lerp(1.0, 1.12, ease(u / 6.25)), 760, 1000)

def sc_d08(c, u):
    return camera(c, lerp(1.04, 1.18, ease(u / 6.25)), 600, 1000)

def sc_d17(c, u):
    halo(c, 880, 870, 210, (70, 90, 140), 0.5)
    moon_screen(c, M17, 880, 870)
    # rippled reflection under the true horizon (~y 1500)
    ry = 2130; h, w = REFL.shape[:2]; ys = np.arange(h)
    dx = (3 + 9 * ys / h) * np.sin(ys * 0.09 - u * 3.2) + 4 * np.sin(ys * 0.031 + u * 1.7)
    idx = np.clip(np.arange(w)[None, :] - dx[:, None], 0, w - 1).astype(int)
    rr = REFL[ys[:, None], idx] * (0.9 + 0.1 * np.sin(ys * 0.21 + u * 4))[:, None, None]
    y0 = ry - h // 2; x0 = 880 - w // 2; y1 = min(CH, y0 + h)
    reg = c[y0:y1, x0:x0 + w]; c[y0:y1, x0:x0 + w] = 255 - (255 - reg) * (255 - rr[: y1 - y0]) / 255
    z = lerp(1.0, 1.18, (u / 7.5) ** 1.6)
    return camera(c, z, 800, 1200)

def sc_d23(c, u):
    moon_over(c, M23, 630, 1152, 0.97)
    halo(c, 640, 1150, 600, (255, 210, 160), 0.16)
    c = camera(c, lerp(1.45, 1.0, eout(u / 3.2)) + 0.06 * ease((u - 3.2) / 4.3), 630, 1150)
    c += np.exp(-u * 4.5) * np.array([255, 238, 212], F32) * 0.95      # flash on the drop
    return c

def sc_d30(c, u):
    moon_screen(c, M30, 420, 520, gain=0.92)
    return camera(c, lerp(1.0, 1.12, ease(u / 6.25)), 560, 900)

def sc_outro(c, u):
    c = camera(c, lerp(1.0, 1.04, u / 6.0), 648, 1100)
    for i, mo in enumerate(MINI):
        k = eout((u - 0.35 - i * 0.3125) / 0.5)
        if k > 0:
            cx = 540 + (i - 2) * 196
            halo(c, cx, 700, 80, (255, 200, 130) if i == 3 else (110, 125, 175), 0.25 * k)
            moon_over(c, mo, cx, 700 + (1 - k) * 30, k)
            if i == 1:   # new moon: thin rim so the icon reads on the dark sky
                d = Image.new("L", (140, 140), 0); ImageDraw.Draw(d).ellipse([4, 4, 135, 135], outline=255, width=2)
                r = np.asarray(d.filter(ImageFilter.GaussianBlur(1)), F32)[..., None] / 255 * 0.45 * k
                y0 = int(700 + (1 - k) * 30) - 70; x0 = cx - 70
                c[y0:y0 + 140, x0:x0 + 140] += r * np.array([150, 165, 210], F32)
    for i, L in enumerate(MINI_NUM):
        draw_line(c, L, eout((u - 0.55 - i * 0.3125) / 0.4), 1.0)
    return c

SC = [sc_intro, sc_d01, sc_d08, sc_d17, sc_d23, sc_d30, sc_outro]

# real clouds for the transitions
CLOUDS = load_clip("clouds2", 1.6, (W, H)).astype(F32)
def clouds_frame(t, tc):
    i = int(np.clip((t - tc + 0.8) * FPS, 0, len(CLOUDS) - 1)); return CLOUDS[i]

def overlays(c, t):
    for tc, hw in ((5.0, 0.6), (32.5, 0.6)):          # pass through real clouds
        dt = t - tc
        if abs(dt) < hw:
            k = ease(np.clip(1.25 * (1 - abs(dt) / hw), 0, 1))
            cl = clouds_frame(t, tc) * 1.25
            c[:] = c * (1 - k) * (1 - 0.4 * k) + cl * k
    for tc, hw in ((11.25, 0.55), (17.5, 0.5)):        # darkness of the night sky
        dt = abs(t - tc)
        if dt < hw: c *= ease(dt / hw) ** 1.3
    if 24.0 < t < 25.0:                               # build: clouds rush in and light up before the drop
        k = ease((t - 24.0) / 1.0)
        cl = clouds_frame(t, 25.0)
        c[:] = c * (1 - k) + cl * 1.4 * k
        c += (k ** 3) * np.array([170, 150, 125], F32)
    dt = t - 38.75                                    # clouds + darkness into the outro
    if abs(dt) < 0.7:
        k = ease(np.clip(1.25 * (1 - abs(dt) / 0.7), 0, 1))
        c[:] = (c * (1 - k) + clouds_frame(t, 38.75) * k) * (1 - 0.75 * k)
    if t < 0.2: c *= ease(t / 0.2)
    if t > TOTAL - 1.0: c *= 1 - ease((t - (TOTAL - 1.0)) / 0.9)

def text(c, s, u, D):
    out_op = 1 - ease((u - (D - 0.5)) / 0.38) if s < 6 else 1.0
    if s == 0:
        draw_line(c, TITLE[0], eout((u - 0.65) / 0.9), out_op); draw_line(c, TITLE[1], eout((u - 1.25) / 0.8), out_op)
    elif s < 6:
        sl = 1 if s == 4 else 0          # the Superluna plate lands a beat later, on the drop
        td, tp, tout, tph = (0.55, 1.1, 3.2, 3.5) if sl else (0.3, 0.65, 2.5, 2.8)
        draw_line(c, TX[s][0], eout((u - td) / 0.6), out_op)
        draw_line(c, TX[s][1], eout((u - tp) / 0.8), out_op * (1 - ease((u - tout) / 0.35)))
        draw_line(c, PH[s][0], eout((u - tph) / 0.8), out_op)
        draw_line(c, PH[s][1], eout((u - tph - 0.45) / 0.8), out_op)
    else:
        draw_line(c, OUT1, eout((u - 2.0) / 0.8), 1); draw_line(c, OUT2, eout((u - 2.45) / 0.8), 1)
        draw_line(c, SIGN, eout((u - 3.1) / 1.3), 1)

def render(out_write, only=None):
    fi = 0
    for s in range(7):
        D = T[s + 1] - T[s]; n_end = int(round(T[s + 1] * FPS)) if s < 6 else NF
        want = only is None or any(T[s] <= x < T[s + 1] for x in only)
        clip = Clip(NAMES[s], D + 0.2) if want else None
        while fi < n_end:
            t = fi / FPS; u = t - T[s]
            if want:
                base = clip.next().astype(F32)
                if only is None or any(abs(t - x) < 0.5 / FPS for x in only):
                    c = SC[s](base, u); c *= VIG; text(c, s, u, D); overlays(c, t)
                    out_write(t, np.clip(c, 0, 255).astype(np.uint8))
            fi += 1

if __name__ == "__main__":
    if os.environ.get("PREVIEW"):
        ts = [float(x) for x in os.environ["PREVIEW"].split(",")]
        render(lambda t, a: Image.fromarray(a).save(os.path.join(HERE, f"q_{t:05.2f}.png")), only=ts)
        sys.exit()
    out, audio = sys.argv[1], sys.argv[2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio,
                           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    render(lambda t, a: ff.stdin.write(a.tobytes()))
    ff.stdin.close(); ff.wait(); print("done", out)
