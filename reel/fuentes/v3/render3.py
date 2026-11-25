"""Couples Christmas reel: AI photos animated with camera moves + light overlays"""
import numpy as np, sys, math, subprocess, os, json
from PIL import Image, ImageFilter
import render as R
import render2 as R2
from render import W, H, FPS, sm, glow, blit, text_layer, draw_text, xx, yy

S = os.environ.get('S', '')
CHOICE = json.load(open(os.path.join(os.path.dirname(__file__), 'choice.json')))
OVER = 1.14  # base image oversize factor

def load(name, blur=0, grade=1.0):
    im = Image.open(f"{S}/img/{CHOICE[name]}.png").convert('RGB')
    bw, bh = int(W * OVER), int(H * OVER)
    # cover-fit
    sc = max(bw / im.width, bh / im.height)
    im = im.resize((int(im.width * sc + 0.5), int(im.height * sc + 0.5)), Image.LANCZOS)
    l, t = (im.width - bw) // 2, (im.height - bh) // 2
    im = im.crop((l, t, l + bw, t + bh))
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=40, threshold=2))
    if blur: im = im.filter(ImageFilter.GaussianBlur(blur))
    return im

def ease(u): return u * u * (3 - 2 * u)

class Photo:
    def __init__(self, name, z=(1.0, 1.08), pan=(0, 0), petals=0, hearts=0, dust=40, dim=0.0, blur=0, seed=0):
        self.im = load(name, blur)
        self.z = z; self.pan = pan; self.dim = dim
        self.petals = R2.Petals(seed + 1, petals, vel=(12, 55), blurfrac=0.6) if petals else None
        self.hb = R2.HBokeh(seed + 2, hearts, 40, 120, 0.22, (3, -14), hearts=0.7) if hearts else None
        self.dust = R2.Dust(seed + 3, dust, inten=0.9) if dust else None
    def base(self, t, d):
        u = ease(min(max(t / d, 0), 1))
        s = self.z[0] + (self.z[1] - self.z[0]) * u
        cw, ch = W * OVER / s / OVER * 1.0, H * OVER / s / OVER * 1.0
        cw, ch = W / s, H / s
        cx = self.im.width / 2 + self.pan[0] * (u - 0.5)
        cy = self.im.height / 2 + self.pan[1] * (u - 0.5)
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        fr = self.im.resize((W, H), Image.BICUBIC, box=box)
        return np.asarray(fr).astype(np.float32) / 255
    def __call__(self, t, d):
        f = self.base(t, d)
        if self.dim: f *= (1 - self.dim)
        add = np.zeros_like(f)
        if self.hb: self.hb.draw(add, t)
        if self.dust: self.dust.draw(add, t)
        f = 1 - (1 - f) * (1 - np.clip(add, 0, 1))  # screen blend
        if self.petals: self.petals.draw(f, t)
        return f

class Titled(Photo):
    def __init__(self, name, items_list, timings, **kw):
        super().__init__(name, **kw)
        self.tls = [text_layer(it) for it in items_list]; self.tm = timings
        self.cy = sum(i[3] for it in items_list for i in it) / sum(len(it) for it in items_list)
    def __call__(self, t, d):
        f = super().__call__(t, d)
        dimc = np.exp(-(((xx - W / 2) / 560) ** 2 + ((yy - self.cy) / 360) ** 2)).astype(np.float32)
        f *= (1 - 0.45 * sm(self.tm[0][0], self.tm[0][1], t) * dimc)[..., None]
        old = R.TEXTDIM; R.TEXTDIM = np.zeros_like(old)
        for tl, (a, b) in zip(self.tls, self.tm):
            k = sm(a, b, t); draw_text(f, tl, k, dy=int(25 * (1 - k)))
        R.TEXTDIM = old
        return f

def mk():
    return [
        lambda: Photo('s1', z=(1.12, 1.0), pan=(0, 0), hearts=14, dust=50, seed=1),
        lambda: Photo('s2', z=(1.0, 1.07), pan=(-30, 0), dust=30, seed=2),
        lambda: Photo('s3', z=(1.02, 1.08), pan=(40, 0), dust=30, seed=3),
        lambda: Photo('s4', z=(1.0, 1.08), pan=(0, -40), dust=40, petals=5, seed=4),
        lambda: Photo('s5', z=(1.0, 1.09), pan=(0, 30), dust=40, petals=6, seed=5),
        lambda: Titled('s6', [[('RESERVA TU SESIÓN', 'Montserrat-SemiBold.ttf', 40, 270, 9, 0),
                               ('Navidad', 'Cormorant-MediumItalic.ttf', 180, 405, 0, 1),
                               ('con Rosa', 'Cormorant-MediumItalic.ttf', 115, 560, 0, 1)]], [(0.2, 1.3)],
                       z=(1.0, 1.06), dim=0.15, dust=50, petals=6, hearts=8, seed=6),
        lambda: Photo('s7', z=(1.0, 1.07), pan=(0, -50), dust=40, seed=7),
        lambda: Photo('s8', z=(1.0, 1.08), pan=(30, 0), dust=40, hearts=10, seed=8),
        lambda: Titled('s9', [[('con todo el cariño', 'Cormorant-MediumItalic.ttf', 64, 520, 0, 1)],
                              [('Feliz', 'Cormorant-MediumItalic.ttf', 170, 650, 0, 1),
                               ('Navidad', 'Cormorant-MediumItalic.ttf', 210, 820, 0, 1)],
                              [('RESERVA TU SESIÓN CON ROSA', 'Montserrat-SemiBold.ttf', 40, 1040, 7, 0),
                               ('Escríbele por mensaje directo', 'Cormorant-MediumItalic.ttf', 66, 1125, 0, 1)]],
                       [(0.1, 1.0), (0.6, 1.6), (1.6, 2.5)], z=(1.06, 1.0), dim=0.38, blur=9, dust=60, petals=8, hearts=10, seed=9),
    ]

CUTS = [0, 3.0, 4.45, 5.8, 8.25, 13.0, 17.4, 21.5, 26.0, 32.0]
XF = [0.6, 0.4, 0.45, 0.6, 0.6, 0.6, 0.6, 0.7]
SHOTS = mk()

def finish(f, fi):
    f = f * (0.55 + 0.45 * R.vign)
    f = f * np.array([1.03, 0.99, 0.94], np.float32)
    g = np.random.default_rng(fi).normal(0, 0.007, (H // 2, W // 2)).astype(np.float32)
    f = f + np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)

def shot_frame(inst, i, t):
    s0, s1 = CUTS[i], CUTS[i + 1]
    lead = XF[i - 1] / 2 if i > 0 else 0
    d = (s1 - s0) + lead + (XF[i] / 2 if i < len(XF) else 0)
    return finish(inst[i](t - s0 + lead, d), int(t * FPS) + i * 1000)

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
        if fi < 8: img = (img.astype(np.float32) * (fi / 8)).astype(np.uint8)
        p.stdin.write(img.tobytes())
        if fi % 30 == 0: print(out, fi, flush=True)
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
