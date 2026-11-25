"""Real-footage reel with typewriter text (no voice)."""
import numpy as np, subprocess, sys, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, '..', 'seg')
FONTS = os.path.join(HERE, '..', 'fonts')
CREAM = (255, 244, 228); GOLD = (242, 204, 140)
SERIF = 'Cormorant-MediumItalic.ttf'; SANS = 'Montserrat-SemiBold.ttf'

def sm(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0); return t * t * (3 - 2 * t)

_f = {}
def font(n, s):
    if (n, s) not in _f: _f[(n, s)] = ImageFont.truetype(os.path.join(FONTS, n), s)
    return _f[(n, s)]

# card: start, end, clip, lines[(text, font, size, y, color, tracking)], type_start, cps, band(y0,y1) for darkening
CUTS = [0, 4.5, 8.8, 13.1, 18.3, 23.5, 28.0, 35.0]
XF = 0.6
CLIPS = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
CARDS = [
    dict(lines=[('La Navidad', SERIF, 128, 860, CREAM, 0), ('se vive con el corazón', SERIF, 100, 1000, CREAM, 0)], t0=0.5, band=(700, 1150)),
    dict(lines=[('Abrazos que', SERIF, 120, 1330, CREAM, 0), ('reconfortan…', SERIF, 120, 1465, CREAM, 0)], t0=0.5, band=(1180, 1620)),
    dict(lines=[('…y miradas', SERIF, 120, 1330, CREAM, 0), ('cómplices', SERIF, 120, 1465, CREAM, 0)], t0=0.5, band=(1180, 1620)),
    dict(lines=[('Regala un momento', SERIF, 104, 1250, CREAM, 0), ('especial a quien', SERIF, 104, 1370, CREAM, 0), ('más quieres', SERIF, 104, 1490, CREAM, 0)], t0=0.45, band=(1100, 1640)),
    dict(lines=[('RESERVA TU SESIÓN', SANS, 46, 800, CREAM, 8), ('de Navidad', SERIF, 140, 945, CREAM, 0), ('con Rosa', SERIF, 140, 1100, GOLD, 0)], t0=0.45, band=(650, 1250)),
    dict(lines=[('Los huecos de', SERIF, 120, 880, CREAM, 0), ('diciembre vuelan', SERIF, 120, 1015, CREAM, 0)], t0=0.45, band=(730, 1170)),
    dict(lines=[('Feliz', SERIF, 170, 700, GOLD, 0), ('Navidad', SERIF, 210, 875, GOLD, 0),
                ('RESERVA TU SESIÓN CON ROSA', SANS, 42, 1100, CREAM, 7), ('Escríbele por mensaje directo', SERIF, 74, 1190, CREAM, 0)],
         t0=0.45, band=(520, 1330), pause_after=1, pause=0.5),
]
CPS = 17.0   # characters per second while typing
CHAR_FADE = 0.16

def layout(card):
    """positions of every character: list of (ch, x, y, fnt, color)"""
    out = []
    for text, fn, size, y, col, tr in card['lines']:
        f = font(fn, size)
        widths = [f.getlength(text[:i + 1]) - f.getlength(text[:i]) for i in range(len(text))]
        total = f.getlength(text) + tr * (len(text) - 1)
        x = (W - total) / 2
        line = []
        for i, ch in enumerate(text):
            xi = (W - total) / 2 + f.getlength(text[:i]) + tr * i
            line.append((ch, xi, y, f, col))
        out.append(line)
    return out

for c in CARDS:
    c['lay'] = layout(c)
    # char appearance times
    times = []; t = c['t0']
    for li, line in enumerate(c['lay']):
        for ch in line:
            times.append(t); t += (0 if ch[0] == ' ' else 1 / CPS) * (0.55 if line[0][3].path.endswith(SANS) else 1.0)
        t += 0.18
        if c.get('pause_after') == li: t += c['pause']
    c['times'] = times; c['tdone'] = t

def text_frame(card, tl, fade):
    """returns (rgba layer as float array premult, mask) for local time tl"""
    lay = Image.new('L', (W, H), 0); col = Image.new('RGB', (W, H), (0, 0, 0))
    dl = ImageDraw.Draw(lay); dc = ImageDraw.Draw(col)
    k = 0
    for line in card['lay']:
        for ch, x, y, f, c in line:
            ta = card['times'][k]; k += 1
            a = sm(ta, ta + CHAR_FADE, tl) * fade
            if a <= 0 or ch == ' ': continue
            dy = 10 * (1 - sm(ta, ta + CHAR_FADE * 1.5, tl))
            dl.text((x, y + dy), ch, font=f, fill=int(255 * a), anchor='ls')
            dc.text((x, y + dy), ch, font=f, fill=c, anchor='ls')
    m = np.asarray(lay).astype(np.float32) / 255
    sh = np.asarray(lay.filter(ImageFilter.GaussianBlur(7))).astype(np.float32) / 255
    rgb = np.asarray(col).astype(np.float32) / 255
    return m, sh, rgb

yy = np.arange(H, dtype=np.float32)[:, None]
xx = np.arange(W, dtype=np.float32)[None, :]
VIG = (1 - 0.45 * np.clip(np.sqrt(((xx / W - 0.5) * 1.1) ** 2 + ((yy / H - 0.48) * 0.8) ** 2) / 0.62, 0, 1) ** 2.2)[..., None]

class Reader:
    def __init__(self, path):
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
        self.last = None
    def next(self):
        b = self.p.stdout.read(W * H * 3)
        if len(b) == W * H * 3: self.last = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
        return self.last
    def close(self):
        self.p.stdout.close(); self.p.kill()

def main(out):
    N = int(CUTS[-1] * FPS)
    enc = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    readers = {}
    for fi in range(N):
        t = fi / FPS
        act = []
        for i in range(len(CLIPS)):
            a = CUTS[i] - (XF / 2 if i > 0 else 0); b = CUTS[i + 1] + (XF / 2 if i < len(CLIPS) - 1 else 0)
            if a <= t < b: act.append(i)
        for i in list(readers):
            if i not in act: readers.pop(i).close()
        for i in act:
            if i not in readers: readers[i] = Reader(os.path.join(SEG, CLIPS[i] + '.mp4'))
        frames = {i: readers[i].next() for i in act}
        if len(act) == 1:
            f = frames[act[0]].copy()
        else:
            i, j = act; w = sm(CUTS[j] - XF / 2, CUTS[j] + XF / 2, t)
            f = frames[i] * (1 - w) + frames[j] * w
        f = f * VIG
        # text of every active card
        for i in act:
            card = CARDS[i]; tl = t - CUTS[i]
            end = CUTS[i + 1] - CUTS[i]
            fade = 1.0 if i == len(CARDS) - 1 else 1 - sm(end - 0.55, end - 0.15, tl)
            if fade <= 0 or tl < 0: continue
            y0, y1 = card['band']
            bandk = sm(0, 0.6, tl) * fade
            g = np.exp(-(((yy - (y0 + y1) / 2) / ((y1 - y0) * 0.6)) ** 2)) * (0.55 if i in (0, 4, 5, 6) else 0.6)
            f *= (1 - g * bandk)[..., None] if g.ndim == 1 else (1 - g * bandk)[..., None].reshape(H, 1, 1)
            m, sh, rgb = text_frame(card, tl, fade)
            f *= (1 - 0.6 * sh)[..., None]
            f = f * (1 - m[..., None]) + rgb * m[..., None]
        if fi < 9: f = f * (fi / 9)
        enc.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
        if fi % 60 == 0: print(fi, flush=True)
    enc.stdin.close(); enc.wait()
    for r in readers.values(): r.close()

if __name__ == '__main__':
    for i, c in enumerate(CARDS):
        print(i, 'typed by', round(c['tdone'], 2), 's of', round(CUTS[i + 1] - CUTS[i], 2))
    if len(sys.argv) > 1: main(sys.argv[1])
