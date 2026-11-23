"""Render the testimonial reel (1080x1920, 60 fps) replicating the reference edit.

usage: python3 render.py <frame_start> <frame_end> <out.mp4>   (one chunk)
"""
import json, math, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = HERE
W, H, FPS = 1080, 1920, 60
BEAT = 60 / 70.0                      # music: "Love is Eternal" @ 70 BPM

# ---------------------------------------------------------------- timeline
T_INTRO_END = 6 * BEAT                # 5.143 s
SEG_BEATS = [8, 8, 8, 8, 9]
seg_starts = [T_INTRO_END]
for b in SEG_BEATS[:-1]:
    seg_starts.append(seg_starts[-1] + b * BEAT)
T_END = seg_starts[-1] + SEG_BEATS[-1] * BEAT     # card+bg fade to black ends here
FADE_OUT = 0.5
HOLD_BLACK = 1.2
DURATION = T_END + HOLD_BLACK
XF = 0.6                              # background cross-dissolve length

CLIPS = [  # file, crop centre (fraction of width) or None for native vertical, in-point
    ('100947.mp4', None, 0.0),
    ('4829.mp4', 0.40, 0.0),
    ('4661.mp4', 0.62, 0.0),
    ('44495.mp4', 0.47, 3.0),
    ('4624.mp4', 0.45, 2.0),
    ('101211.mp4', 0.53, 0.0),
]
bounds = [0.0] + seg_starts + [DURATION]   # clip i lives in [bounds[i], bounds[i+1]]

cards = json.load(open(os.path.join(HERE, 'shots', 'cards.json')))
CARD_KEYS = ['c1', 'c2', 'c3', 'c4', 'c5']
CARD_W = {'c1': 724, 'c2': 724, 'c3': 724, 'c4': 724, 'c5': 880}


def ease_out_cubic(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, a, b):
    if b <= a:
        return 1.0 if t >= b else 0.0
    return min(max((t - a) / (b - a), 0.0), 1.0)


# ---------------------------------------------------------------- backgrounds
class ClipReader:
    """Sequential reader: ffmpeg crops/scales to 1080x1920, we pick frames by time."""

    def __init__(self, path, cx, inpoint, t_open):
        info = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                        '-show_entries', 'stream=width,height,r_frame_rate',
                                        '-of', 'csv=p=0', path]).decode().strip().split(',')
        sw, sh = int(info[0]), int(info[1])
        n, d = info[2].split('/')
        self.fps = float(n) / float(d)
        if cx is None:
            vf = 'scale=1080:1920:flags=lanczos'
        else:
            cw = int(round(sh * 9 / 16 / 2) * 2)
            x0 = int(min(max(cx * sw - cw / 2, 0), sw - cw))
            vf = f'crop={cw}:{sh}:{x0}:0,scale=1080:1920:flags=lanczos,unsharp=5:5:0.55'
        self.ss = inpoint + max(0.0, t_open)
        self.proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{self.ss:.4f}', '-i', path,
                                      '-vf', vf + ',format=rgb24', '-f', 'rawvideo', '-'],
                                     stdout=subprocess.PIPE, bufsize=W * H * 3 * 2)
        self.idx = -1
        self.frame = None
        self.base = max(0.0, t_open)   # local clip time corresponding to first decoded frame

    def get(self, local_t):
        want = int(math.floor((local_t - self.base) * self.fps + 1e-6))
        want = max(want, 0)
        while self.idx < want:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break                      # clip ran out: hold last frame
            self.frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            self.idx += 1
        return self.frame


readers = {}


def clip_local_time(i, t):
    return t - (bounds[i] - XF / 2) if i > 0 else t


def bg_frame(i, t):
    if i not in readers:
        readers[i] = ClipReader(os.path.join(SCR, 'clips', CLIPS[i][0]), CLIPS[i][1], CLIPS[i][2],
                                clip_local_time(i, t))
    lt = clip_local_time(i, t)
    fr = readers[i].get(lt)
    # slow push-in (Ken Burns) over the life of the clip, sub-pixel smooth
    life = (bounds[i + 1] - bounds[i]) + XF
    z = 1.0 + 0.06 * ease_in_out(lt / life)
    im = Image.fromarray(fr)
    a = 1 / z
    ox = (W - W * a) / 2
    oy = (H - H * a) / 2
    return np.asarray(im.transform((W, H), Image.AFFINE, (a, 0, ox, 0, a, oy), resample=Image.BICUBIC),
                      dtype=np.float32)


def background(t):
    # which clip(s)
    for i in range(len(CLIPS)):
        if bounds[i] - XF / 2 <= t < bounds[i + 1] + XF / 2 or (i == len(CLIPS) - 1 and t >= bounds[i]):
            pass
    active = [i for i in range(len(CLIPS)) if (bounds[i] - (XF / 2 if i else 0)) <= t <= bounds[i + 1] + XF / 2]
    # close readers that are done
    for k in list(readers):
        if k not in active and t > bounds[k + 1] + XF / 2:
            readers[k].proc.kill()
            del readers[k]
    if len(active) == 1:
        return bg_frame(active[0], t)
    a, b = active[0], active[1]
    edge = bounds[b]
    m = ease_in_out(ramp(t, edge - XF / 2, edge + XF / 2))
    fa = bg_frame(a, t)
    fb = bg_frame(b, t)
    return fa * (1 - m) + fb * m


# ---------------------------------------------------------------- cards
SHADOW_PAD = 70
card_cache = {}


def card_assets(k):
    if k in card_cache:
        return card_cache[k]
    src = Image.open(os.path.join(HERE, 'shots', k + '.png')).convert('RGB')
    sw, sh = src.size
    cw = CARD_W[k]
    ch = int(round(sh * cw / sw))
    # shadow layer (static)
    sh_layer = Image.new('L', (cw + 2 * SHADOW_PAD, ch + 2 * SHADOW_PAD), 0)
    ImageDraw.Draw(sh_layer).rectangle((SHADOW_PAD - 6, SHADOW_PAD + 4, SHADOW_PAD + cw + 6, SHADOW_PAD + ch + 14), fill=200)
    sh_layer = sh_layer.filter(ImageFilter.GaussianBlur(26))
    card_cache[k] = dict(src=np.asarray(src, dtype=np.float32), size=(cw, ch), shadow=sh_layer,
                         hl=cards[k]['hl'], scale=cw / sw)
    return card_cache[k]


HL_COLOR = np.array([255, 212, 48], np.float32) / 255.0


def card_image(k, t_rel):
    """Card RGBA (with shadow) at card-local time t_rel (s since card landed)."""
    A = card_assets(k)
    src = A['src']
    img = src
    # highlighter sweeps: group g starts at 0.45 + 1.7*g, each line 0.38 s
    mask = None
    for g, group in enumerate(A['hl']):
        g0 = 0.45 + 1.7 * g
        for li, (x0, y0, x1, y1) in enumerate(group):
            p = ease_out_cubic(ramp(t_rel, g0 + 0.38 * li, g0 + 0.38 * (li + 1)))
            if p <= 0:
                continue
            if mask is None:
                mask = Image.new('L', (src.shape[1], src.shape[0]), 0)
                md = ImageDraw.Draw(mask)
            xe = x0 + (x1 - x0) * p
            md.rounded_rectangle((x0, y0 + 3, xe, y1 - 1), radius=8, fill=200)
    if mask is not None:
        m = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.2)), np.float32)[..., None] / 255.0
        img = src * (1 - m + m * HL_COLOR)
    cw, ch = A['size']
    face = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS)
    layer = Image.new('RGBA', (cw + 2 * SHADOW_PAD, ch + 2 * SHADOW_PAD), (0, 0, 0, 0))
    black = Image.new('RGBA', layer.size, (0, 0, 0, 255))
    layer = Image.composite(black, layer, A['shadow'])
    layer.paste(face, (SHADOW_PAD, SHADOW_PAD))
    return layer


def composite_layer(base, layer, cx, cy, scale=1.0, alpha=1.0, blur=0.0, wipe=None):
    """Alpha-composite RGBA layer centred at (cx,cy) onto float base."""
    if blur > 0.3:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    if abs(scale - 1) > 1e-4:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.BICUBIC)
    arr = np.asarray(layer, np.float32)
    rgb, a = arr[..., :3], arr[..., 3:] / 255.0 * alpha
    if wipe is not None:
        # soft-edged right-to-left wipe: wipe=0 -> fully visible, 1 -> gone
        xs = np.arange(arr.shape[1], dtype=np.float32)
        edge = (1 - wipe) * (arr.shape[1] + 120) - 60
        col = np.clip((edge - xs) / 60.0 + 0.5, 0, 1)
        a = a * col[None, :, None]
    x0 = int(round(cx - arr.shape[1] / 2))
    y0 = int(round(cy - arr.shape[0] / 2))
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(W, x0 + arr.shape[1]), min(H, y0 + arr.shape[0])
    if bx1 <= bx0 or by1 <= by0:
        return base
    sl = (slice(by0 - y0, by1 - y0), slice(bx0 - x0, bx1 - x0))
    region = base[by0:by1, bx0:bx1]
    base[by0:by1, bx0:bx1] = region * (1 - a[sl]) + rgb[sl] * a[sl]
    return base


def draw_cards(frame, t):
    for i, k in enumerate(CARD_KEYS):
        s0 = seg_starts[i]
        land = s0 + BEAT                    # card fully in exactly on the beat
        t_in0 = land - 0.45
        last = i == len(CARD_KEYS) - 1
        t_out0 = (seg_starts[i + 1] - 0.75) if not last else None
        if t < t_in0 or (not last and t > t_out0 + 0.56):
            continue
        layer = card_image(k, t - land)
        pin = ease_out_cubic(ramp(t, t_in0, land))
        hold = ramp(t, land, (seg_starts[i + 1] if not last else T_END))
        scale = (1.035 - 0.035 * pin) + 0.02 * ease_in_out(hold)
        blur = 16 * (1 - pin)
        alpha = pin
        wipe = None
        if not last and t >= t_out0:
            wipe = ease_in_out(ramp(t, t_out0, t_out0 + 0.55))
        frame = composite_layer(frame, layer, W / 2, H / 2, scale, alpha, blur, wipe)
    return frame


# ---------------------------------------------------------------- title
TRACK = 3.0


def _font(sz):
    f = ImageFont.truetype(os.path.join(SCR, 'fonts', 'Oswald.ttf'), sz)
    f.set_variation_by_name('Regular')
    return f


# size the type so line 1 matches the reference measurement (734 px wide @1080)
FONT_SIZE = 98
for _sz in range(110, 50, -1):
    FONT = _font(_sz)
    if sum(FONT.getlength(c) + TRACK for c in 'ALGUNOS DE NUESTROS') - TRACK + 12 <= 734:
        FONT_SIZE = _sz
        break
YEL = (246, 205, 18)
WHT = (253, 251, 250)
LINES = [
    [('ALGUNOS', WHT), ('DE', WHT), ('NUESTROS', WHT)],
    [('TESTIMONIOS', YEL, '!'), ('RESERVA', WHT), ('TU', WHT)],
    [('SESIÓN', YEL), ('CON', WHT), ('ROSA', WHT), ('HOY.', WHT)],
]
LINE_TOPS = [788, 920, 1052]


def text_w(s):
    return sum(FONT.getlength(c) + TRACK for c in s) - TRACK


def render_word(word):
    """Word layer RGBA (handles trailing white '!' after a yellow word)."""
    txt, col = word[0], word[1]
    extra = word[2] if len(word) > 2 else ''
    asc, desc = FONT.getmetrics()
    w = int(text_w(txt + extra)) + 40
    hgt = asc + desc + 40
    im = Image.new('RGBA', (w, hgt), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = 20
    for c in txt:
        d.text((x, 20), c, font=FONT, fill=col + (255,))
        x += FONT.getlength(c) + TRACK
    for c in extra:
        d.text((x, 20), c, font=FONT, fill=WHT + (255,))
        x += FONT.getlength(c) + TRACK
    return im


title_words = []   # (layer, shadow, x, y, t_start)


def build_title():
    space = FONT.getlength(' ') + TRACK + 6
    k = 0
    cap_top_offset = FONT.getbbox('H')[1]
    for li, line in enumerate(LINES):
        widths = [text_w(w[0] + (w[2] if len(w) > 2 else '')) for w in line]
        total = sum(widths) + space * (len(line) - 1)
        x = (W - total) / 2
        for wd, ww in zip(line, widths):
            lay = render_word(wd)
            sh = Image.new('RGBA', lay.size, (0, 0, 0, 0))
            sh.putalpha(lay.getchannel('A').point(lambda v: int(v * 0.55)))
            sh = sh.filter(ImageFilter.GaussianBlur(9))
            y = LINE_TOPS[li] - 20 - cap_top_offset
            title_words.append((lay, sh, x - 20, y, 0.30 + 0.115 * k, li))
            k += 1
            x += ww + space


build_title()


def draw_title(frame, t):
    if t > T_INTRO_END + XF / 2:
        return frame
    out_a = 1 - ease_in_out(ramp(t, T_INTRO_END - 0.35, T_INTRO_END + 0.15))
    out_blur = 10 * ease_in_out(ramp(t, T_INTRO_END - 0.35, T_INTRO_END + 0.15))
    for lay, sh, x, y, ts, li in title_words:
        p = ease_out_cubic(ramp(t, ts, ts + 0.32))
        if p <= 0:
            continue
        dy = (1 - p) * 70
        # reveal mask: word rises from below the line's bottom edge
        clip_bottom = LINE_TOPS[li] + int(FONT_SIZE * 0.95)
        for L, isshadow in ((sh, True), (lay, False)):
            arr = np.asarray(L if out_blur < 0.3 else L.filter(ImageFilter.GaussianBlur(out_blur)), np.float32)
            yy = int(round(y + dy)) + (4 if isshadow else 0)
            xx = int(round(x)) + (2 if isshadow else 0)
            rows = np.arange(arr.shape[0]) + yy
            vis = (rows <= clip_bottom).astype(np.float32)[:, None, None]
            a = arr[..., 3:] / 255.0 * vis * min(1.0, p * 1.6) * out_a
            h_, w_ = arr.shape[:2]
            y0, x0 = max(0, yy), max(0, xx)
            y1, x1 = min(H, yy + h_), min(W, xx + w_)
            sl = (slice(y0 - yy, y1 - yy), slice(x0 - xx, x1 - xx))
            frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - a[sl]) + arr[..., :3][sl] * a[sl]
    return frame


# ---------------------------------------------------------------- frame
def render_frame(t):
    if t >= T_END:
        return np.zeros((H, W, 3), np.uint8)
    f = background(t).copy()
    if t < T_INTRO_END + XF:
        # intro: darker, moodier grade for title legibility (reference look)
        g = 1 - 0.18 * (1 - ramp(t, T_INTRO_END - XF / 2, T_INTRO_END + XF / 2))
        f *= g
    f = draw_title(f, t)
    f = draw_cards(f, t)
    fade_in = ease_in_out(ramp(t, 0.0, 1.4))
    fade_out = 1 - ease_in_out(ramp(t, T_END - FADE_OUT, T_END))
    f *= fade_in * fade_out
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


def main():
    f0, f1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p',
                            '-profile:v', 'high', '-g', '120', out], stdin=subprocess.PIPE)
    for n in range(f0, f1):
        enc.stdin.write(render_frame(n / FPS).tobytes())
        if n % 120 == 0:
            print(f'[{out}] frame {n}', flush=True)
    enc.stdin.close()
    enc.wait()
    for r in readers.values():
        r.proc.kill()


if __name__ == '__main__':
    if sys.argv[1] == 'still':
        for ts in sys.argv[2:]:
            t = float(ts)
            Image.fromarray(render_frame(t)).save(os.path.join(HERE, 'stills', f'{t:06.2f}.png'))
            for r in list(readers.values()):
                r.proc.kill()
            readers.clear()
    elif sys.argv[1] == 'info':
        print('duration', DURATION, 'frames', int(round(DURATION * FPS)), 'segs', seg_starts, 'end', T_END)
    else:
        main()
