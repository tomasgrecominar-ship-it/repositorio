"""Reel vertical (1080x1920, 35 s, 30 fps) para Neumáticos Ader.

Uso (desde la raíz del repo):
    python3 reel/musica.py
    python3 reel/generar_reel.py            # video completo
    python3 reel/generar_reel.py --preview  # cuadros sueltos para revisar

Material real: si existen archivos en reel/material/ se usan en lugar de las
tomas generadas por código:
    toma_01 ... toma_09 (.jpg/.png/.mp4/.mov)  -> reemplazan las tomas del montaje
    logo.png (con transparencia)               -> reemplaza el logo provisorio
"""
import os, sys, glob, subprocess, math, shutil
from functools import lru_cache
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assets as A

W, H, FPS, DUR = 1080, 1920, 30, 35.0
NF = int(DUR * FPS)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "salida")
MAT_DIR = os.path.join(HERE, "material")
TMP_DIR = os.path.join(OUT_DIR, "_tmp_material")
RED = (214, 28, 34)
RED_DARK = (92, 6, 10)
YELLOW = RED  # acento (las tomas generadas también pasan a rojo)
WHITE = (255, 255, 255)
BLACK = (14, 14, 16)

# Textos editables -------------------------------------------------------
PROMO_1 = "3"
PROMO_2 = "CUOTAS"
PROMO_3 = "SIN INTERÉS"
CIERRE_1 = "¡Te esperamos!"
CIERRE_2 = "Av. de Mayo 210,"
CIERRE_3 = "Villa Adelina"
# Encuadre de cada foto real del montaje: (zoom inicial, zoom final,
# centro x inicial, centro y inicial, centro x final, centro y final), en
# fracciones de la imagen. Se usa para el movimiento lento de cámara.
ENCUADRES = {
    1: (1.00, 1.15, 0.50, 0.50, 0.45, 0.60),   # pila de neumáticos
    2: (1.18, 1.02, 0.45, 0.40, 0.55, 0.55),   # neumáticos al sol
    3: (1.30, 1.45, 0.55, 0.62, 0.45, 0.66),   # auto rojo en el taller
    4: (1.05, 1.22, 0.55, 0.55, 0.60, 0.62),   # desmontadora
    5: (1.32, 1.48, 0.50, 0.36, 0.52, 0.32),   # balanceadora
    6: (1.05, 1.20, 0.45, 0.45, 0.50, 0.40),   # auto en el elevador
    7: (1.55, 1.75, 0.22, 0.30, 0.30, 0.26),   # estantería con cubiertas
    9: (1.70, 1.90, 0.30, 0.78, 0.36, 0.74),   # banda de rodamiento de cerca
}
# Rótulos de las tomas del montaje (None = sin rótulo).
ROTULOS = ["NEUMÁTICOS", "PARA TU AUTO", "TALLER", "MONTAJE",
           "BALANCEO", "EN BUENAS MANOS", "STOCK", None, None]


# ------------------------------------------------------------- utilidades
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_back(x, s=1.7):
    x = clamp(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def prog(t, start, dur):
    return clamp((t - start) / dur)


@lru_cache(maxsize=None)
def text_img(text, weight, size, color, tracking=0):
    f = A.font(weight, size)
    tmp = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    if tracking:
        widths = [tmp.textlength(ch, font=f) for ch in text]
        tw = sum(widths) + tracking * (len(text) - 1)
    else:
        tw = tmp.textlength(text, font=f)
    asc, desc = f.getmetrics()
    pad = int(size * 0.08)
    img = Image.new("RGBA", (int(tw) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if tracking:
        x = pad
        for ch, w_ in zip(text, widths):
            d.text((x, pad), ch, font=f, fill=color + (255,))
            x += w_ + tracking
    else:
        d.text((pad, pad), text, font=f, fill=color + (255,))
    bb = img.getbbox()
    return img.crop((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad))


def with_alpha(img, a):
    if a >= 0.999:
        return img
    im = img.copy()
    im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
    return im


def put(canvas, img, cx, cy, scale=1.0, alpha=1.0):
    if alpha <= 0.003 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.002:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    img = with_alpha(img, alpha)
    canvas.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))


def put_reveal(canvas, img, cx, cy, p, alpha=1.0):
    """Texto que sube desde una máscara (p 0..1)."""
    if p <= 0:
        return
    p = ease_out(p)
    h = img.height
    off = int((1 - p) * h)
    piece = img.crop((0, 0, img.width, h - off)) if off > 0 else img
    if piece.height <= 0:
        return
    canvas.alpha_composite(with_alpha(piece, alpha), (int(cx - img.width / 2), int(cy - h / 2 + off)))


def new_canvas(color=BLACK):
    return Image.new("RGBA", (W, H), color + (255,))


def cover(img, cx_frac=0.5, cy_frac=0.5, zoom=1.0):
    """Recorta una imagen para cubrir el cuadro con zoom centrado en (cx,cy)."""
    s = max(W / img.width, H / img.height) * zoom
    cw, ch = W / s, H / s
    cx, cy = img.width * cx_frac, img.height * cy_frac
    x0 = clamp(cx - cw / 2, 0, img.width - cw)
    y0 = clamp(cy - ch / 2, 0, img.height - ch)
    cw, ch = min(cw, img.width), min(ch, img.height)
    x0, y0 = max(0.0, min(x0, img.width - cw)), max(0.0, min(y0, img.height - ch))
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, min(img.width, x0 + cw), min(img.height, y0 + ch)))


@lru_cache(maxsize=None)
def radial_bg(cx, cy, rad, col_in, col_out):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / rad
    t = np.clip(d, 0, 1)[..., None] ** 1.2
    a = np.array(col_in, np.float32) / 255
    b = np.array(col_out, np.float32) / 255
    return A.to_img(a * (1 - t) + b * t).convert("RGBA")


@lru_cache(maxsize=None)
def vgrad_overlay(y0, y1, a0, a1):
    al = np.zeros((H, W), np.float32)
    ys = np.arange(H, dtype=np.float32)
    col = np.clip((ys - y0) / (y1 - y0), 0, 1)
    al[:] = (a0 + (a1 - a0) * col)[:, None]
    return A.to_img(np.zeros((H, W, 3), np.float32), al)


def caption(canvas, text, t, dur):
    if not text:
        return
    p_in = prog(t, 0.15, 0.35)
    p_out = prog(t, dur - 0.05, 0.25)
    if p_in <= 0 or p_out >= 1:
        return
    img = text_img(text, "Black", 92, WHITE, 4)
    if img.width > 880:
        img = img.resize((880, int(img.height * 880 / img.width)), Image.LANCZOS)
    y = 1430
    bar_w = int((img.width + 60) * ease_out(p_in))
    a = 1 - p_out
    bar = Image.new("RGBA", (max(bar_w, 1), 14), (255, 40, 46, 255))
    canvas.alpha_composite(with_alpha(bar, a), (70, y + img.height // 2 + 12))
    put_reveal(canvas, img, 70 + img.width / 2, y, prog(t, 0.22, 0.35), a)


# ---------------------------------------------------------------- material
def find_material(name):
    for ext in (".mp4", ".mov", ".m4v", ".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join(MAT_DIR, name + ext)
        if os.path.exists(p):
            return p
    return None


def prepare_material():
    """Convierte videos reales a cuadros 1080x1920 (una sola vez)."""
    found = {}
    for i in range(1, 10):
        p = find_material(f"toma_{i:02d}")
        if not p:
            continue
        if p.lower().endswith((".mp4", ".mov", ".m4v")):
            d = os.path.join(TMP_DIR, f"toma_{i:02d}")
            if not os.path.isdir(d):
                os.makedirs(d)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-t", "4",
                                "-vf", f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
                                os.path.join(d, "%04d.jpg")], check=True)
            found[i] = ("video", d)
        else:
            found[i] = ("image", p)
    return found


MATERIAL = {}


def material_shot(slot):
    kind, path = MATERIAL[slot]
    z0, z1, x0, y0, x1, y1 = ENCUADRES.get(slot, (1.0, 1.12, 0.5, 0.5, 0.5, 0.5))

    def fn(t, dur):
        if kind == "image":
            img = graded_photo(path)
            e = ease_in_out(clamp(t / max(dur, 0.1)) * 0.85 + 0.15 * clamp(t / max(dur, 0.1)))
            c = cover(img, x0 + (x1 - x0) * e, y0 + (y1 - y0) * e, z0 + (z1 - z0) * e).convert("RGBA")
        else:
            frames = sorted(glob.glob(os.path.join(path, "*.jpg")))
            k = int(clamp(t * FPS, 0, len(frames) - 1))
            c = Image.open(frames[k]).convert("RGBA")
        c.alpha_composite(red_grad_bottom())
        caption(c, ROTULOS[slot - 1], t, dur)
        return c
    return fn


@lru_cache(maxsize=None)
def red_grad_bottom():
    al = np.zeros((H, W), np.float32)
    ys = np.arange(H, dtype=np.float32)
    al[:] = (np.clip((ys - 1050) / 870, 0, 1) ** 1.3 * 0.85)[:, None]
    top = np.clip((380 - ys) / 380, 0, 1) * 0.45
    al = np.maximum(al, top[:, None])
    return A.to_img(np.ones((H, W, 3), np.float32) * np.array(RED_DARK, np.float32) / 255, al)


@lru_cache(maxsize=None)
def graded_photo(p):
    """Foto real con corrección de color: contraste y dominante roja."""
    from PIL import ImageOps
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    lim = 2600
    if max(im.size) > lim:
        s = lim / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return A.to_img(grade_arr(np.asarray(im, np.float32) / 255))


def grade_arr(a):
    lum = (a @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    a = a * 0.62 + lum * 0.38                       # baja saturación
    a = np.clip((a - 0.5) * 1.18 + 0.5, 0, 1)       # contraste
    a = a * a * (3 - 2 * a) * 0.35 + a * 0.65        # curva en S suave
    red = np.array(RED, np.float32) / 255
    tint = lum * red * 1.55                           # sombras y medios en rojo
    return a * 0.7 + np.clip(tint, 0, 1) * 0.3


@lru_cache(maxsize=None)
def load_image(p):
    return Image.open(p).convert("RGB")


# ----------------------------------------------------------------- tomas
def wheel_angle(t, w0=1080.0, w1=300.0, k=1.3):
    """Ángulo (grados) con velocidad que baja de w0 a w1."""
    return -(w1 * t + (w0 - w1) * (1 - math.exp(-k * t)) / k), (w1 + (w0 - w1) * math.exp(-k * t))


def shot_intro(t, dur):
    c = radial_bg(540, 1500, 1500, (232, 40, 44), (78, 4, 8)).copy()
    ang, w = wheel_angle(max(t, 0))
    sz = 1360
    cx, cy = 540, 1660
    wheel_img = intro_wheel()
    if wheel_img is not None:
        wh = motion_blur_rot(wheel_img, ang, clamp((w - 150) / 700)).resize((sz, sz), Image.BICUBIC)
    else:
        wh = A.spinning_wheel(sz, ang, clamp((w - 150) / 700))
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((cx - sz // 2 - 10, cy - sz // 2 + 30, cx + sz // 2 + 10, cy + sz // 2 + 60), fill=(30, 0, 0, 150))
    c.alpha_composite(sh.filter(ImageFilter.GaussianBlur(40)))
    c.alpha_composite(wh, (cx - sz // 2, cy - sz // 2))
    c.alpha_composite(A.wheel_light(sz), (cx - sz // 2, cy - sz // 2))
    # logo arriba
    pl = prog(t, 0.15, 0.5)
    if pl > 0:
        put(c, A.logo_ader(760), 540, 190, 0.9 + 0.1 * ease_out(pl), ease_out(pl))
    # textos
    p3 = prog(t, 1.0, 0.45)
    if p3 > 0:
        s = 1 + 1.4 * (1 - ease_back(p3, 2.2))
        put(c, shadowed(PROMO_1, 560, WHITE), 540, 560, s, clamp(p3 * 3))
    put_reveal(c, shadowed(PROMO_2, 200, WHITE, 6), 540, 910, prog(t, 1.5, 0.4))
    pb = ease_out(prog(t, 2.0, 0.4))
    if pb > 0:
        txt = text_img(PROMO_3, "Black", 124, RED, 4)
        bw, bh = txt.width + 80, txt.height + 46
        band = Image.new("RGBA", (int(bw * pb) + 1, bh), WHITE + (255,))
        x0 = 540 - bw // 2
        c.alpha_composite(band, (x0, 1105 - bh // 2))
        if pb > 0.5:
            put_reveal(c, txt, 540, 1105, prog(t, 2.2, 0.35))
    # temblor de cámara en los golpes
    shk = 0
    for hit in (1.0, 2.0):
        if t >= hit:
            shk += 22 * math.exp(-(t - hit) * 12) * math.sin((t - hit) * 70)
    if abs(shk) > 0.5:
        c = c.transform(c.size, Image.AFFINE, (1, 0, 0, 0, 1, shk), Image.BICUBIC)
    z = 1 + 0.04 * clamp(t / dur)
    if z > 1.001:
        c = cover(c, 0.5, 0.5, z)
    return c


# Rueda real para la apertura: (archivo, centro x, centro y, radio) en píxeles
# de la foto original. None = se usa la rueda generada.
RUEDA_INTRO = None


@lru_cache(maxsize=None)
def intro_wheel():
    if not RUEDA_INTRO:
        return None
    from PIL import ImageOps
    f, cx, cy, r = RUEDA_INTRO
    im = ImageOps.exif_transpose(Image.open(os.path.join(MAT_DIR, f))).convert("RGB")
    crop = im.crop((cx - r, cy - r, cx + r, cy + r)).resize((1100, 1100), Image.LANCZOS)
    mask = Image.new("L", (4400, 4400), 0)
    ImageDraw.Draw(mask).ellipse((8, 8, 4392, 4392), fill=255)
    mask = mask.resize((1100, 1100), Image.LANCZOS)
    a = np.asarray(crop, np.float32) / 255
    a = np.clip((a - 0.5) * 1.12 + 0.5, 0, 1)
    out = A.to_img(a).convert("RGBA")
    out.putalpha(mask)
    return out


def motion_blur_rot(img, ang, amt):
    if amt < 0.05:
        return img.rotate(ang, Image.BICUBIC)
    n = 9
    spread = 30 * amt
    acc = np.zeros((img.height, img.width, 4), np.float32)
    for i in range(n):
        acc += np.asarray(img.rotate(ang + (i / (n - 1) - 0.5) * spread, Image.BICUBIC), np.float32)
    return Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8), "RGBA")


@lru_cache(maxsize=None)
def shadowed(text, size, color, tracking=0):
    """Texto con sombra suave para separarlo del fondo rojo."""
    t = text_img(text, "Black", size, color, tracking)
    pad = int(size * 0.12)
    out = Image.new("RGBA", (t.width + 2 * pad, t.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
    a = Image.new("L", out.size, 0)
    a.paste(t.getchannel("A"), (pad, pad + int(size * 0.03)))
    a = a.filter(ImageFilter.GaussianBlur(size * 0.04)).point(lambda v: int(v * 0.55))
    sh.putalpha(a)
    out.alpha_composite(sh)
    out.alpha_composite(t, (pad, pad))
    return out


@lru_cache(maxsize=None)
def tread_tall():
    tile = A.tread_texture(1300, 1400)
    tall = Image.new("RGB", (1300, 1400 * 4))
    for k in range(4):
        tall.paste(tile, (0, 1400 * k))
    return tall


def shot_tread(t, dur, angle=-12, speed=700, zoom=1.15):
    tall = tread_tall()
    y0 = int((t * speed) % 1400)
    crop = tall.crop((0, 1400 - y0 + 200, 1300, 1400 - y0 + 200 + 2600))
    crop = crop.rotate(angle + 3 * t, Image.BICUBIC)
    c = cover(crop, 0.5, 0.5, zoom + 0.05 * t).convert("RGBA")
    c.alpha_composite(radial_light(540, 500, (255, 220, 140), 0.18))
    c.alpha_composite(vgrad_overlay(1100, 1920, 0, 0.6))
    return c


@lru_cache(maxsize=None)
def radial_light(cx, cy, col, strength):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 600 ** 2)) * strength
    return A.to_img(np.ones((H, W, 3), np.float32) * np.array(col, np.float32) / 255, a)


@lru_cache(maxsize=None)
def warehouse_bg():
    c = radial_bg(540, 600, 1500, (52, 54, 60), (12, 12, 14)).copy()
    d = ImageDraw.Draw(c)
    for k in range(0, H, 140):
        d.line((0, k, W, k), fill=(255, 255, 255, 10), width=2)
    fl = A.vgradient(W, 560, (26, 27, 30), (8, 8, 9))
    c.alpha_composite(fl, (0, 1360))
    return c


def shot_stacks(t, dur, direction=1):
    c = warehouse_bg().copy()
    p = t / dur
    layers = [  # (pila, escala, x base, y piso, parallax, tono)
        (A.tire_stack(7, 260, 70, 120, 2, 3), 0.55, 160, 1380, 120, 0.55),
        (A.tire_stack(7, 260, 70, 120, 2, 4), 0.55, 900, 1380, 120, 0.55),
        (A.tire_stack(6, 260, 70, 120, 2, 5), 0.8, 760, 1560, 260, 0.8),
        (A.tire_stack(6, 260, 70, 120, 2, 6), 1.05, 260, 1840, 460, 1.0),
    ]
    for st, sc, x, yfloor, par, tone in layers:
        img = st.resize((int(st.width * sc), int(st.height * sc)), Image.BICUBIC)
        if tone < 1:
            dark = Image.new("RGBA", img.size, (12, 12, 14, int(255 * (1 - tone) * 0.8)))
            dark.putalpha(Image.fromarray((np.asarray(img.getchannel("A"), np.float32) * (1 - tone) * 0.8).astype(np.uint8)))
            img = img.copy()
            img.alpha_composite(dark)
        xx = x - direction * par * (p - 0.5) - img.width / 2
        c.alpha_composite(img, (int(xx), int(yfloor - img.height)))
    c.alpha_composite(radial_light(300, 300, (255, 230, 170), 0.12))
    c.alpha_composite(vgrad_overlay(1200, 1920, 0, 0.5))
    return c


def shot_garage(t, dur):
    g = A.garage_scene()
    z = 1.0 + 0.12 * ease_in_out(t / dur) + 0.02
    c = cover(g, 0.5, 0.47, z).convert("RGBA")
    c.alpha_composite(vgrad_overlay(1150, 1920, 0, 0.55))
    return c


@lru_cache(maxsize=None)
def hub_closeup():
    wh = A.wheel(1600)
    bg = Image.new("RGBA", wh.size, (20, 20, 22, 255))
    bg.alpha_composite(wh)
    cw, ch = 640, 640 * H / W
    box = (800 - cw / 2, 800 - ch / 2, 800 + cw / 2, 800 + ch / 2)
    return bg.resize((W, H), Image.LANCZOS, box=box), 1080 / 640


def shot_wrench(t, dur):
    base, s = hub_closeup()
    zoom = 1.0 + 0.05 * t
    c = cover(base, 0.5, 0.5, zoom).convert("RGBA")
    c.alpha_composite(A.wheel_light(H).crop((420, 0, 420 + W, H)) if False else radial_light(400, 600, (255, 240, 220), 0.15))
    # posición de la tuerca objetivo (en pantalla)
    R = 1600 / 2 * 0.985 * 0.64
    ang = -math.pi / 2 + math.pi / 5
    nx = 800 + math.cos(ang) * 0.135 * R
    ny = 800 + math.sin(ang) * 0.135 * R
    sx = W / 2 + (nx - 800) * s * zoom
    sy = H / 2 + (ny - 800) * s * zoom
    # llave de impacto que entra desde abajo a la derecha
    p = ease_out(prog(t, 0.05, 0.45))
    jit = 0
    if t > 0.5:
        jit = 4 * math.sin(t * 95) + 3 * math.sin(t * 61)
    gx = sx + (1 - p) * 600 + jit
    gy = sy + (1 - p) * 800 + jit * 0.6
    tool = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(tool)
    # movimiento alrededor de la tuerca
    if t > 0.5:
        for k in range(3):
            a0 = (t * 900 + k * 120) % 360
            r = 70 + k * 10
            d.arc((sx - r, sy - r, sx + r, sy + r), a0, a0 + 70, fill=(255, 255, 255, 120), width=6)
    # cubo (socket)
    d.ellipse((gx - 46, gy - 46, gx + 46, gy + 46), fill=(70, 72, 78, 255))
    d.ellipse((gx - 37, gy - 37, gx + 37, gy + 37), fill=(40, 42, 46, 255))
    d.polygon([(gx + 33, gy + 18), (gx + 180, gy + 156), (gx + 144, gy + 198), (gx - 6, gy + 48)], fill=(60, 62, 68, 255))
    # cuerpo de la herramienta
    d.rounded_rectangle((gx + 120, gy + 132, gx + 372, gy + 312), radius=42, fill=(38, 38, 42, 255))
    d.rounded_rectangle((gx + 138, gy + 150, gx + 360, gy + 198), radius=18, fill=YELLOW + (255,))
    d.rounded_rectangle((gx + 216, gy + 282, gx + 300, gy + 540), radius=30, fill=(30, 30, 34, 255))
    d.rounded_rectangle((gx + 225, gy + 312, gx + 237, gy + 528), radius=10, fill=(60, 60, 66, 255))
    sh = tool.filter(ImageFilter.GaussianBlur(25))
    sh_arr = np.asarray(sh.getchannel("A"), np.float32) * 0.6
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(Image.fromarray(sh_arr.astype(np.uint8)))
    c.alpha_composite(shadow, (20, 30))
    c.alpha_composite(tool)
    c.alpha_composite(vgrad_overlay(1150, 1920, 0, 0.5))
    return c


def shot_balance(t, dur):
    c = radial_bg(540, 900, 1100, (46, 50, 58), (10, 10, 12)).copy()
    d = ImageDraw.Draw(c)
    # cuerpo de la balanceadora
    d.rounded_rectangle((60, 1560, 1020, 2100), radius=50, fill=(54, 60, 70, 255))
    d.rounded_rectangle((60, 1560, 1020, 1590), radius=12, fill=YELLOW + (255,))
    d.rectangle((470, 1400, 610, 1570), fill=(80, 86, 96, 255))
    # panel con displays
    d.rounded_rectangle((150, 300, 930, 550), radius=30, fill=(24, 26, 30, 255), outline=(70, 74, 82, 255), width=4)
    for k, (lab, v0, delay) in enumerate((("IZQ", 35, 0.2), ("DER", 20, 0.35))):
        x0 = 200 + k * 370
        d.rounded_rectangle((x0, 335, x0 + 320, 480), radius=14, fill=(8, 8, 10, 255))
        val = int(round(v0 * (1 - ease_in_out(prog(t, delay, 1.1)))))
        ok = val == 0 and t > delay + 1.1
        col = (60, 230, 120) if ok else (255, 60, 50)
        txt = A.mono(100)
        d.text((x0 + 160, 408), f"{val:02d}", font=txt, fill=col, anchor="mm")
        d.text((x0 + 160, 515), lab, font=A.font("Bold", 34), fill=(160, 166, 176), anchor="mm")
    # rueda girando
    ang = -t * 1400
    sz = 840
    wh = A.spinning_wheel(sz, ang, 1.0)
    c.alpha_composite(wh, (540 - sz // 2, 1010 - sz // 2))
    c.alpha_composite(A.wheel_light(sz), (540 - sz // 2, 1010 - sz // 2))
    d = ImageDraw.Draw(c)
    d.ellipse((490, 960, 590, 1060), fill=(120, 126, 136, 255))  # tuerca rápida
    d.ellipse((515, 985, 565, 1035), fill=(60, 64, 70, 255))
    c.alpha_composite(vgrad_overlay(1150, 1920, 0, 0.5))
    z = 1.0 + 0.05 * t / dur
    return cover(c, 0.5, 0.5, z).convert("RGBA")


def shot_align(t, dur):
    c = radial_bg(540, 1050, 1100, (30, 34, 40), (8, 8, 10)).copy()
    d = ImageDraw.Draw(c)
    # guardabarros
    d.pieslice((-60, 360, 1140, 1560), 180, 360, fill=(40, 44, 52, 255))
    d.pieslice((0, 420, 1080, 1500), 180, 360, fill=(8, 8, 10, 255))
    sz = 1000
    wh = A.spinning_wheel(sz, -8 * t, 0)
    c.alpha_composite(wh, (540 - sz // 2, 1050 - sz // 2))
    c.alpha_composite(A.wheel_light(sz), (540 - sz // 2, 1050 - sz // 2))
    # láseres
    las = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dl = ImageDraw.Draw(las)
    green = (40, 255, 140)
    p = ease_out(prog(t, 0.1, 0.5))
    dl.line((540 - 540 * p, 1050, 540 + 540 * p, 1050), fill=green + (255,), width=5)
    dl.line((540, 1050 - 700 * p, 540, 1050 + 700 * p), fill=green + (255,), width=5)
    scan = 650 + 800 * ((t * 0.9) % 1.0)
    dl.line((80, scan, 1000, scan), fill=green + (160,), width=3)
    r = 150 + 10 * math.sin(t * 6)
    dl.ellipse((540 - r, 1050 - r, 540 + r, 1050 + r), outline=green + (255,), width=5)
    glow = las.filter(ImageFilter.GaussianBlur(10))
    c.alpha_composite(glow)
    c.alpha_composite(glow)
    c.alpha_composite(las)
    # lecturas
    d = ImageDraw.Draw(c)
    for k, (lab, v0, delay) in enumerate((("CAÍDA", -1.2, 0.3), ("CONVERGENCIA", 0.8, 0.45))):
        y0 = 320 + k * 180
        pa = ease_out(prog(t, delay - 0.2, 0.3))
        if pa <= 0:
            continue
        x0 = 120 - int((1 - pa) * 80)
        d.rounded_rectangle((x0, y0, x0 + 840, y0 + 150), radius=24, fill=(14, 16, 20, int(230 * pa)),
                            outline=(60, 66, 76, int(255 * pa)), width=3)
        val = v0 * (1 - ease_in_out(prog(t, delay, 1.0)))
        ok = abs(val) < 0.05 and t > delay + 1.0
        if abs(val) < 0.05:
            val = 0.0
        col = (60, 230, 120) if ok else (255, 199, 44)
        d.text((x0 + 40, y0 + 75), lab, font=A.font("Bold", 46), fill=(200, 204, 212, int(255 * pa)), anchor="lm")
        d.text((x0 + 800, y0 + 75), f"{val:+.1f}°", font=A.mono(72), fill=col + (int(255 * pa),), anchor="rm")
    c.alpha_composite(vgrad_overlay(1200, 1920, 0, 0.5))
    return c


def shot_wall(t, dur):
    wall = A.tire_wall()
    p = ease_in_out(t / dur) * 0.6 + 0.2 * (t / dur)
    c = cover(wall, 0.3 + 0.4 * p, 0.35 + 0.3 * p, 1.35).convert("RGBA")
    c.alpha_composite(vgrad_overlay(1150, 1920, 0, 0.55))
    return c


def shot_hero(t, dur):
    c = radial_bg(300, 900, 1400, (60, 50, 22), (8, 8, 10)).copy()
    ang, w = wheel_angle(t + 0.3, 900, 500, 2.0)
    sz = 2100
    wh = A.spinning_wheel(sz, ang, 0.85)
    x0, y0 = 760 - sz // 2, 980 - sz // 2
    c.alpha_composite(wh, (x0, y0))
    c.alpha_composite(A.wheel_light(sz), (x0, y0))
    return cover(c, 0.5, 0.5, 1 + 0.06 * t / dur).convert("RGBA")


def shot_reprise(t, dur):
    c = radial_bg(540, 700, 1400, (226, 36, 40), (110, 6, 10)).copy()
    d = ImageDraw.Draw(c)
    d.polygon([(0, 1250), (W, 1000), (W, 1120), (0, 1370)], fill=(150, 12, 18, 255))
    sz = 760
    cx_end = 540
    p = ease_out(prog(t, 0.0, 1.3))
    cx = -500 + (cx_end + 500) * p
    ang = -(cx - (-500)) / (sz / 2) * 180 / math.pi
    vel = (1 - prog(t, 0.0, 1.3)) ** 2
    cy = 1480
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse((cx - 360, cy + 330, cx + 360, cy + 410), fill=(20, 0, 0, 140))
    c.alpha_composite(sh.filter(ImageFilter.GaussianBlur(20)))
    wheel_img = intro_wheel()
    if wheel_img is not None:
        wh = motion_blur_rot(wheel_img, ang, vel).resize((sz, sz), Image.BICUBIC)
    else:
        wh = A.spinning_wheel(sz, ang, vel)
    c.alpha_composite(wh, (int(cx - sz / 2), cy - sz // 2))
    c.alpha_composite(A.wheel_light(sz), (int(cx - sz / 2), cy - sz // 2))
    put_reveal(c, shadowed("PAGÁ EN", 96, WHITE, 10), 540, 330, prog(t, 0.15, 0.35))
    p3 = prog(t, 0.3, 0.45)
    if p3 > 0:
        s = 1 + 0.8 * (1 - ease_back(p3, 2.0))
        put(c, shadowed("3 " + PROMO_2, 180, WHITE, 4), 540, 560, s, clamp(p3 * 3))
    pb = ease_out(prog(t, 0.6, 0.4))
    if pb > 0:
        txt = text_img(PROMO_3, "Black", 130, RED, 4)
        bw, bh = txt.width + 80, txt.height + 46
        band = Image.new("RGBA", (int(bw * pb) + 1, bh), WHITE + (255,))
        c.alpha_composite(band, (540 - bw // 2, 790 - bh // 2))
        if pb > 0.5:
            put_reveal(c, txt, 540, 790, prog(t, 0.75, 0.35))
    return c


@lru_cache(maxsize=None)
def end_logo():
    p = os.path.join(MAT_DIR, "logo.png")
    if os.path.exists(p):
        img = Image.open(p).convert("RGBA")
        s = min(960 / img.width, 520 / img.height)
        return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS), True
    return A.logo_ader(960), True


@lru_cache(maxsize=None)
def pin_icon(size=64):
    S = size * 4
    im = Image.new("RGBA", (S, int(S * 1.3)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((0, 0, S, S), fill=WHITE + (255,))
    d.polygon([(S * 0.12, S * 0.7), (S * 0.88, S * 0.7), (S / 2, S * 1.3)], fill=WHITE + (255,))
    d.ellipse((S * 0.32, S * 0.32, S * 0.68, S * 0.68), fill=RED + (255,))
    return im.resize((size, int(size * 1.3)), Image.LANCZOS)


def shot_end(t, dur):
    c = end_bg().copy()
    logo, real = end_logo()
    y_logo = 760
    pl = prog(t, 0.25, 0.6)
    if pl > 0:
        put(c, logo, 540, y_logo, 0.85 + 0.15 * ease_back(pl, 1.5), clamp(pl * 2.5))
    # línea blanca bajo el logo
    pln = ease_out(prog(t, 0.6, 0.5))
    if pln > 0:
        ln = Image.new("RGBA", (int(520 * pln) + 1, 8), WHITE + (255,))
        c.alpha_composite(ln, (540 - ln.width // 2, y_logo + 140))
    # textos del cierre (quedan fijos hasta el final)
    put_reveal(c, shadowed(CIERRE_1, 118, WHITE), 540, 1130, prog(t, 0.9, 0.45))
    l2 = shadowed(CIERRE_2, 80, WHITE)
    l3 = shadowed(CIERRE_3, 80, WHITE)
    pa = prog(t, 1.25, 0.45)
    if pa > 0:
        pin = pin_icon(60)
        tot = pin.width + 20 + l2.width
        x_pin = 540 - tot / 2 + pin.width / 2
        put(c, pin, x_pin, 1300, 1.0, ease_out(pa))
        put_reveal(c, l2, x_pin + pin.width / 2 + 20 + l2.width / 2, 1300, pa)
        put_reveal(c, l3, 540, 1405, prog(t, 1.4, 0.45))
    if 1.6 < t < 2.6:
        q = (t - 1.6) / 1.0
        sweep = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        x = -300 + 1700 * q
        ImageDraw.Draw(sweep).polygon([(x, 600), (x + 120, 600), (x - 80, 920), (x - 200, 920)],
                                      fill=(255, 255, 255, 60))
        c.alpha_composite(sweep.filter(ImageFilter.GaussianBlur(18)))
    return c


# Foto de fondo del cierre (archivo en reel/material/), None = solo color.
FONDO_CIERRE = None


@lru_cache(maxsize=None)
def end_bg():
    c = radial_bg(540, 900, 1500, (226, 36, 40), (96, 4, 8)).copy()
    if FONDO_CIERRE and os.path.exists(os.path.join(MAT_DIR, FONDO_CIERRE)):
        ph = cover(graded_photo(os.path.join(MAT_DIR, FONDO_CIERRE)), 0.5, 0.5, 1.0)
        a = np.asarray(ph, np.float32) / 255
        lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
        base = np.asarray(c.convert("RGB"), np.float32) / 255
        mixed = base * (0.55 + 0.75 * lum[..., None])  # textura sutil en rojo
        c = A.to_img(np.clip(mixed, 0, 1)).convert("RGBA")
    return c


# --------------------------------------------------------------- línea de tiempo
# (inicio, duración, función, rótulo, transición de entrada)
def build_timeline():
    montage = [
        (5.0, 2.0, shot_tread, "zoom"),
        (7.0, 2.0, shot_stacks, "push_up"),
        (9.0, 2.0, shot_garage, "fade"),
        (11.0, 1.5, shot_wrench, "push_left"),
        (12.5, 2.0, shot_balance, "zoom"),
        (14.5, 2.0, shot_align, "push_up"),
        (16.5, 1.5, shot_wall, "fade"),
        (18.0, 1.0, shot_hero, "push_left"),
        (19.0, 1.0, lambda t, d: shot_stacks(t, d, -1), "push_up"),
    ]
    tl = [(0.0, 5.0, shot_intro, "cut")]
    for i, (s, d, fn, tr) in enumerate(montage, start=1):
        if i in MATERIAL:
            f = material_shot(i)
        else:
            def f(t, dur, fn=fn, i=i):
                c = fn(t, dur)
                c = A.to_img(grade_arr(np.asarray(c.convert("RGB"), np.float32) / 255)).convert("RGBA")
                c.alpha_composite(red_grad_bottom())
                caption(c, ROTULOS[i - 1], t, dur)
                return c
        tl.append((s, d, f, tr))
    tl.append((20.0, 3.5, shot_reprise, "flash"))
    tl.append((23.5, 11.5, shot_end, "iris"))
    return tl


TL = None
TR_DUR = {"fade": 0.4, "zoom": 0.4, "push_up": 0.36, "push_left": 0.36, "flash": 0.3, "iris": 0.6, "cut": 0}


def render_shot(idx, t):
    s, d, fn, _ = TL[idx]
    return fn(t - s, d)


def arr(img):
    return np.asarray(img.convert("RGB"), np.float32)


def transition(kind, a_img, b_img, p):
    """p 0..1 dentro de la transición."""
    if kind == "fade":
        e = ease_in_out(p)
        return Image.blend(a_img, b_img, e)
    if kind == "zoom":
        e = ease_in_out(p)
        za = cover(a_img, 0.5, 0.5, 1 + 0.35 * e)
        zb = cover(b_img, 0.5, 0.5, 1 + 0.2 * (1 - e))
        return Image.blend(za, zb, e)
    if kind in ("push_up", "push_left"):
        e = ease_in_out(p)
        vel = 6 * p * (1 - p)  # proporcional a la velocidad
        A_, B_ = arr(a_img), arr(b_img)
        axis = 0 if kind == "push_up" else 1
        L = H if axis == 0 else W
        strip = np.concatenate([A_, B_], axis=axis)
        acc = np.zeros_like(A_)
        n = 5
        for k in range(n):
            off = int(clamp(e + (k / (n - 1) - 0.5) * 0.06 * vel, 0, 1) * L)
            acc += strip[off:off + L] if axis == 0 else strip[:, off:off + L]
        return Image.fromarray(np.clip(acc / n, 0, 255).astype(np.uint8)).convert("RGBA")
    if kind == "flash":
        base = a_img if p < 0.5 else b_img
        f = 1 - abs(p - 0.5) * 2
        white = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        return Image.blend(base, white, f ** 1.5 * 0.9)
    if kind == "iris":
        e = ease_in_out(p)
        r = 2300 * e
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse((540 - r, 1480 - r, 540 + r, 1480 + r), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(6))
        out = a_img.copy()
        out.paste(b_img, (0, 0), mask)
        return out
    return b_img


@lru_cache(maxsize=None)
def finish_layers():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.75)) ** 2)
    vig = 1 - 0.30 * np.clip(d - 0.45, 0, 1) ** 1.5
    rng = np.random.default_rng(11)
    grains = [rng.normal(0, 4.0, (H, W, 1)).astype(np.float32) for _ in range(6)]
    return vig[..., None], grains


def bug(canvas, t):
    """Recordatorio fijo durante el montaje."""
    a = prog(t, 5.3, 0.4) * (1 - prog(t, 19.75, 0.25))
    if a <= 0:
        return
    txt = text_img("3 CUOTAS SIN INTERÉS", "Black", 46, RED, 3)
    pill = Image.new("RGBA", (txt.width + 64, txt.height + 30), (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle((0, 0, pill.width - 1, pill.height - 1),
                                           radius=pill.height // 2, fill=WHITE + (255,))
    pill.alpha_composite(txt, (32, 15))
    put(canvas, pill, 540, 215 - 30 * (1 - ease_out(a)), 1.0, a)


def render_frame(i):
    t = i / FPS
    # tramo actual
    idx = max(k for k, sh in enumerate(TL) if sh[0] <= t + 1e-9)
    img = render_shot(idx, t)
    # transición de entrada (centrada en el corte)
    kind = TL[idx][3]
    td = TR_DUR.get(kind, 0)
    if td and t < TL[idx][0] + td / 2:
        p = (t - (TL[idx][0] - td / 2)) / td
        img = transition(kind, render_shot(idx - 1, t), img, p)
    # transición de salida (el siguiente tramo empieza antes)
    if idx + 1 < len(TL):
        nk = TL[idx + 1][3]
        ntd = TR_DUR.get(nk, 0)
        if ntd and t >= TL[idx + 1][0] - ntd / 2:
            p = (t - (TL[idx + 1][0] - ntd / 2)) / ntd
            img = transition(nk, img, render_shot(idx + 1, t), p)
    img = img.convert("RGBA")
    bug(img, t)
    vig, grains = finish_layers()
    a = np.asarray(img.convert("RGB"), np.float32) * vig + grains[i % len(grains)]
    if i < 6:  # entrada desde negro
        a *= i / 6
    return np.clip(a, 0, 255).astype(np.uint8)


def init_worker(material):
    global TL, MATERIAL
    MATERIAL = material
    TL = build_timeline()


def main():
    global TL, MATERIAL
    os.makedirs(OUT_DIR, exist_ok=True)
    MATERIAL = prepare_material()
    if MATERIAL:
        print("Material real encontrado para tomas:", sorted(MATERIAL))
    TL = build_timeline()

    if "--preview" in sys.argv:
        prev = os.path.join(OUT_DIR, "previews")
        os.makedirs(prev, exist_ok=True)
        times = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1:]] or \
            [0.5, 1.6, 3.0, 5.0, 6.0, 8.0, 10.0, 11.8, 13.6, 15.8, 17.2, 18.5, 19.5, 20.0, 21.6, 23.5, 25.0, 30.0, 34.9]
        for tt in times:
            Image.fromarray(render_frame(int(round(tt * FPS)))).save(os.path.join(prev, f"t{tt:05.2f}.jpg"), quality=88)
        print("previews en", prev)
        return

    audio = os.path.join(OUT_DIR, "musica.wav")
    if not os.path.exists(audio):
        subprocess.run([sys.executable, os.path.join(HERE, "musica.py")], check=True, cwd=os.path.dirname(HERE))
    out = os.path.join(OUT_DIR, "neumaticos_ader_reel.mp4")
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", audio,
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "12M", "-bufsize", "24M", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-level", "4.2",
           "-af", "volume=-4dB", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
           "-t", str(DUR), "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker, initargs=(MATERIAL,)) as pool:
        for k, fr in enumerate(pool.imap(render_frame, range(NF), chunksize=3)):
            ff.stdin.write(fr.tobytes())
            if k % 60 == 0:
                print(f"cuadro {k}/{NF}", flush=True)
    ff.stdin.close()
    ff.wait()
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    print("listo:", out)


if __name__ == "__main__":
    main()
