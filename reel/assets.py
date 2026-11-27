"""Elementos gráficos generados por código (neumáticos, llantas, taller)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from functools import lru_cache

FONT_DIR = "/usr/share/fonts/opentype/inter/"
MONO = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf"

YELLOW = (255, 199, 44)
BG = (14, 15, 17)


def font(weight, size):
    try:
        return ImageFont.truetype(FONT_DIR + f"InterDisplay-{weight}.otf", size)
    except OSError:
        return ImageFont.truetype(FONT_DIR + "Inter-Bold.otf", size)


def mono(size):
    try:
        return ImageFont.truetype(MONO, size)
    except OSError:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)


def to_img(rgb, alpha=None):
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    if alpha is None:
        return Image.fromarray(rgb, "RGB")
    a = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- texto polar
def _sidewall_text_strip(W, H, text_items):
    """Tira de texto para el flanco: se envuelve en polares."""
    strip = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(strip)
    for txt, pos, size in text_items:
        f = font("Black", size)
        bb = d.textbbox((0, 0), txt, font=f)
        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]
        d.text((pos * W - tw / 2 - bb[0], (H - th) / 2 - bb[1]), txt, font=f, fill=255)
    return np.asarray(strip, dtype=np.float32) / 255


# --------------------------------------------------------------- rueda frontal
@lru_cache(maxsize=None)
def wheel(size=1100, with_rim=True, ss=2):
    """Rueda vista de frente (neumático + llanta de aleación). RGBA."""
    S = size * ss
    y, x = np.mgrid[0:S, 0:S].astype(np.float32)
    c = (S - 1) / 2
    dx, dy = x - c, y - c
    R = S / 2 * 0.985
    r = np.sqrt(dx * dx + dy * dy) / R
    th = np.arctan2(dy, dx)
    rgb = np.zeros((S, S, 3), np.float32)
    alpha = smooth(1.0, 0.996, r)

    # --- neumático
    tire = r > 0.64
    v = 0.055 + 0.055 * np.clip(np.sin(np.pi * np.clip((r - 0.64) / 0.36, 0, 1)), 0, 1) ** 0.8
    # banda de rodamiento con tacos
    blocks = 56
    ph = (th / (2 * np.pi) * blocks) % 1.0
    notch = (r > 0.945) & (ph < 0.22)
    v = np.where(notch, 0.02, v)
    v = np.where((r > 0.94) & (r < 0.95), 0.03, v)
    v += np.where((r > 0.885) & (r < 0.895), 0.025, 0)  # línea de flanco
    v -= np.where((r > 0.645) & (r < 0.665), 0.02, 0)   # asiento de llanta
    # texto en el flanco
    W, Hs = 4096, 160
    strip = _sidewall_text_strip(W, Hs, [
        ("NEUMÁTICOS ADER", 0.15, 110), ("205/55 R16", 0.40, 84),
        ("NEUMÁTICOS ADER", 0.65, 110), ("TUBELESS", 0.90, 84)])
    r0, r1 = 0.735, 0.835
    band = (r > r0) & (r < r1)
    sx = (((th + np.pi) / (2 * np.pi)) * W).astype(np.int32) % W
    sy = np.clip(((r1 - r) / (r1 - r0) * Hs).astype(np.int32), 0, Hs - 1)
    txt = np.where(band, strip[sy, sx], 0)
    v = v + txt * 0.16
    rubber = np.dstack([v, v, v * 1.02])

    # --- llanta
    metal = np.zeros_like(rgb)
    rr = r / 0.64
    # labio de la llanta
    lip = np.clip(1 - np.abs(rr - 0.955) / 0.045, 0, 1)
    barrel = 0.10 + 0.05 * rr
    m = barrel.copy()
    # disco de freno visible detrás
    disc = (rr > 0.30) & (rr < 0.80)
    m = np.where(disc, 0.20 + 0.04 * np.sin(rr * 90), m)
    holes = disc & ((((th / (2 * np.pi)) * 36) % 1.0) < 0.18) & (np.abs(((rr - 0.30) * 20) % 1.0 - 0.5) < 0.25)
    m = np.where(holes, 0.06, m)
    # rayos: 5 dobles
    nsp = 5
    a = (th + np.pi / 2) % (2 * np.pi / nsp) - np.pi / nsp
    for off in (-0.13, 0.13):
        aa = a - off * np.clip(1.05 - rr, 0, 1)
        dist = np.abs(aa) * rr
        hw = 0.075 - 0.035 * rr
        sp = (rr > 0.18) & (rr < 0.93) & (dist < hw)
        bevel = np.clip(dist / hw, 0, 1)
        shade = 0.78 - 0.32 * bevel + 0.12 * np.sign(aa) * bevel
        m = np.where(sp, shade, m)
        # borde suave
        edge = (rr > 0.18) & (rr < 0.93) & (dist >= hw) & (dist < hw + 0.006)
        m = np.where(edge, 0.25, m)
    m = np.where(rr > 0.91, 0.45 + 0.4 * lip, m)
    # centro
    hub = rr < 0.22
    m = np.where(hub, 0.62 + 0.15 * (1 - rr / 0.22), m)
    # tuercas
    for k in range(5):
        ang = -np.pi / 2 + k * 2 * np.pi / 5 + np.pi / 5
        nx, ny = np.cos(ang) * 0.135, np.sin(ang) * 0.135
        d = np.sqrt((dx / R / 0.64 - nx) ** 2 + (dy / R / 0.64 - ny) ** 2)
        m = np.where(d < 0.032, 0.30 + 0.5 * (1 - d / 0.032) ** 2, m)
        m = np.where((d >= 0.032) & (d < 0.038), 0.18, m)
    cap = rr < 0.075
    m = np.where(cap, 0.85 - 0.25 * rr / 0.075, m)
    m = np.where((rr > 0.075) & (rr < 0.082), 0.3, m)
    metal = np.dstack([m * 0.95, m * 0.97, m * 1.0])
    if not with_rim:
        alpha = alpha * smooth(0.63, 0.645, r)
        metal = np.zeros_like(metal) + 0.02
    rim_mask = smooth(0.648, 0.638, r)[..., None]
    rgb = rubber * (1 - rim_mask) + metal * rim_mask
    img = to_img(rgb, alpha)
    if ss > 1:
        img = img.resize((size, size), Image.LANCZOS)
    if with_rim:
        # letra "A" en la tapa central
        d = ImageDraw.Draw(img)
        f = font("Black", int(size * 0.045))
        cx = size / 2
        d.text((cx, cx), "A", font=f, fill=(25, 25, 28, 255), anchor="mm")
    return img


@lru_cache(maxsize=None)
def wheel_blur(size=1100, with_rim=True, spread=0.55, n=18):
    """Versión con desenfoque de movimiento rotacional."""
    base = wheel(size, with_rim)
    acc = np.zeros((size, size, 4), np.float32)
    for i in range(n):
        ang = (i / (n - 1) - 0.5) * np.degrees(spread)
        acc += np.asarray(base.rotate(ang, Image.BICUBIC), np.float32)
    acc /= n
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8), "RGBA")


def spinning_wheel(size, angle_deg, blur_amt, with_rim=True):
    """Rueda rotada con mezcla entre nítida y desenfocada (0..1)."""
    base = wheel(1100, with_rim)
    if blur_amt > 0.02:
        b = wheel_blur(1100, with_rim)
        base = Image.blend(base, b, float(np.clip(blur_amt, 0, 1)))
    im = base.rotate(angle_deg, Image.BICUBIC)
    if size != 1100:
        im = im.resize((size, size), Image.LANCZOS if size < 1100 else Image.BICUBIC)
    return im


@lru_cache(maxsize=None)
def wheel_light(size):
    """Brillo fijo (no gira) para dar volumen a la rueda."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    dx, dy = (x - c) / c, (y - c) / c
    r = np.sqrt(dx * dx + dy * dy)
    spec = np.exp(-((dx + 0.35) ** 2 + (dy + 0.45) ** 2) / 0.18) * 0.22
    rim = np.exp(-((r - 0.86) ** 2) / 0.002) * np.clip(-(dx + dy) * 0.7, 0, 1) * 0.35
    a = (spec + rim) * (r < 0.985)
    rgb = np.ones((size, size, 3), np.float32)
    return to_img(rgb, a)


# ---------------------------------------------------------- neumático acostado
def lying_tire_arr(a, b, h, tone=1.0):
    """Neumático acostado visto de costado. Devuelve RGBA numpy (alto, ancho)."""
    W = int(2 * a + 4)
    Ht = int(2 * b + h + 4)
    y, x = np.mgrid[0:Ht, 0:W].astype(np.float32)
    cx, cy = W / 2, b + 2
    u = (x - cx) / a
    inside_u = np.abs(u) <= 1
    s = np.sqrt(np.clip(1 - u * u, 0, 1))
    yfront = cy + b * s
    top = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
    hole = ((x - cx) / (a * 0.6)) ** 2 + ((y - cy) / (b * 0.6)) ** 2 <= 1
    bandm = inside_u & (y >= cy) & (y <= yfront + h) & ~top
    bandm |= inside_u & (y > yfront) & (y <= yfront + h)
    v = np.zeros_like(x)
    # banda: cilindro con surcos
    phi = np.arcsin(np.clip(u, -1, 1))
    lug = ((phi / np.pi * 34) % 1.0) < 0.28
    yy = (y - yfront) / h
    groove_c = (np.abs(yy - 0.36) < 0.035) | (np.abs(yy - 0.64) < 0.035)
    vb = 0.05 + 0.10 * s ** 1.5
    vb = np.where((lug & (yy > 0.12) & (yy < 0.88)) | groove_c, vb * 0.35, vb)
    vb *= 1 - 0.35 * smooth(0.75, 1.0, yy)
    v = np.where(bandm, vb, v)
    # cara superior (flanco)
    rr = np.sqrt(((x - cx) / a) ** 2 + ((y - cy) / b) ** 2)
    vt = 0.06 + 0.07 * np.sin(np.pi * np.clip((rr - 0.6) / 0.4, 0, 1)) + 0.03 * (-(y - cy) / b)
    v = np.where(top, vt, v)
    # agujero central: pared interior oscura
    vh = 0.015 + 0.03 * np.clip((y - cy) / b + 0.6, 0, 1)
    v = np.where(hole, vh, v)
    alpha = (top | bandm).astype(np.float32)
    v *= tone
    rgb = np.dstack([v, v, v * 1.03])
    return rgb, alpha


@lru_cache(maxsize=None)
def tire_stack(n=6, a=260, b=70, h=120, ss=2, seed=0):
    a2, b2, h2 = a * ss, b * ss, h * ss
    rgb1, al1 = lying_tire_arr(a2, b2, h2)
    Wt, Ht = rgb1.shape[1], rgb1.shape[0]
    Htot = Ht + (n - 1) * h2
    rgb = np.zeros((Htot, Wt, 3), np.float32)
    alpha = np.zeros((Htot, Wt), np.float32)
    rr = np.random.default_rng(seed)
    for i in range(n):
        y0 = Htot - Ht - i * h2
        dxo = int(rr.integers(-8, 9)) * ss
        tone = 0.85 + 0.3 * rr.random()
        r_i = np.roll(rgb1 * tone, dxo, axis=1)
        a_i = np.roll(al1, dxo, axis=1)[..., None]
        sl = slice(y0, y0 + Ht)
        rgb[sl] = rgb[sl] * (1 - a_i) + r_i * a_i
        alpha[sl] = np.maximum(alpha[sl], a_i[..., 0])
    img = to_img(rgb, alpha)
    return img.resize((Wt // ss, Htot // ss), Image.LANCZOS)


# ---------------------------------------------------------------- textura banda
@lru_cache(maxsize=None)
def tread_texture(W=1300, H=1400):
    """Banda de rodamiento direccional (se repite verticalmente)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W / 2
    u = (x - cx) / (W * 0.46)            # -1..1 ancho de banda
    au = np.abs(u)
    period = H / 7
    # surcos en V direccionales
    p = ((y + au * 260) / period) % 1.0
    vg = (p < 0.16) & (au > 0.08) & (au < 0.98)
    # surcos circunferenciales
    cg = (np.abs(au - 0.08) < 0.035) | (np.abs(au - 0.52) < 0.03)
    # laminillas
    p2 = ((y + au * 260) / (period / 3)) % 1.0
    sipe = (p2 < 0.03) & (au > 0.12) & (au < 0.9) & ~vg
    curv = np.sqrt(np.clip(1 - (u * 0.92) ** 2, 0, 1))
    v = 0.09 + 0.10 * curv
    # borde iluminado de cada taco (luz rasante desde arriba)
    edge = (p >= 0.16) & (p < 0.21) & (au > 0.08) & (au < 0.98)
    v = np.where(edge, v + 0.07, v)
    v = np.where(vg | cg, 0.02 + 0.02 * curv, v)
    v = np.where(sipe, v * 0.45, v)
    # hombros
    sh = au > 0.98
    v = np.where(sh, 0.05 * curv + 0.02, v)
    v *= np.where(au > 1.08, 0.3, 1)
    grain = np.random.default_rng(3).standard_normal((H, W)).astype(np.float32) * 0.006
    v = v + grain
    rgb = np.dstack([v, v, v * 1.03])
    return to_img(rgb)


# ---------------------------------------------------------------- taller
def glow_rect(img, box, color, blur=40, strength=1.0):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=8, fill=color + (int(255 * strength),))
    g = layer.filter(ImageFilter.GaussianBlur(blur))
    img.alpha_composite(g)
    img.alpha_composite(layer)


def vgradient(w, h, top, bottom):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    a = np.array(top, np.float32)[None, None] / 255
    b = np.array(bottom, np.float32)[None, None] / 255
    arr = a * (1 - t) + b * t
    arr = np.broadcast_to(arr, (h, w, 3))
    return to_img(arr).convert("RGBA")


@lru_cache(maxsize=None)
def garage_scene(W=1400, H=2300):
    img = vgradient(W, H, (30, 33, 38), (18, 19, 22))
    d = ImageDraw.Draw(img)
    floor_y = int(H * 0.70)
    # portón levadizo al fondo
    dx0, dx1 = int(W * 0.12), int(W * 0.88)
    dy0 = int(H * 0.22)
    d.rectangle((dx0, dy0, dx1, floor_y), fill=(44, 48, 54))
    for yy in range(dy0, floor_y, 26):
        d.line((dx0, yy, dx1, yy), fill=(34, 37, 42), width=4)
        d.line((dx0, yy + 6, dx1, yy + 6), fill=(54, 58, 64), width=2)
    d.rectangle((dx0 - 24, dy0 - 24, dx1 + 24, dy0), fill=(26, 28, 32))
    # piso
    fl = vgradient(W, H - floor_y, (40, 42, 46), (12, 12, 14))
    img.alpha_composite(fl, (0, floor_y))
    d = ImageDraw.Draw(img)
    # franjas de seguridad
    sy = floor_y + int((H - floor_y) * 0.62)
    for k in range(-2, 22):
        x0 = k * 90
        d.polygon([(x0, sy), (x0 + 45, sy), (x0 + 85, sy + 46), (x0 + 40, sy + 46)], fill=YELLOW + (255,))
    d.rectangle((0, sy + 46, W, sy + 60), fill=(20, 20, 22))
    # columnas del elevador
    for px in (int(W * 0.10), int(W * 0.84)):
        d.rectangle((px, int(H * 0.18), px + 70, floor_y + 30), fill=(232, 178, 30))
        d.rectangle((px + 10, int(H * 0.18), px + 22, floor_y + 30), fill=(250, 210, 90))
        d.rectangle((px - 20, floor_y + 20, px + 90, floor_y + 44), fill=(60, 62, 66))
    # auto sobre el elevador (silueta)
    cy = int(H * 0.47)
    body = [(W * 0.13, cy + 40), (W * 0.16, cy - 50), (W * 0.30, cy - 80), (W * 0.40, cy - 190),
            (W * 0.66, cy - 195), (W * 0.78, cy - 90), (W * 0.88, cy - 70), (W * 0.89, cy + 40)]
    d.polygon([(int(px), int(py)) for px, py in body], fill=(52, 56, 64))
    d.polygon([(int(W * 0.42), cy - 175), (int(W * 0.52), cy - 175), (int(W * 0.52), cy - 92),
               (int(W * 0.33), cy - 92)], fill=(88, 104, 120))
    d.polygon([(int(W * 0.545), cy - 175), (int(W * 0.645), cy - 175), (int(W * 0.745), cy - 92),
               (int(W * 0.545), cy - 92)], fill=(88, 104, 120))
    d.line((int(W * 0.15), cy - 30, int(W * 0.87), cy - 30), fill=(70, 75, 84), width=6)
    d.rectangle((int(W * 0.12), cy + 40, int(W * 0.9), cy + 62), fill=(40, 42, 46))  # brazos
    for wx in (0.27, 0.74):
        wsz = 260
        wh = wheel(1100).resize((wsz, wsz), Image.LANCZOS)
        img.alpha_composite(wh, (int(W * wx - wsz / 2), int(cy - 20 - wsz / 2 + 60)))
    d = ImageDraw.Draw(img)
    # pilas de neumáticos al costado
    st = tire_stack(5, 150, 40, 70, 2, 1)
    img.alpha_composite(st, (int(W * 0.0) - 40, floor_y + 60 - st.size[1] + 80))
    st2 = tire_stack(4, 150, 40, 70, 2, 2)
    img.alpha_composite(st2, (int(W * 0.80), floor_y + 60 - st2.size[1] + 100))
    # luces lineales
    for k, lx in enumerate((0.2, 0.5, 0.8)):
        x0 = int(W * lx - 150)
        glow_rect(img, (x0, int(H * 0.08), x0 + 300, int(H * 0.08) + 18), (255, 250, 235), 50, 1.0)
    # reflejo de luces en el piso
    refl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(refl)
    for lx in (0.2, 0.5, 0.8):
        dr.ellipse((int(W * lx - 120), floor_y + 120, int(W * lx + 120), floor_y + 220), fill=(255, 250, 235, 40))
    img.alpha_composite(refl.filter(ImageFilter.GaussianBlur(30)))
    return img


@lru_cache(maxsize=None)
def tire_wall(W=2000, H=2900):
    img = vgradient(W, H, (34, 36, 40), (16, 17, 19))
    t = wheel(1100, with_rim=False).resize((340, 340), Image.LANCZOS)
    rows = 9
    for rI in range(rows):
        y = 60 + rI * 320
        off = 0 if rI % 2 == 0 else 160
        for cI in range(-1, 7):
            x = off + cI * 320
            img.alpha_composite(t, (x, y - 30))
        d = ImageDraw.Draw(img)
        # estante
        d.rectangle((0, y + 296, W, y + 316), fill=(222, 168, 34))
        d.rectangle((0, y + 316, W, y + 326), fill=(90, 70, 20))
    # luz superior suave
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    light = np.exp(-((x - W * 0.4) ** 2 / (W * 0.5) ** 2 + (y - H * 0.3) ** 2 / (H * 0.5) ** 2)) * 0.22
    ov = to_img(np.ones((H, W, 3), np.float32) * np.array([1, 0.97, 0.9]), light)
    img.alpha_composite(ov)
    return img


# -------------------------------------------------------------------- logo
@lru_cache(maxsize=None)
def logo_wordmark(scale=1.0):
    """Logotipo provisorio de texto (se reemplaza con material/logo.png)."""
    f_big = font("Black", int(260 * scale))
    f_small = font("Bold", int(64 * scale))
    tmp = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), "ADER", font=f_big)
    w_big = bb[2] - bb[0]
    h_big = bb[3] - bb[1]
    small = "NEUMÁTICOS"
    track = int(30 * scale)
    ws = sum(d.textlength(ch, font=f_small) for ch in small) + track * (len(small) - 1)
    W = int(max(w_big, ws) + 40 * scale)
    gap = int(30 * scale)
    bar = int(16 * scale)
    H = int(90 * scale + gap + h_big + gap + bar + 20 * scale)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = (W - ws) / 2
    for ch in small:
        d.text((x, 0), ch, font=f_small, fill=(255, 255, 255, 255))
        x += d.textlength(ch, font=f_small) + track
    y_big = int(90 * scale + gap)
    d.text(((W - w_big) / 2 - bb[0], y_big - bb[1]), "ADER", font=f_big, fill=YELLOW + (255,))
    yb = y_big + h_big + gap
    d.rectangle(((W - w_big) / 2, yb, (W + w_big) / 2, yb + bar), fill=(255, 255, 255, 255))
    return img


# ------------------------------------------------------- logo Neumáticos Ader
LOGO_RED = (226, 38, 34)


@lru_cache(maxsize=None)
def logo_ader(width=900, shadow=True):
    """Reconstrucción en alta resolución del logo (cursiva gruesa,
    NEUMATICOS blanco y ADER rojo, con contorno negro)."""
    size = 400
    f = font("Black", size)
    s_out = int(size * 0.06)   # contorno negro
    s_in = int(size * 0.035)    # filete blanco de ADER
    d0 = ImageDraw.Draw(Image.new("L", (4, 4)))
    w1 = d0.textlength("NEUMATICOS", font=f)
    gap = size * 0.03
    w2 = d0.textlength("ADER", font=f)
    pad = s_out * 2 + 40
    Wc = int(w1 + gap + w2 + 2 * pad)
    Hc = int(size * 1.3 + 2 * pad)
    x1, x2, y = pad, pad + w1 + gap, pad

    def layer(draw_fn):
        im = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
        draw_fn(ImageDraw.Draw(im))
        return im

    out = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0))
    black = (16, 14, 20, 255)
    out.alpha_composite(layer(lambda d: (d.text((x1, y), "NEUMATICOS", font=f, fill=black, stroke_width=s_out, stroke_fill=black),
                                         d.text((x2, y), "ADER", font=f, fill=black, stroke_width=s_out, stroke_fill=black))))
    out.alpha_composite(layer(lambda d: d.text((x2, y), "ADER", font=f, fill=(255, 255, 255, 255),
                                               stroke_width=s_in, stroke_fill=(255, 255, 255, 255))))
    out.alpha_composite(layer(lambda d: (d.text((x1, y), "NEUMATICOS", font=f, fill=(255, 255, 255, 255)),
                                         d.text((x2, y), "ADER", font=f, fill=LOGO_RED + (255,)))))
    # condensar y poner en cursiva
    cond = 0.93
    out = out.resize((int(Wc * cond), Hc), Image.LANCZOS)
    sh = 0.17
    W2 = int(out.width + Hc * sh)
    out = out.transform((W2, Hc), Image.AFFINE, (1, sh, -Hc * sh, 0, 1, 0), Image.BICUBIC)
    if shadow:
        a = out.getchannel("A").filter(ImageFilter.GaussianBlur(14))
        sh_img = Image.new("RGBA", out.size, (0, 0, 0, 0))
        sh_img.putalpha(a.point(lambda v: int(v * 0.55)))
        base = Image.new("RGBA", out.size, (0, 0, 0, 0))
        base.alpha_composite(sh_img, (10, 16))
        base.alpha_composite(out)
        out = base
    out = out.crop(out.getbbox())
    s = width / out.width
    return out.resize((width, int(out.height * s)), Image.LANCZOS)
