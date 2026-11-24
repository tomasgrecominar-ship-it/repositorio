#!/usr/bin/env python3
"""High-res redraw of the NEUMATICOSADER wordmark (white + red, dark outline, 3D extrusion)."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

B = os.path.dirname(os.path.abspath(__file__)) + '/'
LOGO_RED = (214, 10, 28)
OUTLINE = (22, 16, 26)
WHITE = (255, 255, 255)


def build(width=2400, rim=True):
    s = 300  # font size at working scale
    f = ImageFont.truetype(B + 'fonts/Kanit-BlackItalic.ttf', s)
    parts = [('NEUMATICOS', WHITE), ('ADER', LOGO_RED)]
    track = -6
    stroke = int(s * 0.055)
    depth = int(s * 0.07)
    tw = sum(sum(f.getlength(c) for c in t) + track * (len(t) - 1) for t, _ in parts) + track
    asc, desc = f.getmetrics()
    pad = stroke * 3 + depth + 40
    Wc, Hc = int(tw + 2 * pad), int(asc + desc + 2 * pad)

    def layer(fill_fn, stroke_w):
        im = Image.new('RGBA', (Wc, Hc), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        x = pad
        for t, col in parts:
            for c in t:
                fc = fill_fn(col)
                d.text((x, pad), c, font=f, fill=fc, stroke_width=stroke_w, stroke_fill=fc if stroke_w else None)
                x += f.getlength(c) + track
            x += track
        return im

    # silhouette (fill + outline) for extrusion / rim
    sil = layer(lambda c: (0, 0, 0, 255), stroke)
    a_sil = np.array(sil)[..., 3]

    out = Image.new('RGBA', (Wc, Hc), (0, 0, 0, 0))
    if rim:  # thin light rim around everything (incl. extrusion) for pop on red/dark backgrounds
        ext = np.zeros_like(a_sil, dtype=np.float32)
        for k in range(depth + 1):
            ext = np.maximum(ext, np.roll(np.roll(a_sil, k, 0), int(k * 0.6), 1))
        rim_mask = Image.fromarray(ext.astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
        rim_im = Image.new('RGBA', (Wc, Hc), (255, 255, 255, 255))
        rim_im.putalpha(rim_mask)
        out.alpha_composite(rim_im)
    # extrusion (dark, slightly lighter towards the back)
    for k in range(depth, 0, -1):
        sh = Image.new('RGBA', (Wc, Hc), (12, 8, 14, 255))
        m = np.roll(np.roll(a_sil, k, 0), int(k * 0.6), 1)
        sh.putalpha(Image.fromarray(m))
        out.alpha_composite(sh)
    # outline
    ol = Image.new('RGBA', (Wc, Hc), OUTLINE + (255,))
    ol.putalpha(sil.split()[3])
    out.alpha_composite(ol)
    # fills with subtle vertical gradient
    fill = layer(lambda c: c + (255,), 0)
    fa = np.array(fill).astype(np.float32)
    g = np.linspace(1.04, 0.90, Hc)[:, None, None]
    fa[..., :3] = np.clip(fa[..., :3] * g, 0, 255)
    out.alpha_composite(Image.fromarray(fa.astype(np.uint8)))
    # extra italic shear to match the original's slant
    shear = 0.10
    out = out.transform((int(Wc + Hc * shear), Hc), Image.AFFINE, (1, shear, -Hc * shear, 0, 1, 0), Image.BICUBIC)
    out = out.crop(out.getbbox())
    r = width / out.size[0]
    return out.resize((width, int(out.size[1] * r * 1.09)), Image.LANCZOS)


if __name__ == '__main__':
    lg = build(2400)
    lg.save(B + 'logo/logo_neumaticosader.png')
    bg = Image.new('RGBA', (lg.size[0] + 200, lg.size[1] + 200), (165, 30, 22, 255))
    bg.alpha_composite(lg, (100, 100))
    bg.convert('RGB').resize((bg.size[0] // 2, bg.size[1] // 2), Image.LANCZOS).save(B + 'logo/preview.jpg')
    print(lg.size)
