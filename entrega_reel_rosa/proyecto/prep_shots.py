"""Prepare censored testimonial screenshots + highlight boxes (source pixel coords)."""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

U = 'capturas_originales/'  # original screenshots go here
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shots')
os.makedirs(OUT, exist_ok=True)


def censor(im, box, shape='rect', radius=14):
    """Irreversible censor: mosaic to a few blocks, then heavy blur, soft mask."""
    x0, y0, x1, y1 = box
    reg = im.crop(box)
    w, h = reg.size
    small = reg.resize((max(1, w // 28), max(1, h // 28)), Image.BILINEAR)
    mos = small.resize((w, h), Image.NEAREST).filter(ImageFilter.GaussianBlur(18))
    mask = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(mask)
    if shape == 'circle':
        d.ellipse((2, 2, w - 3, h - 3), fill=255)
    else:
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    im.paste(mos, (x0, y0), mask)


cards = {}

# --- Rosario 1 -------------------------------------------------------------
im = Image.open(U + '14f5ea96-image.jpg').convert('RGB')
censor(im, (196, 146, 292, 242), 'circle')          # profile photo
censor(im, (304, 156, 604, 212))                      # contact name
im.save(f'{OUT}/c1.png')
cards['c1'] = dict(hl=[
    [(350, 718, 648, 768), (60, 786, 404, 838)],      # me ha dicho / que me quiere
    [(258, 1398, 566, 1452)],                         # felices juntos
])

# --- Rosario 2 (remove "Tapa mi foto y nombre" bubble) ---------------------
im = Image.open(U + 'fd36b57e-image.jpg').convert('RGB')
censor(im, (196, 146, 292, 242), 'circle')
censor(im, (304, 156, 604, 212))
a = np.array(im)
a[1495:1524, 830:920] = a[1395:1424, 830:920]   # erase scroll button remnant
a = np.concatenate([a[:1524], a[1638:]], axis=0)   # drop the private-request bubble rows
im = Image.fromarray(a)
im.save(f'{OUT}/c2.png')
cards['c2'] = dict(hl=[
    [(400, 878, 672, 932), (60, 948, 480, 1000)],     # te lo devo a / ti y a tus hipnosis
    [(390, 1015, 638, 1067), (60, 1083, 420, 1134)],  # tu hipnosis / funciona 100%
])

# --- Monica 1 ----------------------------------------------------------------
im = Image.open(U + 'ed0575b7-image.jpg').convert('RGB')
censor(im, (200, 148, 296, 244), 'circle')
censor(im, (304, 160, 452, 222))
im.save(f'{OUT}/c3.png')
cards['c3'] = dict(hl=[
    [(60, 556, 500, 606), (60, 623, 636, 672)],       # momentos de más / complicidad y cariño
    [(266, 826, 704, 876), (60, 893, 486, 945)],      # noto que está más / cercano conmigo
])

# --- Monica 2 ----------------------------------------------------------------
im = Image.open(U + 'd20c8d61-image.jpg').convert('RGB')
censor(im, (200, 148, 296, 244), 'circle')
censor(im, (304, 152, 452, 200))                      # name (keep "Cuenta de empresa")
im.save(f'{OUT}/c4.png')
cards['c4'] = dict(hl=[
    [(60, 1188, 656, 1240)],                          # La verdad que súper bien
    [(60, 1523, 546, 1575)],                          # Le veo más cariñoso
])

# --- Third client (cropped below the header) ---------------------------------
im = Image.open(U + '2d943d2d-image.jpg').convert('RGB')
a = np.array(im)
# remove the bottom of the "Hoy" pill that straddles the crop line: copy a clean
# strip (same rows) from further right where only wallpaper/bubble edge exists
a[150:192, 500:680] = a[150:192, 720:900]
im = Image.fromarray(a[150:])
censor(im, (262, 24, 494, 256), 'circle')             # voice-note avatar (y-150)
im.save(f'{OUT}/c5.png')
cards['c5'] = dict(hl=[
    [(72, 716, 770, 778)],                            # Un poco flipando estoy
    [(434, 1132, 892, 1192), (72, 1218, 412, 1278)],  # el compromiso / va más alla
])

for k, v in cards.items():
    w, h = Image.open(f'{OUT}/{k}.png').size
    v['size'] = [w, h]
json.dump(cards, open(f'{OUT}/cards.json', 'w'), indent=1)
print(json.dumps(cards))
