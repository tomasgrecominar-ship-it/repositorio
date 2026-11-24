#!/usr/bin/env python3
"""Reel cover 1080x1920; key content kept inside the centred 1080x1350 grid crop."""
import os, sys, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw
B = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.insert(0, B)
import render as R

W, H = 1080, 1920
# background: wheel splashing through pothole, 9:16 crop
raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', '9.3', '-i', B + 'clips/16983.mp4', '-frames:v', '1',
                      '-vf', 'crop=trunc(ih*9/16/2)*2:ih:trunc((iw-ih*9/16)*0.42/2)*2:0,scale=1080:1920:flags=lanczos,unsharp=5:5:0.5,format=bgr24',
                      '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
bg = np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()
bg = R.grade(bg, 0, 0.0)
# deep red duotone towards the top, footage kept at the bottom
yy = np.linspace(0, 1, H)[:, None, None]
k = np.clip(1.15 - yy * 1.25, 0.25, 0.92)
bg = R.red_plate(bg, 0.0)
lum = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None] / 255.0
tint = np.clip(R.PLATE * (0.62 + 0.48 * lum), 0, 255)
bg = np.clip(bg * (1 - k) + tint * k, 0, 255).astype(np.uint8)
# bottom darkening so the footage reads as photo, not flat
dark = np.clip((yy - 0.62) / 0.38, 0, 1) * 0.55
bg = (bg * (1 - dark)).astype(np.uint8)

frame = bg.copy()
def put(img, x, y, anchor='c'):
    el = R.El(img, x, y, -1, 99, anim='none')
    el.anchor = anchor
    R.blend(frame, el, 0)

# layout (grid-safe zone: y 285..1635)
put(R.lockup(1.0, 700), W / 2, 420)
rows = [('¿TUS CUBIERTAS', 118, R.WHITE), ('AGUANTAN', 178, R.WHITE), ('UN POZO', 178, R.WHITE), ('MÁS?', 250, R.RED)]
imgs = []
for txt, size, col in rows:
    f = R.anton(size)
    while R.text_width(txt, f) > 860:
        size -= 6; f = R.anton(size)
    imgs.append(R.text_img(txt, f, col, shadow=True, d3=max(7, size // 20)))
y = 540
for im in imgs:
    put(im, W / 2, y + im.size[1] / 2)
    y += im.size[1] - 18
chip = R.pill('3 SEÑALES PARA CAMBIARLAS', R.mont(42, 900), R.WHITE, R.RED, 40, 24, tracking=2)
put(chip, W / 2, y + 60)
print('chip y', y + 60)
cv2.imwrite(B + 'work/portada_reel.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
# previews: full + grid crop (3:4 and 4:5)
g34 = frame[(H - 1440) // 2:(H + 1440) // 2]
g45 = frame[(H - 1350) // 2:(H + 1350) // 2]
prev = np.hstack([cv2.resize(frame, (540, 960)),
                  np.full((960, 20, 3), 255, np.uint8),
                  cv2.copyMakeBorder(cv2.resize(g34, (540, 720)), 120, 120, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
                  np.full((960, 20, 3), 255, np.uint8),
                  cv2.copyMakeBorder(cv2.resize(g45, (540, 675)), 142, 143, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))])
cv2.imwrite(B + 'qc/cover_prev.jpg', prev)
print('ok')
