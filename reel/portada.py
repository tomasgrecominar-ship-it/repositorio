"""Portada del reel (1080x1920) para Neumáticos Ader.

Uso: python3 reel/portada.py  ->  reel/salida/portada.jpg
Lo importante queda dentro del cuadrado central (y 420 a 1500), que es lo que
muestra la grilla del perfil de Instagram.
"""
import os, sys
from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generar_reel as G
import assets as A

W, H = G.W, G.H


def main():
    c = G.kenburns("pila_neumaticos.jpg", (1.05, 1.05, 0.5, 0.5, 0.5, 0.5), 0, 1)
    c.alpha_composite(Image.new("RGBA", (W, H), G.RED_DARK + (150,)))
    c.alpha_composite(G.radial_light(540, 900, (230, 40, 46), 0.5))

    # rueda decorativa abajo, fuera del recorte de la grilla
    sz = 1300
    c.alpha_composite(A.spinning_wheel(sz, -18, 0), (540 - sz // 2, 1560))
    c.alpha_composite(A.wheel_light(sz), (540 - sz // 2, 1560))

    G.put(c, A.logo_ader(920), 540, 520)
    G.put(c, G.promo_box(), 540, 860)
    G.put(c, G.fit(G.shadowed("Neumáticos · Frenos", 62, G.WHITE), 960), 540, 1150)
    G.put(c, G.fit(G.shadowed("Tren delantero · Alineación", 62, G.WHITE), 960), 540, 1228)

    l2 = G.shadowed("Av. de Mayo 210, Villa Adelina", 50, G.WHITE)
    pin = G.pin_icon(42)
    tot = pin.width + 16 + l2.width
    x0 = 540 - tot / 2
    G.put(c, pin, x0 + pin.width / 2, 1380)
    G.put(c, l2, x0 + pin.width + 16 + l2.width / 2, 1380)

    # viñeta suave como en el video
    vig, _ = G.finish_layers()
    a = np.asarray(c.convert("RGB"), np.float32) * vig
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    os.makedirs(G.OUT_DIR, exist_ok=True)
    path = os.path.join(G.OUT_DIR, "portada.jpg")
    out.save(path, quality=94)
    print("ok", path)


if __name__ == "__main__":
    main()
