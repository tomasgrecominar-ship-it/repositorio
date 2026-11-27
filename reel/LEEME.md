# Reel Neumáticos Ader

Video final: `salida/neumaticos_ader_reel.mp4` (1080×1920, 30 fps, 35 s, H.264 + AAC).

## Regenerar
```
pip install numpy pillow scipy
python3 reel/musica.py          # música original 120 BPM
python3 reel/generar_reel.py    # video (unos 4 minutos)
```

## Usar fotos o videos reales del local
Copiá archivos en `reel/material/` y volvé a correr `generar_reel.py`:

| Archivo | Reemplaza |
|---|---|
| `toma_01` … `toma_09` (.jpg/.png/.mp4/.mov) | las tomas del montaje, en orden (5 s a 20 s) |
| `logo.png` (fondo transparente) | la reconstrucción del logo (`assets.logo_ader`) |

Los textos y los rótulos de cada toma se editan al principio de `generar_reel.py`.

## Fotos actuales
Las tomas 1 a 7 y 9 son fotos de Wikimedia Commons con licencia libre, con tinte rojo.
Autores y licencias en `material/CREDITOS.md`: las CC BY piden mencionar al autor en la publicación.
Para usar fotos del local, reemplazá los `toma_XX.jpg` y ajustá `ENCUADRES` en `generar_reel.py`.
