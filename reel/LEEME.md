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
| `logo.png` (fondo transparente) | el logo provisorio del cierre |

Los textos y los rótulos de cada toma se editan al principio de `generar_reel.py`.
