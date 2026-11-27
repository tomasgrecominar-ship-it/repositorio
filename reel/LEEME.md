# Reel Neumáticos Ader

Video final: `salida/neumaticos_ader_reel.mp4` (1080×1920, 30 fps, 35 s, H.264 + AAC).

| Tiempo | Escena |
|---|---|
| 0–5 s | Logo, neumático girando y «3 CUOTAS SIN INTERÉS» |
| 5–7 s | Toma de neumáticos |
| 7–9 s | «Nuestros servicios» |
| 9–21,5 s | 5 servicios, 2,5 s cada uno, con texto grande |
| 21,5–35 s | Cierre: logo, 3 cuotas sin interés, horario y dirección |

## Regenerar
```
pip install numpy pillow scipy
python3 reel/musica.py          # música original pop-rock, 120 BPM
python3 reel/generar_reel.py    # video (unos 5 minutos)
python3 reel/generar_reel.py --preview   # cuadros sueltos en salida/previews
```

## Cambiar fotos y textos
Todo se configura al principio de `generar_reel.py`:
- `SERVICIOS`: texto, foto y encuadre de cada servicio. Si falta la foto
  (hoy `frenos.jpg` y `llantas.jpg`), se usa una animación generada.
- `INTRO_TOMAS`, `FONDO_TITULO`, `FONDO_CIERRE`: fotos del resto del video.
- `HORARIO`, `DIRECCION`: textos del cierre.
- `reel/material/logo.png` (fondo transparente) reemplaza la reconstrucción del logo.

Las fotos actuales son de Wikimedia Commons: ver `material/CREDITOS.md`.

## Portada
`python3 reel/portada.py` genera `salida/portada.jpg` (1080×1920). Lo importante queda en el cuadrado central para que se vea bien en la grilla del perfil.
