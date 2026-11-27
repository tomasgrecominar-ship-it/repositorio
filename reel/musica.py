"""Música original para el reel: pop-rock enérgico en Re mayor, 120 BPM, 35 s.

Sintetizada con numpy/scipy (batería, bajo, guitarras saturadas y punteo).
Genera reel/salida/musica.wav. Los acentos coinciden con los cortes del video:
1.0 s (aparece el "3"), 5.0 s (primera toma), 7.0 s ("Nuestros servicios"),
9.0 s y cada 2.5 s (un servicio) y 21.5 s (cierre con el logo).
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
import wave, os

SR = 44100
DUR = 35.0
BPM = 120
BEAT = 60 / BPM
N = int(SR * DUR)
rng = np.random.default_rng(21)

HITS = [1.0, 5.0, 7.0, 9.0, 21.5]          # golpes fuertes (cortes principales)
SOFT_HITS = [11.5, 14.0, 16.5, 19.0]       # cambio de servicio
FINAL = 33.0


def t_arr(d):
    return np.arange(int(SR * d)) / SR


def filt(x, kind, f, order=2):
    return sosfilt(butter(order, f, btype=kind, fs=SR, output="sos"), x)


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def note(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# ------------------------------------------------------------------ batería
def kick():
    t = t_arr(0.4)
    f = 50 + 130 * np.exp(-t * 35)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    s += filt(rng.standard_normal(len(t)), "band", [1500, 5000]) * np.exp(-t * 250) * 0.4
    return np.tanh(s * 2.0)


def snare():
    t = t_arr(0.3)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25) * 0.6
    n = filt(rng.standard_normal(len(t)), "band", [1800, 9000]) * np.exp(-t * 16)
    return np.tanh((body + n) * 1.4) * 0.8


def hat(open_=False):
    t = t_arr(0.25 if open_ else 0.04)
    n = filt(rng.standard_normal(len(t)), "high", 8000)
    return n * np.exp(-t * (12 if open_ else 110)) * (0.3 if open_ else 0.22)


def crash():
    t = t_arr(1.8)
    n = filt(rng.standard_normal(len(t)), "high", 4500)
    return n * np.exp(-t * 2.4) * 0.35


def tom(freq):
    t = t_arr(0.35)
    f = freq * (1 + 0.5 * np.exp(-t * 20))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.8


# --------------------------------------------------------------- instrumentos
def power_chord(root, d, palm=False):
    """Guitarra saturada: quinta + octava, desafinadas y distorsionadas."""
    t = t_arr(d)
    sig = np.zeros_like(t)
    for n, g in ((root, 1.0), (root + 7, 0.8), (root + 12, 0.6)):
        for det in (-0.004, 0.004):
            f = note(n) * (1 + det)
            sig += g * (2 * ((t * f + rng.random()) % 1.0) - 1)
    sig = np.tanh(sig * 3.5)
    sig = filt(sig, "low", 2200 if palm else 4200)
    sig = filt(sig, "high", 90)
    env = np.exp(-t * (14 if palm else 1.4)) * np.minimum(t / 0.004, 1)
    return sig * env * 0.32


def pluck(n, d=0.5):
    """Punteo tipo Karplus-Strong."""
    f = note(n)
    L = int(SR / f)
    out = np.zeros(int(SR * d))
    buf = rng.uniform(-1, 1, L)
    for i in range(len(out)):
        out[i] = buf[i % L]
        buf[i % L] = 0.5 * (buf[i % L] + buf[(i + 1) % L]) * 0.996
    return filt(out, "high", 200) * 0.5


def bass_note(n, d):
    t = t_arr(d)
    f = note(n)
    s = np.sin(2 * np.pi * f * t) + 0.5 * np.tanh(3 * np.sin(2 * np.pi * f * t))
    s = filt(s, "low", 1200)
    return s * np.minimum(t / 0.005, 1) * np.exp(-t * 3) * 0.55


def impact():
    t = t_arr(2.0)
    f = 40 + 70 * np.exp(-t * 12)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    return np.tanh(s * 1.4) * 0.8


def riser(d):
    t = t_arr(d)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    seg = int(SR * 0.04)
    for k in range(0, len(t), seg):
        fc = 300 + 9000 * (k / len(t)) ** 2
        out[k:k + seg] = filt(n[k:k + seg], "band", [fc * 0.7, min(fc * 1.4, 20000)], 1)
    return out * (t / d) ** 2 * 0.45


def reverb(x, secs=1.4, mix=0.2):
    t = t_arr(secs)
    ir = filt(rng.standard_normal(len(t)) * np.exp(-t * 5), "low", 7000)
    ir /= np.sqrt(np.sum(ir ** 2))
    return x + fftconvolve(x, ir)[: len(x)] * mix


def main():
    drums = np.zeros(N)
    bass = np.zeros(N)
    gtr = np.zeros(N)
    lead = np.zeros(N)
    fx = np.zeros(N)
    K, S, H, HO = kick(), snare(), hat(), hat(True)

    # Re mayor: D - A - Bm - G (I - V - vi - IV), un compás por acorde
    prog = [50, 45, 47, 43]          # raíces (D3, A2, B2, G2)
    riff = [74, 76, 78, 81, 78, 76, 74, 69]   # punteo en corcheas
    beats = int(DUR / BEAT)
    for b in range(beats):
        tb = b * BEAT
        bar, ib = b // 4, b % 4
        root = prog[bar % 4]
        intro = tb < 1.0
        ending = tb >= FINAL
        if intro or ending:
            continue
        # batería: bombo en 1 y 3 con síncopa, redoblante en 2 y 4
        place(drums, K, tb, 1.0)
        if ib in (1, 3):
            place(drums, S, tb, 0.9)
        if ib == 2:
            place(drums, K, tb + BEAT * 0.5, 0.7)
        for s in (0.0, 0.5):
            place(drums, H if s == 0 else HO, tb + BEAT * s, 0.6 if s == 0 else 0.45)
        # bajo: corcheas sobre la raíz
        for s in (0.0, 0.5):
            n = root - 12 + (12 if (ib == 3 and s == 0.5) else 0)
            place(bass, bass_note(n, BEAT * 0.48), tb + BEAT * s)
        # guitarras: acorde largo al inicio del compás + palm mute en corcheas
        if tb >= 5.0:
            if ib == 0:
                place(gtr, power_chord(root, BEAT * 4), tb, 0.9)
            else:
                for s in (0.0, 0.5):
                    place(gtr, power_chord(root, BEAT * 0.45, True), tb + BEAT * s, 0.55)
        else:
            if ib == 0:
                place(gtr, power_chord(root, BEAT * 4), tb, 0.6)
        # punteo durante la sección de servicios y el cierre
        if 9.0 <= tb < FINAL:
            for s, k in ((0.0, 0), (0.5, 1)):
                n = riff[(b * 2 + k) % len(riff)] + (root - 50 if bar % 4 in (1, 3) else 0)
                place(lead, pluck(n, 0.45), tb + BEAT * s, 0.45)

    for at in HITS:
        place(fx, impact(), at, 0.7)
        place(drums, crash(), at, 1.0)
    for at in SOFT_HITS:
        place(drums, crash(), at, 0.5)
    place(fx, riser(1.0), 0.0, 0.8)
    place(fx, riser(1.5), 3.5, 0.7)
    place(fx, riser(1.0), 8.0, 0.6)
    place(fx, riser(1.5), 20.0, 0.7)
    # redoble de toms antes del cierre
    for k in range(8):
        place(drums, tom(200 - k * 12), 20.5 + k * BEAT / 4, 0.5 + 0.05 * k)
    # final
    place(fx, impact(), FINAL, 0.8)
    place(drums, crash(), FINAL, 1.0)
    place(gtr, power_chord(50, 2.0), FINAL, 1.0)
    place(bass, bass_note(38, 2.0), FINAL, 1.0)

    gtr = reverb(gtr, 1.0, 0.15)
    lead = reverb(lead, 1.6, 0.3)
    drums = reverb(drums, 0.8, 0.07)
    fx = reverb(fx, 1.8, 0.2)
    mono = drums * 0.85 + bass * 0.9 + fx
    d1, d2 = int(0.013 * SR), int(0.021 * SR)
    left = mono + gtr * 0.7 + np.roll(lead, d1) * 0.9
    right = mono + np.roll(gtr, d2) * 0.66 + lead
    st = np.stack([left, right], 1)
    st = np.tanh(st / np.max(np.abs(st)) * 1.6)
    tt = np.arange(N) / SR
    st *= (np.clip((DUR - tt) / 1.5, 0, 1) * np.clip(tt / 0.05, 0, 1))[:, None]
    st /= np.max(np.abs(st)) / 0.89

    os.makedirs("reel/salida", exist_ok=True)
    with wave.open("reel/salida/musica.wav", "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    print("ok reel/salida/musica.wav")


if __name__ == "__main__":
    main()
