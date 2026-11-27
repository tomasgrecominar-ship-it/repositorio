"""Música original para el reel (120 BPM, 37 s), sintetizada con numpy/scipy.

Genera reel/salida/musica.wav. Los golpes fuertes coinciden con los cortes
del video: 1.0 s (aparece el "3"), 5.0 s (tomas del local),
9.0 s ("Nuestros servicios") y 23.5 s (cierre con el logo).
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
import wave, os

SR = 44100
DUR = 37.0
BPM = 120
BEAT = 60 / BPM
N = int(SR * DUR)
rng = np.random.default_rng(7)


def t_arr(d):
    return np.arange(int(SR * d)) / SR


def filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output="sos")
    return sosfilt(sos, x)


def place(buf, sig, at, gain=1.0):
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def kick():
    t = t_arr(0.45)
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 7.5)
    s += filt(rng.standard_normal(len(t)), "high", 2000) * np.exp(-t * 300) * 0.3
    return np.tanh(s * 1.6)


def clap():
    t = t_arr(0.35)
    n = rng.standard_normal(len(t))
    env = np.exp(-t * 18)
    for d in (0.0, 0.011, 0.022):
        env += (t >= d) * np.exp(-np.clip(t - d, 0, None) * 140) * 0.6
    return filt(n, "band", [900, 4200]) * env * 0.7


def hat(open_=False):
    t = t_arr(0.22 if open_ else 0.05)
    n = filt(rng.standard_normal(len(t)), "high", 7500)
    return n * np.exp(-t * (14 if open_ else 90)) * (0.35 if open_ else 0.25)


def saw(freq, d, detune=(0.0,)):
    t = t_arr(d)
    out = np.zeros_like(t)
    for dt in detune:
        f = freq * (1 + dt)
        out += 2 * ((t * f + rng.random()) % 1.0) - 1
    return out / len(detune)


def note(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def riser(d):
    t = t_arr(d)
    n = rng.standard_normal(len(t))
    out = np.zeros_like(t)
    seg = int(SR * 0.05)
    for k in range(0, len(t), seg):
        frac = k / len(t)
        fc = 400 + 9000 * frac ** 2
        chunk = n[k:k + seg]
        out[k:k + seg] = filt(chunk, "band", [fc * 0.7, min(fc * 1.4, 20000)], 1)
    sweep_f = 200 + 1400 * (t / d) ** 2
    out += np.sin(2 * np.pi * np.cumsum(sweep_f) / SR) * 0.15
    return out * (t / d) ** 2 * 0.5


def impact():
    t = t_arr(2.5)
    f = 38 + 60 * np.exp(-t * 10)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.8)
    s += filt(rng.standard_normal(len(t)), "low", 3000) * np.exp(-t * 9) * 0.5
    return np.tanh(s * 1.3) * 0.9


def reverb(x, secs=1.6, mix=0.25):
    t = t_arr(secs)
    ir = rng.standard_normal(len(t)) * np.exp(-t * 4.5)
    ir = filt(ir, "low", 6000)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = fftconvolve(x, ir)[: len(x)]
    return x + wet * mix


def main():
    drums = np.zeros(N)
    bass = np.zeros(N)
    synth = np.zeros(N)
    fx = np.zeros(N)
    side = np.ones(N)

    K, C, H, HO = kick(), clap(), hat(), hat(True)
    duck = 1 - 0.75 * np.exp(-t_arr(0.3) * 14)

    # Progresión (La menor): Am - F - C - G, un compás por acorde
    prog = [(57, [69, 72, 76]), (53, [65, 69, 72]), (48, [67, 72, 76]), (55, [67, 71, 74])]
    beats = int(DUR / BEAT)

    for b in range(beats):
        tb = b * BEAT
        bar = b // 4
        in_bar = b % 4
        root, chord = prog[bar % 4]
        # secciones
        intro = tb < 1.0
        outro_soft = 23.5 <= tb < 27.5
        ending = tb >= 35.0
        full = not intro and not ending

        if full and not outro_soft:
            place(drums, K, tb, 1.0)
            i = int(tb * SR)
            j = min(N, i + len(duck))
            side[i:j] = np.minimum(side[i:j], duck[: j - i])
        if full and in_bar in (1, 3) and tb >= 5.0:
            place(drums, C, tb, 0.8)
        if full and tb >= 1.0:
            place(drums, HO, tb + BEAT / 2, 0.8)
            if tb >= 5.0:
                for s in (0.25, 0.75):
                    place(drums, H, tb + BEAT * s, 0.7)
        # bajo: corcheas en el offbeat
        if full and tb >= 1.0:
            for s, oct_ in ((0.5, 0), (0.75, 12 if in_bar == 3 else 0)):
                bs = saw(note(root - 12 + oct_), BEAT * 0.24, (0, 0.006))
                env = np.exp(-t_arr(BEAT * 0.24) * 9)
                place(bass, filt(bs, "low", 900) * env, tb + BEAT * s, 0.55)
            sub = np.sin(2 * np.pi * note(root - 24) * t_arr(BEAT * 0.9)) * np.exp(-t_arr(BEAT * 0.9) * 3)
            place(bass, sub, tb + 0.02, 0.35)
        # stabs de acordes
        if (full and tb >= 5.0) or outro_soft:
            for s in (0.5,) if in_bar % 2 == 0 else (0.5, 0.75):
                d = BEAT * 0.3
                st = sum(saw(note(n), d, (-0.01, 0, 0.012)) for n in chord) / 3
                env = np.exp(-t_arr(d) * 10)
                place(synth, filt(st, "low", 3200) * env, tb + BEAT * s, 0.32)
        # pad en el cierre
        if (outro_soft or tb >= 27.5) and in_bar == 0:
            d = BEAT * 4
            pd = sum(saw(note(n), d, (-0.008, 0, 0.009)) for n in chord) / 3
            tt = t_arr(d)
            env = np.minimum(tt / 0.4, 1) * np.exp(-tt * 0.5)
            place(synth, filt(pd, "low", 1500) * env, tb, 0.22)

    # Golpes y subidas sincronizados con el video
    place(fx, riser(1.0), 0.0, 0.8)
    for at in (1.0, 5.0, 9.0, 23.5):
        place(fx, impact(), at, 0.75)
    place(fx, riser(1.5), 3.5, 0.7)
    place(fx, riser(1.5), 7.5, 0.8)
    place(fx, riser(1.0), 22.5, 0.7)
    # redoble antes de la repetición
    for k in range(16):
        place(drums, C, 8.0 + k * BEAT / 8, 0.15 + 0.4 * k / 16)
    # golpe final
    place(fx, impact(), 35.0, 0.8)
    d = 2.0
    fin = sum(saw(note(n), d, (-0.01, 0, 0.01)) for n in [57, 64, 69, 72, 76]) / 5
    place(synth, filt(fin, "low", 2500) * np.exp(-t_arr(d) * 1.6), 35.0, 0.45)

    bass *= side
    synth *= side ** 0.7
    synth = reverb(synth, 1.8, 0.35)
    drums = reverb(drums, 0.9, 0.08)
    fx = reverb(fx, 2.0, 0.2)

    mono = drums * 0.9 + bass + synth + fx
    # estéreo simple: synth y fx un poco abiertos
    d1, d2 = int(0.011 * SR), int(0.017 * SR)
    left = mono + np.roll(synth, d1) * 0.15
    right = mono + np.roll(synth, d2) * 0.15
    st = np.stack([left, right], 1)
    st = np.tanh(st * 1.1)
    # fundido final
    tt = np.arange(N) / SR
    fade = np.clip((DUR - tt) / 1.2, 0, 1)
    fade_in = np.clip(tt / 0.05, 0, 1)
    st *= (fade * fade_in)[:, None]
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
