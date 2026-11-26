"""Original ambient-pop track for the reel (96 BPM, B minor). Drop at 17.5 s."""
import numpy as np, sys
from scipy.signal import butter, lfilter, fftconvolve
from scipy.io import wavfile

SR = 44100
BPM = 96
BEAT = 60 / BPM          # 0.625 s
BAR = 4 * BEAT           # 2.5 s
TOTAL = 32.4
DROP = 17.5
N = int(TOTAL * SR)
rng = np.random.default_rng(3)

dryL = np.zeros(N); dryR = np.zeros(N); send = np.zeros(N)

def mtof(m): return 440 * 2 ** ((m - 69) / 12)

def add(sig, t0, gain=1.0, pan=0.0, rev=0.0):
    i0 = int(t0 * SR)
    if i0 >= N: return
    sig = sig[: N - i0]
    l = np.cos((pan + 1) * np.pi / 4); r = np.sin((pan + 1) * np.pi / 4)
    dryL[i0:i0 + len(sig)] += sig * gain * l * 1.414
    dryR[i0:i0 + len(sig)] += sig * gain * r * 1.414
    send[i0:i0 + len(sig)] += sig * gain * rev

def tt(d): return np.arange(int(d * SR)) / SR

def hp(x, f): b, a = butter(2, f / (SR / 2), "high"); return lfilter(b, a, x)
def lp(x, f): b, a = butter(2, f / (SR / 2), "low"); return lfilter(b, a, x)

# ---------- instruments ----------
def pad(notes, dur, att=0.9, rel=1.4):
    t = tt(dur + rel)
    env = np.clip(t / att, 0, 1) * np.where(t < dur, 1, np.exp(-(t - dur) * 3.5))
    s = np.zeros_like(t)
    for m in notes:
        f = mtof(m)
        for dc in (-0.07, 0, 0.07):
            ff = f * 2 ** (dc / 12)
            ph = rng.uniform(0, 6.28)
            s += np.sin(2 * np.pi * ff * t + ph) + 0.28 * np.sin(4 * np.pi * ff * t + ph) + 0.1 * np.sin(6 * np.pi * ff * t)
    s *= env * (0.85 + 0.15 * np.sin(2 * np.pi * 0.3 * t))
    return s / (len(notes) * 3)

def pluck(m, d=1.6, bright=1.0):
    t = tt(d); f = mtof(m)
    s = (np.sin(2 * np.pi * f * t) + 0.45 * bright * np.sin(4 * np.pi * f * t) * np.exp(-t * 7)
         + 0.2 * bright * np.sin(6 * np.pi * f * t) * np.exp(-t * 12))
    return s * np.exp(-t * 4.2) * np.clip(t / 0.004, 0, 1)

def bell(m, d=2.5):
    t = tt(d); f = mtof(m)
    s = np.sin(2 * np.pi * f * t + 1.2 * np.sin(2 * np.pi * f * 3.5 * t) * np.exp(-t * 3))
    return s * np.exp(-t * 2.2) * np.clip(t / 0.003, 0, 1)

def bass(m, d):
    t = tt(d + 0.15); f = mtof(m)
    env = np.clip(t / 0.01, 0, 1) * np.where(t < d, 0.75 + 0.25 * np.exp(-t * 6), 0.75 * np.exp(-(t - d) * 30))
    return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * env

def kick():
    t = tt(0.5); f = 44 + 95 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 7.5) + 0.3 * rng.standard_normal(len(t)) * np.exp(-t * 400)

def hat(d=0.07):
    t = tt(d); return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t * 55)

def clap():
    t = tt(0.4); n = rng.standard_normal(len(t))
    n = hp(lp(n, 3500), 900)
    env = np.exp(-t * 16) + sum(np.exp(-((t - k) * 300) ** 2) * 0.6 for k in (0.0, 0.012, 0.024))
    return n * env * 0.8

def snare_hit(d=0.18):
    t = tt(d); n = hp(lp(rng.standard_normal(len(t)), 6000), 400)
    return (n * 0.8 + 0.4 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 22)

def impact(big=True):
    d = 4.0 if big else 2.5
    t = tt(d)
    f = 34 + 50 * np.exp(-t * 6)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (1.2 if big else 2.0))
    crash = hp(rng.standard_normal(len(t)), 4500) * np.exp(-t * (1.1 if big else 1.8)) * 0.35
    return sub, crash

def riser(d):
    t = tt(d); x = t / d
    n = rng.standard_normal(len(t))
    lo = lp(n, 1200); hi = hp(n, 2500)
    noise = (lo * (1 - x) + hi * x * 1.4) * x ** 2.2
    sweep = np.sin(2 * np.pi * np.cumsum(220 + 900 * x ** 2) / SR) * x ** 2 * 0.25
    return noise * 0.6 + sweep

# ---------- harmony ----------
CH = [  # (bass, pad notes, arp tones)
    (35, [54, 59, 62, 66], [71, 74, 78, 83]),        # Bm
    (31, [55, 59, 62, 69], [71, 74, 79, 81]),        # Gadd9
    (38, [54, 57, 62, 64], [69, 74, 76, 78]),        # Dadd9
    (33, [52, 57, 61, 64], [69, 73, 76, 81]),        # A
]
NBARS = int(np.ceil(TOTAL / BAR))

def in_(t, a, b): return a <= t < b

# hook at 0: soft impact + shimmer
sub, cr = impact(False)
add(sub, 0, 0.55); add(cr, 0, 0.25, rev=0.5)
for i, m in enumerate([83, 78, 74, 71]):
    add(bell(m), 0.05 + i * 0.16, 0.10, pan=(-0.4 + i * 0.25), rev=0.9)

for b in range(NBARS):
    t0 = b * BAR
    root, pnotes, arp = CH[b % 4]
    if t0 >= TOTAL - 0.5: break
    # pad everywhere (louder after drop), held chord at the end
    if t0 >= 29.9:
        add(pad([47] + CH[0][1], 1.2, att=0.3, rel=1.6), t0, 0.6, rev=0.9)
        add(bell(83, 2.4), t0, 0.08, rev=1.0)
        continue
    g = 0.42 if t0 < DROP else (0.6 if t0 < 26.25 else 0.45)
    add(pad(pnotes, BAR, att=0.7 if t0 else 0.25), t0, g, rev=0.6)

    for k in range(16):  # 16th grid
        t = t0 + k * BEAT / 4
        if t >= TOTAL - 1.0: break
        eighth = k % 2 == 0
        beat = k % 4 == 0
        q = k // 4
        # ---- intro (0-5): quarter plucks
        if t < 5.0:
            if beat and t > 0.6:
                add(pluck(arp[q % 4], bright=0.6), t, 0.14, pan=0.3 * (-1) ** q, rev=0.8)
        # ---- verse (5-12.5)
        elif t < 12.5:
            if eighth:
                add(pluck(arp[[0, 1, 2, 3, 2, 1, 2, 3][k // 2]]), t, 0.13, pan=0.35 * (-1) ** (k // 2), rev=0.6)
            if k in (0, 8): add(kick(), t, 0.55)
            if k in (2, 6, 10, 14): add(hat(), t, 0.07, pan=0.2)
            if k == 0: add(bass(root, BAR * 0.95), t, 0.32)
        # ---- build (12.5-17.5)
        elif t < DROP:
            if t >= DROP - 0.25:  # breath before the drop
                continue
            dens = 2 if t < 15.0 else 1
            if k % dens == 0:
                add(pluck(arp[(k // dens) % 4], bright=1.2), t, 0.12, pan=0.35 * (-1) ** k, rev=0.6)
            if beat and t < 15.0: add(kick(), t, 0.5)
            if t >= 15.0:
                frac = (t - 15.0) / (DROP - 15.0)
                step = 4 if frac < 0.4 else (2 if frac < 0.8 else 1)
                if k % step == 0:
                    add(snare_hit(), t, 0.12 + 0.25 * frac, rev=0.3)
                    if step == 1:  # 32nds at the very end
                        add(snare_hit(), t + BEAT / 8, 0.12 + 0.25 * frac, rev=0.3)
            if k == 0 and t < 15.0: add(bass(root, BAR * 0.95), t, 0.30)
        # ---- drop (17.5-26.25)
        elif t < 26.25:
            add(pluck(arp[[0, 2, 1, 3][k % 4]] + (12 if k % 8 == 7 else 0), bright=1.1), t, 0.10,
                pan=0.4 * (-1) ** k, rev=0.5)
            if beat: add(kick(), t, 0.62)
            if k in (4, 12): add(clap(), t, 0.26, rev=0.45)
            if k % 4 == 2: add(hat(0.09), t, 0.10, pan=-0.2)
            if eighth: add(bass(root, BEAT / 2 * 0.85), t, 0.30)
            if k == 0:
                add(bell(arp[3] + 12), t, 0.06, pan=0.2, rev=0.9)
        # ---- outro (26.25-...)
        else:
            if beat:
                add(pluck(arp[[3, 2, 1, 0][q]], bright=0.6), t, 0.12, pan=0.3 * (-1) ** q, rev=0.9)

# riser into the drop + big impact on the drop
add(riser(DROP - 0.25 - 12.5), 12.5, 0.32, rev=0.3)
sub, cr = impact(True)
add(sub, DROP, 0.95); add(cr, DROP, 0.45, rev=0.6)
for i, m in enumerate([95, 90, 86, 83, 78]):
    add(bell(m, 3), DROP + i * 0.04, 0.05, pan=(-0.6 + i * 0.3), rev=1.0)
# soft impact into the outro
sub, cr = impact(False)
add(sub, 26.25, 0.45); add(cr, 26.25, 0.18, rev=0.6)
# soft cloud "whooshes" on transitions
for tw in (5.0, 8.75, 12.5, 22.5):
    d = 1.4; t = tt(d); x = np.sin(np.pi * t / d) ** 2
    w = lp(hp(rng.standard_normal(len(t)), 300), 2500) * x
    add(w, tw - d / 2, 0.10, rev=0.4)

# ---------- reverb ----------
irn = int(3.0 * SR); it = np.arange(irn) / SR
irL = rng.standard_normal(irn) * np.exp(-it * 2.0); irR = rng.standard_normal(irn) * np.exp(-it * 2.0)
irL = lp(irL, 6000); irR = lp(irR, 6000)
irL /= np.sqrt(np.sum(irL ** 2)); irR /= np.sqrt(np.sum(irR ** 2))
wetL = fftconvolve(send, irL)[:N]; wetR = fftconvolve(send, irR)[:N]
L = dryL + 0.55 * wetL; R = dryR + 0.55 * wetR
L = hp(L, 28); R = hp(R, 28)
mix = np.stack([L, R], 1)
mix = np.tanh(mix / np.max(np.abs(mix)) * 1.6) / np.tanh(1.6)
fade = np.ones(N); fl = int(1.6 * SR); fade[-fl:] = np.linspace(1, 0, fl) ** 2
mix *= fade[:, None] * 0.89
wavfile.write(sys.argv[1] if len(sys.argv) > 1 else "music.wav", SR, (mix * 32767).astype(np.int16))
print("ok", TOTAL)
