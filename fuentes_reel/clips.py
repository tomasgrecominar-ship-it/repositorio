import subprocess, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
MK = os.path.join(HERE, "real", "mk")
CW, CH = 1296, 2304   # working canvas (1.2x the 1080x1920 output, room for camera moves)

# scene clip specs: file, start time, horizontal crop centre (source px), playback speed
SPEC = {
    "intro": dict(f="45585.mp4", t0=1.0, cx=773, speed=1.0),
    "d01":   dict(f="4148.mp4", t0=4.0, cx=700, speed=1.0),
    "d08":   dict(f="46101.mp4", t0=1.0, cx=560, speed=1.0),
    "d17":   dict(f="1704_1080.mp4", t0=2.0, cx=1100, speed=1.0),
    "d23":   dict(f="46898.mp4", t0=3.0, cx=640, speed=1.0),
    "d30":   dict(f="4695.mp4", t0=0.5, cx=640, speed=1.0),
    "outro": dict(f="4124.mp4", t0=2.0, cx=640, speed=1.0),
    "clouds": dict(f="47299.mp4", t0=2.0, cx=640, speed=1.0),
    "clouds2": dict(f="45587.mp4", t0=14.9, cx=640, speed=1.0),
}

def probe(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    return int(o[0]), int(o[1])

def read_clip(name, dur, fps=30):
    s = SPEC[name]; path = os.path.join(MK, s["f"]); sw, sh = probe(path)
    cw = int(round(sh * 9 / 16)) // 2 * 2
    x0 = int(np.clip(s["cx"] - cw / 2, 0, sw - cw))
    vf = (f"setpts=PTS/{s['speed']},fps={fps},crop={cw}:{sh}:{x0}:0,"
          f"scale={CW}:{CH}:flags=lanczos,unsharp=5:5:0.5:5:5:0")
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(s["t0"]), "-i", path, "-t", str(dur * s["speed"] + 0.5),
                        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, CH, CW, 3)
    n = int(round(dur * fps)) + 2
    if len(a) < n:   # hold the last frame if the clip is short
        a = np.concatenate([a, np.repeat(a[-1:], n - len(a), 0)])
    return a[:n]

def src_to_canvas(name, x, y):
    s = SPEC[name]; path = os.path.join(MK, s["f"]); sw, sh = probe(path)
    cw = int(round(sh * 9 / 16)) // 2 * 2; x0 = int(np.clip(s["cx"] - cw / 2, 0, sw - cw))
    k = CH / sh
    return (x - x0) * k, y * k, k
