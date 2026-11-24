#!/usr/bin/env python3
"""Builds the final audio mix (VO + ducked music + SFX), loudness-normalised for Instagram."""
import os, sys, subprocess
B = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.insert(0, B)
import render as R  # timings only (main() not executed)

TOTAL = R.TOTAL
MUSIC_IN = 13.468
PEAK = {'1492': 1.103, '1490': 0.708, '1489': 0.72, '772': 0.488, '1143': 0.592, '817': 0.302, '3005': 0.058}

# (sfx id, time where its peak should land, gain, optional trim start)
sfx = [('772', 0.0, 0.6, 0.47), ('1489', R.CUTS[1], 0.30, 0)]
for s in R.SHOTS[1:]:
    if s.trans == 'whip':
        sfx.append(('1143' if s.i == len(R.SHOTS) - 1 else '1492', s.start, 0.55, 0))
for ln in (5, 6, 7):
    sfx.append(('1490', R.wt(ln, 0)[0] + 0.12, 0.28, 0))
sfx.append(('772', R.t_ader + 0.05, 0.45, 0))
sfx.append(('817', R.CUTS[12] + 0.35, 0.45, 0))
for el in R.ELS:
    pass
pops = [R.wt(2, 8)[0], R.wt(3, 0)[0], R.wt(8, 4)[0], R.wt(8, 9)[0], R.wt(9, 0)[0], R.t_cta, R.T_SAVE]
for t in pops:
    sfx.append(('3005', t + 0.04, 0.30, 0))

inputs, filters, labels = [], [], []
# VO lines
for i in range(0):
    inputs += ['-i', B + f'vo/l{i+1}.wav']
    d = int(R.LS[i] * 1000)
    filters.append(f'[{i}:a]aresample=48000,aformat=channel_layouts=mono,adelay={d}|{d}[v{i}]')
# no voice-over: music carries the piece
mi = 0
inputs += ['-ss', f'{MUSIC_IN}', '-t', f'{TOTAL + 0.5}', '-i', B + 'music/126.mp3']
filters.append(f'[{mi}:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.62[mus]')
filters.append('[mus]anull[musd]')
filters.append(f'[musd]afade=t=out:st={TOTAL - 1.6}:d=1.6,atrim=0:{TOTAL}[musf]')
# sfx
k = mi + 1
sfx_labels = []
for j, (sid, t, g, trim) in enumerate(sfx):
    inputs += ['-i', B + f'sfx/{sid}.mp3']
    start = max(0.0, t - (PEAK[sid] - trim))
    d = int(start * 1000)
    tr = f'atrim=start={trim},asetpts=PTS-STARTPTS,' if trim else ''
    filters.append(f'[{k}:a]aresample=48000,aformat=channel_layouts=stereo,{tr}volume={g},adelay={d}|{d}[s{j}]')
    sfx_labels.append(f'[s{j}]')
    k += 1
filters.append(f'[musf]{"".join(sfx_labels)}amix=inputs={1 + len(sfx_labels)}:normalize=0,'
               f'atrim=0:{TOTAL},alimiter=limit=0.89:level=false[pre]')

out_pre = B + 'work/mix_pre.wav'
cmd = ['ffmpeg', '-loglevel', 'error', '-y'] + inputs + ['-filter_complex', ';'.join(filters), '-map', '[pre]', '-ar', '48000', out_pre]
subprocess.run(cmd, check=True)

# two-pass loudnorm to -14 LUFS / -1.5 dBTP
import json, re
r = subprocess.run(['ffmpeg', '-hide_banner', '-i', out_pre, '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json', '-f', 'null', '-'],
                   capture_output=True, text=True)
js = json.loads(re.findall(r'\{[^{}]+\}', r.stderr)[-1])
af = (f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={js['input_i']}:measured_TP={js['input_tp']}:"
      f"measured_LRA={js['input_lra']}:measured_thresh={js['input_thresh']}:offset={js['target_offset']}:linear=true")
subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', out_pre, '-af', af + ',aresample=48000', '-ar', '48000', B + 'work/mix.wav'], check=True)
print('measured', js['input_i'], '->', 'mix.wav')
print('sfx', [(s, round(t, 2)) for s, t, _, _ in sfx])
