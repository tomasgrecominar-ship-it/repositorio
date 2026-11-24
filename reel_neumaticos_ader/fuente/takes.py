import subprocess, difflib, re, unicodedata, librosa, numpy as np, soundfile as sf, json
from faster_whisper import WhisperModel
B='/tmp/claude-0/-home-user-repositorio/fc78fd73-722f-5b61-bc76-45bfdb780b60/scratchpad/'
lines=[l.strip() for l in open(B+'vo/lines.txt',encoding='utf-8') if l.strip()]
m=WhisperModel("small",device="cpu",compute_type="int8")
def norm(s):
    s=unicodedata.normalize('NFD',s.lower()); s=''.join(c for c in s if unicodedata.category(c)!='Mn')
    s=s.replace('1,6','uno coma seis').replace('1 ,6','uno coma seis').replace('210','doscientos diez')
    return re.sub(r'[^a-z ]','',s).split()
report={}
for i,l in enumerate(lines,1):
    best=None
    for k in range(8):
        ns=[0.667,0.5,0.4,0.667,0.5,0.4,0.6,0.45][k]
        raw=B+f'takes/r{i}_{k}.wav'
        subprocess.run(['piper','-m',B+'voice/es_MX-claude-high.onnx','--length_scale','0.93','--noise_scale',str(ns),'-f',raw],input=(', '+l).encode(),stderr=subprocess.DEVNULL,check=True)
        y,sr=librosa.load(raw,sr=48000)
        y=librosa.effects.time_stretch(y,rate=1.06)
        a16=librosa.resample(y,orig_sr=48000,target_sr=16000)
        a16=np.concatenate([np.zeros(8000),a16]).astype(np.float32)
        segs,_=m.transcribe(a16,language="es",beam_size=5)
        txt=' '.join(s.text for s in segs)
        ref,hyp=norm(l),norm(txt)
        score=difflib.SequenceMatcher(None,ref,hyp).ratio()
        first_ok = 1 if hyp[:1]==ref[:1] else 0
        tot=score+0.5*first_ok
        if best is None or tot>best[0]: best=(tot,k,txt,y)
        if score==1.0: break
    tot,k,txt,y=best
    env=np.convolve(np.abs(y),np.ones(240)/240,'same'); idx=np.where(env>10**(-62/20))[0]
    a=max(0,idx[0]-int(0.03*48000)); b=min(len(y),idx[-1]+int(0.06*48000)); y=y[a:b]
    f=np.ones_like(y); n=int(0.004*48000); f[:n]=np.linspace(0,1,n); f[-n:]=np.linspace(1,0,n)
    sf.write(B+f'vo/l{i}.wav',y*f,48000,subtype='PCM_24')
    report[i]=(round(tot,3),k,txt,round(len(y)/48000,2)); print(i,report[i],flush=True)
