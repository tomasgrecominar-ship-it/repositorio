import json, re, librosa, numpy as np
from faster_whisper import WhisperModel
B='/tmp/claude-0/-home-user-repositorio/fc78fd73-722f-5b61-bc76-45bfdb780b60/scratchpad/'
lines=[l.strip() for l in open(B+'vo/lines.txt',encoding='utf-8') if l.strip()]
m=WhisperModel("small",device="cpu",compute_type="int8")
res=[]
PAD=0.5
for i,l in enumerate(lines,1):
    y,_=librosa.load(B+f'vo/l{i}.wav',sr=16000)
    a=np.concatenate([np.zeros(int(PAD*16000)),y,np.zeros(int(0.3*16000))]).astype(np.float32)
    segs,_=m.transcribe(a,language="es",word_timestamps=True,beam_size=5,initial_prompt=l)
    ws=[(w.word.strip(),max(0,w.start-PAD),max(0,w.end-PAD)) for s in segs for w in s.words]
    spoken=re.findall(r"[\wáéíóúñü]+",l.lower())
    print(i,len(spoken),len(ws),' '.join(w for w,_,_ in ws))
    res.append({'text':l,'spoken':spoken,'asr':ws,'dur':len(y)/16000})
json.dump(res,open(B+'vo/words.json','w'),ensure_ascii=False,indent=1)
