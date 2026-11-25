import numpy as np, wave, sys
SR=44100; DUR=34.0; N=int(SR*DUR)
rng=np.random.default_rng(7)
L=np.zeros(N); R=np.zeros(N)
def hz(m): return 440*2**((m-69)/12)
bar=2.9
# Chords (midi) per bar - original progression in D major
prog=[[50,62,66,69,73],[47,62,66,69,71],[43,59,62,66,71],[45,61,64,69,71],
      [50,62,66,69,73],[47,62,66,69,74],[43,59,62,67,71],[45,61,64,67,69],
      [50,62,66,69,73],[47,62,66,69,74],[50,62,66,69,76]]
t=np.arange(N)/SR
def add(sig,start,pan=0.0,gain=1.0):
    i=int(start*SR); j=min(N,i+len(sig)); s=sig[:j-i]*gain
    L[i:j]+=s*np.cos((pan+1)*np.pi/4); R[i:j]+=s*np.sin((pan+1)*np.pi/4)
def pad(m,dur):
    n=int(SR*(dur+1.5)); tt=np.arange(n)/SR; f=hz(m); s=np.zeros(n)
    for d in (-0.08,0,0.07):
        for k,a in ((1,1),(2,.35),(3,.12),(4,.05)):
            s+=a*np.sin(2*np.pi*f*k*(1+d/100)*tt+rng.uniform(0,6.28))
    env=np.minimum(tt/1.2,1)*np.where(tt>dur,np.exp(-(tt-dur)/0.6),1)
    return s*env*(1+0.15*np.sin(2*np.pi*0.3*tt))
def bell(m,dur=2.5):
    n=int(SR*dur); tt=np.arange(n)/SR; f=hz(m)
    s=np.sin(2*np.pi*f*tt)*np.exp(-tt*2.2)+0.35*np.sin(2*np.pi*f*4.0*tt)*np.exp(-tt*6)+0.15*np.sin(2*np.pi*f*2.76*tt)*np.exp(-tt*4)
    return s*np.minimum(tt/0.004,1)
def bass(m,dur):
    n=int(SR*(dur+0.5)); tt=np.arange(n)/SR; f=hz(m-12)
    s=np.sin(2*np.pi*f*tt)+0.25*np.sin(2*np.pi*2*f*tt)
    return s*np.minimum(tt/0.05,1)*np.exp(-tt*0.5)
for b,ch in enumerate(prog):
    st=b*bar; dur=bar if b<len(prog)-1 else 4.5
    for m in ch[1:4]: add(pad(m,dur),st,pan=rng.uniform(-.4,.4),gain=0.05)
    add(bass(ch[0],dur),st,gain=0.16)
    if b==len(prog)-1:
        for k,m in enumerate([62,66,69,74,78]): add(bell(m+12,4),st+k*0.18,pan=-.5+k*.25,gain=0.11)
        continue
    # music-box arpeggio, 8th notes
    arp=[ch[1]+12,ch[2]+12,ch[3]+12,ch[4]+12,ch[3]+12,ch[2]+12,ch[4]+12,ch[3]+24]
    for k,m in enumerate(arp):
        if b==0 and k<2: continue
        add(bell(m),st+k*bar/8+rng.normal(0,0.006),pan=np.sin(k)*0.5,gain=0.085*(1 if k%2==0 else 0.75))
# sparkles
for _ in range(26):
    tt0=rng.uniform(1,30); add(bell(rng.choice([86,88,90,93,95]),1.5),tt0,pan=rng.uniform(-.8,.8),gain=0.025)
# reverb
irn=int(SR*2.8); it=np.arange(irn)/SR
def ir(): 
    x=rng.normal(0,1,irn)*np.exp(-it*2.4); x[:int(.012*SR)]=0; return x/np.sqrt((x**2).sum())
from numpy.fft import rfft, irfft
def conv(a,b):
    n=len(a)+len(b); n2=1<<(n-1).bit_length(); return irfft(rfft(a,n2)*rfft(b,n2),n2)[:len(a)]
Lw=conv(L,ir()); Rw=conv(R,ir())
L=0.75*L+0.55*Lw; R=0.75*R+0.55*Rw
fade=np.ones(N); fo=int(3.5*SR); fade[-fo:]=np.linspace(1,0,fo)**2; fade[:int(.3*SR)]=np.linspace(0,1,int(.3*SR))
st=np.stack([L*fade,R*fade],1); st/=np.abs(st).max()/0.8
with wave.open(sys.argv[1],'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st*32767).astype('<i2').tobytes())
