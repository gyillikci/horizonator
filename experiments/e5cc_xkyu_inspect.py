#!/usr/bin/env python3
"""E5cc, XKYU3498: are the "missing" far ridges really invisible?

Draws the seam (used) and raw eWaSR boundaries against the COP30
skyline at the truth coloured by range, and measures the contrast of
the terrain at the DEM ridge in three column groups.

    python3 e5cc_xkyu_inspect.py  -> out/visibility/xkyu_boundaries.png
"""
import sys, os, math, json, numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import skyfix as F, skyline as S, extract
from assets import celnav
fr=next(f for f in json.load(open('data/campaign_frames.json'))['frames'] if f['tag']=='38_XKYU3498')
img=extract.load_image(celnav(fr['path'])); H,W,_=img.shape
F.EXTRACTOR='ewasr'; F.BOUNDARY_ROWS=None
el_obs,w,d=F.observation(img,fr['fov'],fr['heading'],0.0)   # falls back to seam
F.EXTRACTOR='ewasr'
# raw eWaSR boundary without the fallback
from ewasr_bridge import EWasr; from assets import need
cls=EWasr(need('eWaSR'),need('ewasr_resnet18.pth')).predict(img); sky=cls==2
er=np.array([np.argmax(~sky[:,x]) if sky[:5,x].any() else np.nan for x in range(W)],float)
cm=S.CMarcher(os.path.expanduser('~/.horizonator/DEMs_COP30'),(fr['lat']-.6,fr['lat']+.6),(fr['lon']-.8,fr['lon']+.8),d_min=1000.)
el,r=cm.skyline(fr['lat'],fr['lon'],fr['z'],F.AZ); el=np.nan_to_num(el+np.where(np.isfinite(r)&(r>0),9/np.maximum(r,1000),0))
c,s,b=F.photo_cost(el_obs,w,el,np.arange(-300,301,2),np.linspace(-.03,.03,61))
f=(W/2)/math.tan(math.radians(fr['fov'])/2)
rl=(F.AZ-(fr['heading']+0.1*s)+180)%360-180; k=np.abs(rl)<fr['fov']/2
u=f*np.tan(np.radians(rl[k])); v=np.tan(el[k]-b)*np.hypot(u,f); x=u+(W-1)/2; y=(H-1)/2-v
fig,ax=plt.subplots(2,1,figsize=(14,8))
for a,(x0,x1) in zip(ax,((0,W),(W*0.55,W))):
    a.imshow(img); a.plot(np.arange(W),d['rows'],'y-',lw=1.3,label='seam (used: eWaSR fell back)')
    a.plot(np.arange(W),er,'m-',lw=1,label='eWaSR raw')
    sc=a.scatter(x,y,c=r[k]/1e3,s=2,cmap='viridis',vmin=0,vmax=40,label='COP30 at truth, colour = range')
    rows=np.r_[d['rows'],er[np.isfinite(er)]]; lo,hi=np.nanpercentile(rows,[1,99])
    a.set_xlim(x0,x1); a.set_ylim(hi+60,lo-60); a.legend(fontsize=7,loc='lower left')
plt.colorbar(sc,ax=ax,label='crest range km',fraction=.02)
ax[0].set_title('XKYU3498 at the truth (heading %+.1f, beta %+.1f mrad); bottom: right half'%(0.1*s,b*1e3))
plt.savefig('out/visibility/xkyu_boundaries.png',dpi=100,bbox_inches='tight')
import e5cc_visibility as V
L=V.lin(img)@np.array([0.2126,0.7152,0.0722])
yy=np.interp(np.arange(W),x[np.argsort(x)],y[np.argsort(x)])
rx=np.interp(np.arange(W),x[np.argsort(x)],(r[k]/1e3)[np.argsort(x)])
for x0,x1 in ((50,400),(1330,1500),(850,1000)):
    cs=[]
    for xx in range(x0,x1):
        t=int(round(yy[xx])); sk=np.median(L[t-12:t-3,xx]); te=np.median(L[t+3:t+9,xx]); cs.append((sk-te)/sk)
    print('cols %4d-%4d: DEM crest %.0f km, contrast at the DEM ridge %.3f; photo boundary %+.0f px below DEM'%(x0,x1,np.median(rx[x0:x1]),np.median(cs),np.median(d['rows'][x0:x1]-yy[x0:x1])))
