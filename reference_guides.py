"""Procedural geometry for reproducible structural-control examples."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image,ImageDraw
from .auto_regions import label_layout,label_image,overlay_image

def make_guide(count,width,height,pose='group'):
 if count not in (2,3,5,7) or width%16 or height%16:raise ValueError('Guide count must be 2, 3, 5 or 7 and dimensions divisible by 16')
 im=Image.new('L',(width,height),0);draw=ImageDraw.Draw(im)
 def ellipse(box,c):draw.ellipse(tuple(int(v*s) for v,s in zip(box,[width,height,width,height])),fill=c)
 def polygon(points,c):draw.polygon([(int(x*width),int(y*height)) for x,y in points],fill=c)
 def limb(points,c,r=.025):
  xy=[(int(x*width),int(y*height)) for x,y in points];radius=max(2,int(r*min(width,height)));draw.line(xy,fill=c,width=radius*2,joint='curve')
  for x,y in xy:draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=c)
 if count==2:
  for c,cx in [(2,.67),(1,.33)]:
   ellipse((cx-.135,.12,cx+.135,.39),c);polygon([(cx-.11,.4),(cx+.11,.4),(cx+.15,.95),(cx-.15,.95)],c)
   limb([(cx-.08,.43),(cx-.13,.65),(cx-.12,.88)],c,.045)
  limb([(.35,.45),(.49,.5),(.68,.46),(.79,.4)],1,.04)
  limb([(.69,.57),(.58,.72),(.36,.76),(.24,.7)],2,.04)
 else:
  if count==3:people=[(.19,.28,False),(.5,.28,False),(.81,.28,False)]
  elif count==5:people=[(.2,.48,False),(.35,.17,True),(.5,.48,False),(.65,.17,True),(.8,.48,False)]
  else:people=[(.13,.48,False),(.25,.17,True),(.375,.48,False),(.5,.17,True),(.625,.48,False),(.75,.17,True),(.87,.48,False)]
  for back in [True,False]:
   for c,(cx,cy,isback) in enumerate(people,1):
    if isback!=back:continue
    r=.09 if count==3 else .067;end=.97 if not back else .7
    ellipse((cx-r,cy-r*.95,cx+r,cy+r*.95),c);polygon([(cx-r*.85,cy+r),(cx+r*.85,cy+r),(cx+r,end-.1),(cx-r,end-.1)],c)
    limb([(cx-r*.45,end-.14),(cx-r*.5,end)],c,r*.28);limb([(cx+r*.45,end-.14),(cx+r*.5,end)],c,r*.28)
    shoulder=cy+r*1.3
    if count==3:
     limb([(cx-r*.8,shoulder),(cx-r*1.35,shoulder+.12),(cx-r*1.7,shoulder+.13)],c,r*.25);limb([(cx+r*.8,shoulder),(cx+r*1.35,shoulder+.12),(cx+r*1.7,shoulder+.13)],c,r*.25)
    else:
     limb([(cx-r*.8,shoulder),(cx-r*1.35,shoulder+.12)],c,r*.26)
     limb([(cx+r*.8,shoulder),(cx+r*1.3,shoulder-.05),(cx+r*1.45,shoulder-.18)],c,r*.26)
 labels=np.asarray(im);layout=label_layout(labels,width,height,6)
 palette=label_image(layout);rgb=palette.clone();background=(rgb.sum(-1)==0);rgb[background]=1
 layout['reference_image']=rgb
 return layout,rgb,palette
