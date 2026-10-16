"""可选二次元人物前景：官方ISNet ONNX预处理，不将语义前景冒充逐人身份。"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

REPO='skytnt/anime-seg';REVISION='493cb60893f47441b26ec4fb9a306bce9e342982'

def predict_foreground(rgb,path,backend="isnet"):
 import onnxruntime as ort
 ort.disable_telemetry_events()
 options=ort.SessionOptions();options.log_severity_level=3;options.intra_op_num_threads=min(4,os.cpu_count() or 1)
 session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
 try:
  inp=session.get_inputs()
  if len(inp)!=1 or inp[0].type!='tensor(float)' or inp[0].shape!=[1,3,1024,1024]:raise ValueError('Expected a supported foreground ONNX model with 1x3x1024x1024 float32 input')
  h0,w0=rgb.shape[:2]
  if backend=='isnet':
   scale=1024/max(h0,w0);h=max(1,int(h0*scale));w=max(1,int(w0*scale));y=(1024-h)//2;x=(1024-w)//2
   padded=np.zeros((1024,1024,3),np.float32);padded[y:y+h,x:x+w]=np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR),np.float32)/255
  elif backend=='toonout':
   h=w=1024;y=x=0
   padded=np.asarray(Image.fromarray(rgb).resize((1024,1024),Image.Resampling.BILINEAR),np.float32)/255
   padded=(padded-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
  else:raise ValueError('Unsupported foreground backend')
  out=session.run(None,{inp[0].name:np.ascontiguousarray(padded.transpose(2,0,1)[None])})[0]
  if out.shape!=(1,1,1024,1024) or not np.isfinite(out).all() or out.min()<-.001 or out.max()>1.001:raise ValueError('Expected a finite foreground probability map in [0, 1]')
  return np.array(Image.fromarray(out[0,0,y:y+h,x:x+w]).resize((w0,h0),Image.Resampling.BILINEAR)).clip(0,1)
 finally:del session
