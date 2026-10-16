"""Public nodes organized by function with stable IDs and localized metadata."""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import backend as b
from .model_store import MODELS,ensure_model,available
from .anime_foreground import predict_foreground
SCHEMA=json.loads((Path(__file__).with_name('node_schema.json')).read_text(encoding='utf-8'))

def schema(cls):
 spec=copy.deepcopy(SCHEMA[cls.__name__]['inputs'])
 if 'sam2_checkpoint' in spec.get('required',{}):spec['required']['sam2_checkpoint'][0]=available('sam2',['sam2.1_hiera_tiny.pt'])
 if cls.__name__=='AnimaPersonDetect':spec['required']['detector'][0]=available('anima_contact_detectors',['person_detect_v1.3_s.onnx','person_detect_v1.1_m.onnx'])
 return spec

def checkpoint(name,auto_download):
 if name=='sam2.1_hiera_tiny.pt':ensure_model('sam2-tiny',auto_download)
 import folder_paths
 return folder_paths.get_full_path_or_raise('sam2',name)

class AnimaRegionCanvas(b._RegionCanvas):pass
class AnimaSceneConditioning(b._SceneConditioning):pass
class AnimaOwnershipEditor(b._OwnershipEditor):
 def paint(self,**kwargs):
  result=super().paint(**kwargs);layout=result['result'][0]
  result['result']=(*result['result'],layout['width'],layout['height'])
  return result
class AnimaIdentityMapImport(b._IdentityMapImport):pass
class AnimaLayoutControl(b._LayoutControl):pass
class AnimaContactLLLiteApply(b._LLLiteApply):pass
class AnimaTileUpscale(b._TileUpscale):pass
class AnimaRegionalInspect(b._RegionalInspect):pass

class AnimaIdentity(b._Identity):
 def make(self,**kwargs):return super().make(**kwargs,unmatched_policy='error')

class AnimaRegionalApply(b._RegionalApply):
 def apply(self,self_region_isolation=0.,isolation_start_layer=.5,**kwargs):
  result=super().apply(**kwargs);model,pipe,report=result['result']
  options=dict(self_region_isolation=float(self_region_isolation),isolation_start_layer=float(isolation_start_layer))
  if not 0<=self_region_isolation<=1 or not 0<=isolation_start_layer<=1:raise ValueError('Isolation parameters must be in [0, 1]')
  pipe['settings'].update(options);pipe['bundle'].update(options)
  report+='\nSpatial self-attention isolation: '+str(options)
  return {'ui':{'text':[report]},'result':(model,pipe,report)}

class AnimaPersonDetect(b._PersonDetect):
 OUTPUT_NODE=False
 def detect(self,detector,auto_download=True,**kwargs):
  key={'person_detect_v1.3_s.onnx':'detector-small','person_detect_v1.1_m.onnx':'detector-medium'}.get(detector)
  if key:ensure_model(key,auto_download)
  return super().detect(detector=detector,**kwargs)

class AnimaPersonSegment(b._PersonSegment):
 def analyze(self,sam2_checkpoint,auto_download=True,**kwargs):
  checkpoint(sam2_checkpoint,auto_download)
  return super().analyze(sam2_checkpoint=sam2_checkpoint,**kwargs)

class AnimaSeedSegment(b._SeedSegment):
 def analyze(self,sam2_checkpoint,auto_download=True,**kwargs):
  checkpoint(sam2_checkpoint,auto_download)
  return super().analyze(sam2_checkpoint=sam2_checkpoint,**kwargs)

class AnimaIdentitySeeds(b._IdentitySeeds):
 def prepare(self,image,canvas_width,canvas_height,expected_people,strokes_json):
  raw=json.loads(strokes_json or '{}');key=b.reference_key(image,canvas_width,canvas_height,expected_people)
  if raw.get('source_id') and raw['source_id']!=key:raw={'version':1,'source_id':key,'strokes':[]}
  result=super().prepare(image,canvas_width,canvas_height,expected_people,json.dumps(raw))
  result['ui']['anima_mask_editor'][0]['strokes_state']=raw
  return result

class AnimaForeground:
 def segment(self,image,backend,auto_download=True,fusion='consensus'):
  if image.ndim!=4 or image.shape[0]!=1:raise ValueError('Foreground segmentation requires one RGB image')
  rgb=b.to_rgb(image,image.shape[2],image.shape[1]);keys=['isnet','toonout'] if backend=='ensemble' else [backend]
  masks=[torch.from_numpy(predict_foreground(rgb,ensure_model(key,auto_download),key))[None] for key in keys]
  mask=AnimaMaskCombine().combine(masks[0],masks[1],fusion)[0] if len(masks)>1 else masks[0]
  preview=mask[:,:,:,None].expand(-1,-1,-1,3).contiguous()
  report=f'Foreground backend={backend}; fusion={fusion}; source={image.shape[2]}x{image.shape[1]}; union foreground only, no instance identifiers.'
  return {'ui':{'text':[report]},'result':(mask,preview,report)}

class AnimaMaskCombine:
 def combine(self,mask_a,mask_b,fusion):
  a=mask_a.detach().float();c=mask_b.detach().float()
  if a.ndim!=3 or c.ndim!=3 or a.shape[0]!=1 or c.shape[0]!=1:raise ValueError('Expected single-image MASK tensors')
  if not torch.isfinite(a).all() or not torch.isfinite(c).all():raise ValueError('Masks must contain finite probabilities')
  c=F.interpolate(c[None],size=a.shape[-2:],mode='bilinear',align_corners=False)[0].to(a)
  a=a.clamp(0,1);c=c.clamp(0,1)
  out=(a*c).sqrt() if fusion=='consensus' else torch.maximum(a,c) if fusion=='coverage' else torch.minimum(a,c) if fusion=='intersection' else None
  if out is None:raise ValueError('Unknown mask fusion policy')
  return (out,)

class AnimaContactForegroundOwnership:
 def partition(self,image,foreground_mask,seeds,analysis_resolution=768,mask_threshold=.5,edge_weight=12.,contact_width=8):
  """以明确种子约束前景归属；不加载 SAM 或人物检测模型。"""
  from .auto_regions import partition_foreground,label_layout,label_image,overlay_image,reference_key
  from PIL import Image
  width,height,count=seeds['width'],seeds['height'],seeds['count']
  if image.shape[0]!=1 or seeds['source_id']!=reference_key(image,width,height,count):raise ValueError('Identity seeds must match the reference image, canvas and identity count.')
  alpha=foreground_mask.detach().float().cpu()
  if alpha.ndim!=3 or alpha.shape[0]!=1 or not torch.isfinite(alpha).all():raise ValueError('Expected one finite foreground MASK.')
  ratio=analysis_resolution/max(width,height);w=max(16,round(width*ratio));h=max(16,round(height*ratio))
  rgb=b.to_rgb(image,w,h);prob=F.interpolate(alpha[None],size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy().clip(0,1)
  if not np.any(prob>mask_threshold):raise ValueError('The foreground model found no usable subject. Use a real subject image, review the foreground preview or adjust the threshold. Seed strokes alone do not define a completed subject mask.')
  manual=np.array(Image.fromarray(seeds['labels']).resize((w,h),Image.Resampling.NEAREST))
  labels,_,uncertain=partition_foreground(rgb,prob,[],backend='foreground',foreground=prob>mask_threshold,manual_labels=manual,manual_count=count,edge_weight=edge_weight,contact_px=max(0,round(contact_width*ratio)))
  mapping=[dict(char_id=f'char{i}',source='foreground_seeds') for i in range(1,count+1)]
  layout=label_layout(labels,width,height,contact_width,image.detach().cpu(),mapping)
  layout['layout_id']=reference_key(image,width,height,count)+'-'+layout['layout_id']
  report=f'Foreground ownership: {count} seeded identities; analysis={w}x{h}; foreground and image-edge geodesic allocation; no SAM model.'
  return {'ui':{'text':[report]},'result':(layout,overlay_image(layout,image,uncertain),label_image(layout),report)}

class AnimaLLLiteLoader:
 def load(self,model_name,auto_download=True):
  if model_name not in ('outline','regional','tile'):raise ValueError('Unknown Anima LLLite preset')
  from .shape_control import load_lllite
  profiles={'outline':'any-test-v2','regional':'regional-exp-v3','tile':'tile-repair-v20'}
  path=ensure_model(model_name,auto_download);weights=load_lllite(path,profiles[model_name])
  return weights,f'Validated {model_name} model: {Path(path).name}'

class AnimaModelDownload:
 OUTPUT_NODE=True
 def download(self,model_name):
  path=ensure_model(model_name,True)
  return {'ui':{'text':[f'Model ready: {path}']},'result':(path,)}

class AnimaContactSAM3Segment:
 def segment(self,image,semantic_mode,semantic_text,expected_people,confidence,mask_threshold,nms_iou,minimum_pixels,canvas_width,canvas_height,contact_width,device='auto',auto_download=True):
  from .sam3_segmentation import concepts,require_sam3,infer_instances,instance_layout
  if image.ndim!=4 or image.shape[0]!=1:raise ValueError('SAM3 requires one RGB image')
  if canvas_width%16 or canvas_height%16:raise ValueError('Canvas dimensions must be divisible by 16')
  queries=concepts(semantic_mode,semantic_text);require_sam3();path=ensure_model('sam3',auto_download)
  rgb=b.to_rgb(image,image.shape[2],image.shape[1]);candidates=infer_instances(rgb,path,queries,confidence,mask_threshold,device)
  layout,union,mapping=instance_layout(rgb,candidates,canvas_width,canvas_height,expected_people,nms_iou,minimum_pixels,contact_width)
  report='SAM3 concepts: '+', '.join(queries)+'\n'+'\n'.join(f"{r['char_id']}: {r['concept']} / {r['confidence']:.3f}" for r in mapping)
  return {'ui':{'text':[report]},'result':(layout,b.overlay_image(layout,image),b.label_image(layout),union,len(mapping),report)}

NODE_CLASS_MAPPINGS={name:globals()[name] for name in SCHEMA}
NODE_DISPLAY_NAME_MAPPINGS={name:spec['display_name'] for name,spec in SCHEMA.items()}
for name,cls in NODE_CLASS_MAPPINGS.items():
 meta=SCHEMA[name];cls.INPUT_TYPES=classmethod(schema);cls.CATEGORY=meta['category'];cls.DESCRIPTION=meta['description'];cls.RETURN_TYPES=tuple(meta['outputs']);cls.RETURN_NAMES=tuple(meta['output_names']);cls.FUNCTION=meta['function']
