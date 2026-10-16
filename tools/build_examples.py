"""生成中文示例工作流和可拖入加载图像的程序化参考 PNG。"""
import copy,importlib.util,json,sys,types,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
fp=types.ModuleType('folder_paths');fp.models_dir=str(ROOT.parent/'build_models');fp.folder_names_and_paths={};fp.get_filename_list=lambda n:[];sys.modules['folder_paths']=fp
spec=importlib.util.spec_from_file_location('anima_examples',ROOT/'__init__.py',submodule_search_locations=[str(ROOT)]);pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from anima_examples.nodes import SCHEMA
from anima_examples.reference_guides import make_guide
from anima_examples.shape_control import layout_control
from PIL import Image
ZH=json.loads((ROOT/'locales/zh/nodeDefs.json').read_text())
BUILTINS={
 'UNETLoader':([], [('unet_name','anima-base-v1.0.safetensors'),('weight_dtype','default')],[('MODEL','MODEL')]),
 'ModelSamplingAuraFlow':([('model','MODEL')],[('shift',3.)],[('MODEL','MODEL')]),
 'CLIPLoader':([],[('clip_name','qwen_3_06b_base.safetensors'),('type','stable_diffusion'),('device','default')],[('CLIP','CLIP')]),
 'VAELoader':([],[('vae_name','qwen_image_vae.safetensors')],[('VAE','VAE')]),
 'KSampler':([('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],[('seed',813579),('control_after_generate','fixed'),('steps',28),('cfg',4.),('sampler_name','euler'),('scheduler','normal'),('denoise',1.)],[('LATENT','LATENT')]),
 'EmptyLatentImage':([],[('width',1024),('height',1024),('batch_size',1)],[('LATENT','LATENT')]),
 'VAEDecode':([('samples','LATENT'),('vae','VAE')],[],[('IMAGE','IMAGE')]),
 'SaveImage':([('images','IMAGE')],[('filename_prefix','AnimaContact')],[]),
 'PreviewImage':([('images','IMAGE')],[],[]),
 'LoadImage':([],[('image',''),('upload','image')],[('IMAGE','IMAGE'),('MASK','MASK')])}
class Graph:
 def __init__(self):self.nodes=[];self.links=[];self.groups=[];self.n=0;self.group=None
 def addgroup(self,name,x,y,w,h):self.group=dict(id=len(self.groups)+1,title=name,bounding=[x,y,w,h],color='#345763',font_size=24,flags={});self.groups.append(self.group)
 def add(self,kind,x,y,values=None):
  self.n+=1;ins=[];ws=[];outs=[]
  if kind in SCHEMA:
   d=SCHEMA[kind]
   for section in d['inputs'].values():
    for k,e in section.items():
     t=e[0];o=e[1]
     if isinstance(t,list) or t in ('INT','FLOAT','STRING','BOOLEAN') and not o.get('forceInput'):ws.append((k,o.get('default',t[0] if isinstance(t,list) else 0)))
     else:ins.append((k,t))
   outs=list(zip(d['output_names'],d['outputs']))
  else:ins,ws,outs=BUILTINS[kind]
  named=dict(ws);named.update(values or {});node=dict(id=self.n,type=kind,pos=[x,y],size=[520,850 if kind in ['AnimaOwnershipEditor','AnimaIdentity'] else max(170,100+len(ws)*28)],flags={},order=self.n,mode=0,inputs=[dict(name=k,type=t,link=None) for k,t in ins],outputs=[dict(name=k,type=t,links=[],slot_index=i) for i,(k,t) in enumerate(outs)],properties={'Node name for S&R':kind},widgets_values=list(named.values()),widgets_values_named=named)
  if kind in SCHEMA:
   node['properties']['animaWidgetState']={'version':1,'type':kind,'values':copy.deepcopy(named)}
   node['properties']['DocsGroup']=self.group['id']
  self.nodes.append(node);return node
 def connect(self,a,slot,b,key):
  idx=next((i for i,e in enumerate(b['inputs']) if e['name']==key),None)
  if idx is None:idx=len(b['inputs']);b['inputs'].append(dict(name=key,type=a['outputs'][slot]['type'],widget={'name':key},link=None))
  assert a['outputs'][slot]['type']==b['inputs'][idx]['type'],(a['type'],b['type'],key)
  n=len(self.links)+1;b['inputs'][idx]['link']=n;a['outputs'][slot]['links'].append(n);self.links.append([n,a['id'],slot,b['id'],idx,a['outputs'][slot]['type']])
 def note(self,title,text,x,y,w=460,h=500,**props):
  self.n+=1;n=dict(id=self.n,type='Note',pos=[x,y],size=[w,h],flags={},mode=0,order=0,inputs=[],outputs=[],title=title,properties=props,widgets_values=[text]);self.nodes.append(n)
 def save(self,name):
  # Plugin documentation stays to the left of the corresponding group.
  for group in self.groups:
   ops=[n for n in self.nodes if n.get('properties',{}).get('DocsGroup')==group['id']];cursor=group['bounding'][1]+80
   for n in ops:
    meta=ZH[n['type']];text=meta['description']+'\n\n'+'\n\n'.join(e['name']+'：'+e['tooltip'] for e in meta['inputs'].values())
    y=max(cursor,n['pos'][1]) if n['type']!='AnimaIdentity' else n['pos'][1]
    self.note(meta['display_name'],text,group['bounding'][0]+20,y,h=min(1050,max(300,len(text)*.42)),DocsFor=n['id']);cursor=y+self.nodes[-1]['size'][1]+35
   group['bounding'][3]=max(group['bounding'][3],cursor-group['bounding'][1]+40)
  from anima_examples.model_store import MODELS,SAM3_REPO,SAM3_REVISION
  text='Anima 底模架构：Base 1.0，28 层。\n底模、文本编码器、VAE：https://huggingface.co/circlestone-labs/Anima\nmodels/diffusion_models\nmodels/text_encoders\nmodels/vae\n\n插件启动时创建缺失目录；节点执行且启用自动下载时获取模型。\n\n'
  if self.route=='sam3':text+='SAM3：models/sam3/facebook-sam3\nhttps://huggingface.co/'+SAM3_REPO+'/tree/'+SAM3_REVISION+'\n安装 requirements_sam3.txt；公开仓库无需登录，执行节点可自动下载。\n\n'
  keys=['outline']+(['tile'] if self.upscale else [])+(['detector-small','sam2-tiny'] if self.route=='sam2' else ['isnet','toonout'] if self.route=='foreground' else [])
  text+='\n\n'.join(k+'\nmodels/'+MODELS[k][0]+'\n'+MODELS[k][1] for k in keys)
  if self.route=='sam2':text+='\n\nSAM2 需要安装官方 sam2 Python 软件包：https://github.com/facebookresearch/sam2'
  self.note('所需模型、来源与放置目录',text,-850,0,w=750,h=1900,ModelSources=True)
  data=dict(id=str(uuid.uuid5(uuid.NAMESPACE_DNS,name)),revision=0,last_node_id=self.n,last_link_id=len(self.links),nodes=self.nodes,links=self.links,groups=self.groups,config={},extra={'ds':{'scale':.45,'offset':[50,50]}},version=.4)
  data['extra']['animaExample']={'people':self.count,'route':self.route,'tile':self.upscale}
  (ROOT/'example_workflows'/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))

def build(count,route='sam3',upscale=False):
 g=Graph();g.route=route;g.upscale=upscale;g.count=count;g.addgroup('底模与共享场景',0,0,1800,1900)
 unet=g.add('UNETLoader',520,80);sampling=g.add('ModelSamplingAuraFlow',520,430);g.connect(unet,0,sampling,'model');clip=g.add('CLIPLoader',1100,80);vae=g.add('VAELoader',1100,430)
 scene=('2girls, hugging, facing each other, side view, arms around each other, upper body, white background' if count==2 else '3girls, standing in a row, holding hands, full body, looking at viewer, white background' if count==3 else f'{count}girls, multiple girls, group portrait, standing in staggered rows, waving, raised hands, looking at viewer, white background')
 cond=g.add('AnimaSceneConditioning',520,950,{'scene_prompt':scene,'quality_prompt':''});g.connect(clip,0,cond,'clip')
 g.addgroup('参考图与人物归属',1900,0,1800,3200)
 w,h={2:(1024,1024),3:(1344,896),5:(1536,1024),7:(1536,1152)}[count]
 image=g.add('LoadImage',2420,100,{'image':''})
 if route=='sam3':
  segment=g.add('AnimaContactSAM3Segment',3000,100,{'canvas_width':w,'canvas_height':h,'expected_people':count})
  g.connect(image,0,segment,'image');paint=g.add('AnimaOwnershipEditor',3000,1050);g.connect(segment,0,paint,'layout')
  preview=g.add('PreviewImage',2420,900);g.connect(segment,1,preview,'images')
  g.note('参考图与语义分割','将对应人数的真实人物参考图拖入加载图像。自动语义探测人物；需要限制目标时改为自定义语义并填写简短英语概念。运行后核对归属预览与图例，再调整身份条件。\nexamples/reference_guides 内 PNG 是动作几何示意，不是语义模型识别精度样本。',2420,1700,w=520,h=280)
 elif route=='sam2':
  detector=g.add('AnimaPersonDetect',2420,800,{'expected_people':count})
  segment=g.add('AnimaPersonSegment',3000,100,{'canvas_width':w,'canvas_height':h,'analysis_resolution':768,'refinement':'mask_refine'})
  g.connect(image,0,detector,'image');g.connect(image,0,segment,'image');g.connect(detector,0,segment,'people')
  paint=g.add('AnimaOwnershipEditor',3000,1050);g.connect(segment,0,paint,'layout');preview=g.add('PreviewImage',2420,1400);g.connect(segment,1,preview,'images')
  g.note('参考图与实例轮廓','将双人参考图拖入加载图像。人物检测器提供边界框，SAM2 在框内预测实例轮廓。运行后检查交叠肢体归属，使用归属编辑器修正。\n人物数量核对不一致时，先检查检测框及检测阈值。',2420,2100,w=520,h=260)
 elif route=='foreground':
  foreground=g.add('AnimaForeground',3000,100,{'backend':'toonout'})
  seeds=g.add('AnimaIdentitySeeds',2420,800,{'canvas_width':w,'canvas_height':h,'expected_people':count,'strokes_json':'{"version":1,"strokes":[]}'})
  segment=g.add('AnimaContactForegroundOwnership',3000,750)
  for node,key in [(foreground,'image'),(seeds,'image'),(segment,'image')]:g.connect(image,0,node,key)
  g.connect(foreground,0,segment,'foreground_mask');g.connect(seeds,0,segment,'seeds')
  paint=g.add('AnimaOwnershipEditor',3000,1550);g.connect(segment,0,paint,'layout');preview=g.add('PreviewImage',2420,1850);g.connect(foreground,1,preview,'images')
  g.note('前景与人物归属','拖入真实双人参考图。先局部执行身份种子节点以载入画布，在每个人物内部标注身份种子，交叠肢体补充对应身份种子，然后队列整个工作流。更换参考图后旧种子自动清空。\n前景模型输出人物并集，身份种子负责区分人物；此流程不加载 SAM 或人物检测模型。\nexamples/reference_guides 中 PNG 仅演示动作和归属几何，不是前景模型的识别样本。',2420,2500,w=520,h=350)
 else:raise ValueError('Unknown segmentation route')
 g.addgroup('独立身份条件',3850,0,1250,1050*count+150);last=None
 hairs=['blue hair, short hair, blue eyes','orange hair, long hair, brown eyes','green hair, bob cut, green eyes','pink hair, ponytail, violet eyes','purple hair, long hair, purple eyes','white hair, short hair, red eyes','brown hair, long hair, brown eyes']
 for i in range(count):
  ident=g.add('AnimaIdentity',4370,100+i*1050,{'char_id':f'char{i+1}','identity_prompt':hairs[i]+', simple dress','lora_stack_json':'{"version":1,"entries":[]}'})
  for src,slot,key in [(clip,0,'clip'),(paint,0,'layout'),(cond,2,'scene_text'),(cond,3,'scene_policy')]:g.connect(src,slot,ident,key)
  if last:g.connect(last,0,ident,'previous_roles')
  last=ident
 g.addgroup('区域路由与结构控制',5250,0,1800,3000)
 apply=g.add('AnimaRegionalApply',5770,100);g.connect(sampling,0,apply,'model');g.connect(last,0,apply,'roles');g.connect(paint,0,apply,'layout')
 control=g.add('AnimaLayoutControl',5770,1150);g.connect(paint,0,control,'layout');weights=g.add('AnimaLLLiteLoader',6350,1150,{'model_name':'outline'})
 cn=g.add('AnimaContactLLLiteApply',6350,100);g.connect(apply,1,cn,'pipeline');g.connect(weights,0,cn,'control');g.connect(control,0,cn,'image');g.connect(control,1,cn,'effect_mask')
 view=g.add('PreviewImage',6350,1800);g.connect(control,0,view,'images')
 g.addgroup('采样与诊断',7200,0,1800,2200)
 latent=g.add('EmptyLatentImage',7720,100,{'width':w,'height':h});g.connect(paint,4,latent,'width');g.connect(paint,5,latent,'height')
 sample=g.add('KSampler',8300,100);g.connect(cn,0,sample,'model');g.connect(cond,0,sample,'positive');g.connect(cond,1,sample,'negative');g.connect(latent,0,sample,'latent_image')
 decode=g.add('VAEDecode',7720,850);g.connect(sample,0,decode,'samples');g.connect(vae,0,decode,'vae');save=g.add('SaveImage',8300,850);g.connect(decode,0,save,'images')
 report=g.add('AnimaRegionalInspect',7720,1400);g.connect(cn,1,report,'pipeline');g.connect(sample,0,report,'samples');g.connect(decode,0,report,'image');view=g.add('PreviewImage',8300,1400);g.connect(report,0,view,'images')
 if upscale:
  g.addgroup('Tile 二采精修',9150,0,2350,2200);tile=g.add('AnimaLLLiteLoader',9670,100,{'model_name':'tile'});bridge=g.add('AnimaTileUpscale',9670,650);g.connect(cn,1,bridge,'pipeline');g.connect(sample,0,bridge,'samples');g.connect(vae,0,bridge,'vae');g.connect(tile,0,bridge,'tile_control')
  second=g.add('KSampler',10250,100,{'steps':20,'denoise':.2});g.connect(bridge,0,second,'model');g.connect(bridge,1,second,'latent_image');g.connect(cond,0,second,'positive');g.connect(cond,1,second,'negative');d=g.add('VAEDecode',10250,850);g.connect(second,0,d,'samples');g.connect(vae,0,d,'vae');o=g.add('SaveImage',10830,850);g.connect(d,0,o,'images')
 g.save('07_CN_双人_Tile二采放大' if upscale else {'sam3':'01_CN_双人_SAM3','sam2':'02_CN_双人_SAM2','foreground':'03_CN_双人_前景分割'}[route] if count==2 else {3:'04_CN_三人',5:'05_CN_五人',7:'06_CN_七人'}[count])

if __name__=='__main__':
 out=ROOT/'example_workflows';out.mkdir(exist_ok=True)
 for p in out.rglob('*.json'):p.unlink()
 for route in ['sam3','sam2','foreground']:build(2,route)
 for n in [3,5,7]:build(n,'sam3')
 build(2,'sam3',True)
 folder=ROOT/'examples/reference_guides';folder.mkdir(parents=True,exist_ok=True)
 for n in [2,3,5,7]:
  w,h={2:(1024,1024),3:(1344,896),5:(1536,1024),7:(1536,1152)}[n];layout,rgb,ids=make_guide(n,w,h);outline,_=layout_control(layout,'outline',1,3,16)
  for name,tensor in [('reference',rgb),('identities',ids),('outline',outline)]:Image.fromarray((tensor[0].numpy()*255).astype('uint8')).save(folder/f'{n}_{name}.png')
 print('7 focused workflows; 12 generated reference images')
