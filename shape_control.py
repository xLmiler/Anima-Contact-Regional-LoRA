"""只将轮廓/色块送入LLLite；前期结构控制可渐退，后续继续区域身份路由。"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass,field
import re
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

KEY='anima_contact_shape_v07'
TARGET=re.compile(r'lllite_dit_blocks_(\d+)_(self_attn_[qkv]_proj|cross_attn_q_proj|mlp_layer1)$')

@dataclass
class LLLiteWeights:
    patcher:object
    metadata:dict
    filename:str
    profile:str

@dataclass
class ShapeSpec:
    weights:LLLiteWeights
    image:torch.Tensor
    effect_mask:torch.Tensor|None
    strength:float
    start_percent:float
    end_percent:float
    start_t:float
    end_t:float
    fade_t:float
    second_pass_strength:float=0.
    runtime:dict=field(default_factory=dict)

def load_lllite(path,profile):
    import comfy.utils,comfy.ops,comfy.model_management as mm,comfy.model_patcher as patchers
    from safetensors import safe_open
    from .vendor.lllite_native import AnimaLLLite
    with safe_open(path,framework='pt',device='cpu') as f:meta=dict(f.metadata() or {})
    title=meta.get('modelspec.title','').lower()
    if profile=='tile-repair-v20' and 'tile' not in title:raise ValueError('不是Tile用途的Anima LLLite权重')
    if profile=='any-test-v2' and not ('anytest' in title or 'any-test' in title):raise ValueError('该权重元数据不是any-test模型；请核对权重，或明确选择custom-v2自行验证')
    if profile=='regional-exp-v3' and 'regional' not in title:raise ValueError('该权重元数据不是区域色块模型；不能把其他LLLite当作regional使用')
    sd=comfy.utils.load_torch_file(path,safe_load=True)
    if 'lllite_conditioning1.conv1.weight' not in sd:raise ValueError('不是Anima named-key v2 LLLite；不要选择SDXL、Flux、VACE或人物LoRA')
    if sd['lllite_conditioning1.conv1.weight'].shape[1]!=3:raise ValueError('结构简图需要3通道LLLite；本节点不加载4通道inpainting权重')
    with torch.device('meta'):net=AnimaLLLite(sd,meta,dtype=comfy.utils.weight_dtype(sd),operations=comfy.ops.manual_cast)
    if net.model_dim!=2048 or net.block_count!=28:raise ValueError('此接入需要Anima 2048维、28层LLLite；扩层底模需单独适配')
    net.load_state_dict(sd,strict=True,assign=True);del sd;net.eval().requires_grad_(False)
    cls=getattr(patchers,'CoreModelPatcher',patchers.ModelPatcher)
    patcher=cls(net,load_device=mm.get_torch_device(),offload_device=mm.unet_offload_device())
    return LLLiteWeights(patcher,meta,str(path),profile)

def t_at(source,percent):
    sampling=source.get_model_object('model_sampling')
    return float(sampling.timestep(torch.as_tensor(sampling.percent_to_sigma(percent))).max())

def make_spec(weights,image,effect_mask,source,strength,start,end,fade,second_strength):
    if not 0<=start<end<=1:raise ValueError('须满足0≤start<end≤1')
    if not 0<=fade<=end-start:raise ValueError('渐退长度不能超过本轮控制区间')
    if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]<3:raise ValueError('控制图需要单张RGB IMAGE')
    validate_condition_image(image,weights.profile)
    return ShapeSpec(weights,image[:,:,:,:3].detach().cpu(),None if effect_mask is None else effect_mask.detach().cpu(),
        float(strength),float(start),float(end),t_at(source,start),t_at(source,end),t_at(source,end-fade),float(second_strength))

def validate_condition_image(image,profile):
    if not torch.isfinite(image).all() or float(image.min())<0 or float(image.max())>1:raise ValueError('结构图像素必须有限且在0~1之间')
    if profile=='any-test-v2':
        # 当前封装专门用于标签轮廓，阻止误接RGB/深度等高信息条件。
        a=image[:,:,:,:3]
        if float((a.amax(-1)-a.amin(-1)).amax())>.02:raise ValueError('轮廓CN应接结构简图的黑白输出，不要接原参考RGB或带颜色种子的图')
        gray=a[...,0]
        if float(((gray>.08)&(gray<.92)).float().mean())>.08:raise ValueError('输入含大量灰度明暗，可能误接深度/照片；本轮廓封装应接标签生成的稀疏黑白线图')
    elif profile=='regional-exp-v3':
        # 纯色标签可有少量边缘色，但不应是纹理复杂的参考图。
        sampled=image[0,::max(1,image.shape[1]//64),::max(1,image.shape[2]//64),:3]
        if len(torch.unique((sampled*255).round().reshape(-1,3),dim=0))>32:raise ValueError('区域CN需要白底少量纯色标签，不要接原图、种子覆盖图或深度图')

def strength_at(current,start_t,end_t,fade_t,strength):
    # start/end先通过真实model_sampling映射；不按本轮steps或首次调用猜进度。
    lo,hi=min(start_t,end_t),max(start_t,end_t)
    if current<lo or current>hi:return 0.
    if fade_t==end_t:return float(strength)
    fraction=(current-end_t)/(fade_t-end_t)
    return float(strength)*min(1.,max(0.,fraction))

def attach_shape(model,pipeline,spec):
    dit=model.get_model_object('diffusion_model');net=spec.weights.patcher.model
    if len(dit.blocks)!=net.block_count:raise ValueError('LLLite层数与底模不同')
    if not hasattr(model,'set_additional_models'):raise RuntimeError('ComfyUI缺少set_additional_models；请更新核心')
    if spec.strength>0:model.set_additional_models(KEY,[spec.weights.patcher])
    pipeline['shape_control']=spec
    pipeline['bundle']['shape_control']=spec if spec.strength>0 else None

def rebuild_shape(old,pipeline):
    if old.second_pass_strength<=0:return None
    return make_spec(old.weights,old.image,old.effect_mask,pipeline['source_model'],old.second_pass_strength,
                     old.start_percent,old.end_percent,min(old.end_percent-old.start_percent,.15),old.second_pass_strength)

def layout_control(layout,mode='outline',smooth=2,line_width=3,margin=16,release_mask=None):
    h,w=layout['height'],layout['width']
    if 'raster_masks' not in layout:raise ValueError('请接人物实例布局；不把旧矩形框当作精确轮廓')
    layers=[]
    for mask in layout['raster_masks'].values():
        m=F.interpolate(mask.float()[:,None],size=(h,w),mode='nearest')[0,0].numpy()>.5
        if smooth:m=ndimage.gaussian_filter(m.astype(np.float32),float(smooth))>.5
        layers.append(m)
    union=np.any(layers,0);image=np.ones((h,w,3),np.float32)
    if mode=='outline':
        edges=np.zeros((h,w),bool)
        for m in layers:edges|=m^ndimage.binary_erosion(m,iterations=max(1,int(line_width)))
        image[edges]=0
    elif mode=='region_color':
        palette=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[.5,0,0],[0,.5,0],[0,0,.5],[1,.5,0],[0,.5,1],[.5,0,1],[1,0,.5],[.5,1,0],[0,1,.5],[.5,.5,.5]],np.float32)
        for i,m in enumerate(layers):image[m]=palette[i]
    else:raise ValueError('未知结构简图类型')
    effect=ndimage.maximum_filter(union.astype(np.float32),size=2*int(margin)+1)
    effect=ndimage.gaussian_filter(effect,2).clip(0,1)
    if release_mask is not None:
        r=F.interpolate(release_mask.float().reshape(-1,1,*release_mask.shape[-2:])[:1],size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy().clip(0,1)
        effect*=1-r
    return torch.from_numpy(image)[None],torch.from_numpy(effect.copy())[None]

def effect_gate(mask,value):
    if value.ndim==5:
        g=F.interpolate(mask.float().reshape(-1,1,*mask.shape[-2:])[:1],size=value.shape[2:4],mode='bilinear',align_corners=False)[0,0]
        return g[None,None,:,:,None].to(value)
    if value.ndim==3:
        # 当前Anima T=1，序列长度与本轮条件token数一致，二维形状由调用方提前重建。
        if mask.numel()!=value.shape[1]:raise ValueError('LLLite效果遮罩token数与模型不一致')
        return mask.reshape(1,-1,1).to(value)
    raise ValueError('LLLite投影输入形状不支持')

@contextmanager
def scoped_lllite(dit,spec,args,kwargs):
    t=args[1] if len(args)>1 else kwargs.get('timesteps')
    current=float(t.detach().max());amount=strength_at(current,spec.start_t,spec.end_t,spec.fade_t,spec.strength)
    if amount<=0 or (spec.effect_mask is not None and float(spec.effect_mask.max())<=0):
        spec.runtime['skipped']=spec.runtime.get('skipped',0)+1
        yield;return
    x=args[0] if args else kwargs['x']
    if x.ndim==5 and x.shape[2]!=1:raise ValueError('当前LLLite只支持单帧Anima')
    net=spec.weights.patcher.model;handles=[];state={};count=[0]
    try:
        ph=int(getattr(dit,'patch_spatial',2));th=(x.shape[-2]+ph-1)//ph;tw=(x.shape[-1]+ph-1)//ph
        image=F.interpolate(spec.image.movedim(-1,1),size=(th*16,tw*16),mode='bilinear',align_corners=False).clamp(0,1)
        state['conditioning']=net.encode_conditioning(image.to(x)*2-1)
        gate=None if spec.effect_mask is None else F.interpolate(spec.effect_mask.float().reshape(-1,1,*spec.effect_mask.shape[-2:])[:1],size=(th,tw),mode='bilinear',align_corners=False)[0,0]
        for name in sorted(net.module_names):
            match=TARGET.fullmatch(name)
            if match is None:raise ValueError('未知LLLite目标层：'+name)
            index=int(match[1]);target=match[2]
            if target.startswith('self_attn_'):layer=getattr(dit.blocks[index].self_attn,target[len('self_attn_'):])
            elif target=='cross_attn_q_proj':layer=dit.blocks[index].cross_attn.q_proj
            elif target=='mlp_layer1':layer=dit.blocks[index].mlp.layer1
            else:raise ValueError('未知LLLite目标层')
            def pre(module,inputs,index=index,target=target):
                value=inputs[0];new=net.apply(value,state['conditioning'],index,target,amount)
                if gate is not None:new=value+(new-value)*effect_gate(gate,new)
                count[0]+=1
                return (new,*inputs[1:])
            handles.append(layer.register_forward_pre_hook(pre))
        yield
        if not count[0]:raise RuntimeError('LLLite处于有效时间区间，但没有任何控制模块执行')
        spec.runtime.update(calls=spec.runtime.get('calls',0)+1,last_strength=amount,modules_last=count[0])
    finally:
        for handle in reversed(handles):handle.remove()
        state.clear()
