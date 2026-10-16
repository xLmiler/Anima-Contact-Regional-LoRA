"""SAM3 概念实例分割；直接保留实例轮廓，不用几何距离重新切分人物。"""
from __future__ import annotations
import gc
import numpy as np
import torch
from PIL import Image
from .auto_regions import label_layout,reference_key

AUTO_CONCEPTS=('person','girl','boy')

def concepts(mode,text):
    if mode=='automatic':return list(AUTO_CONCEPTS)
    if mode!='custom':raise ValueError('Unknown SAM3 semantic mode')
    items=list(dict.fromkeys(x.strip() for x in text.splitlines() if x.strip()))
    if not items or len(items)>16 or any(len(x)>200 for x in items):raise ValueError('Provide 1–16 concept phrases, one per line, each at most 200 characters.')
    return items

def require_sam3():
    try:
        from transformers import Sam3Model,Sam3Processor
    except ImportError as e:
        raise RuntimeError('SAM3 requires a Transformers build with Sam3Model and Sam3Processor. Install requirements_sam3.txt using the ComfyUI Python environment, then restart. Other segmentation nodes remain independent.') from e
    return Sam3Model,Sam3Processor

def infer_instances(rgb,path,queries,confidence,mask_threshold,device='auto'):
    Model,Processor=require_sam3()
    if device=='auto':
        try:
            import comfy.model_management as mm
            target=mm.get_torch_device()
            mm.free_memory(5*1024**3,target)
        except ImportError:target=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:target=torch.device(device)
    dtype=torch.float16 if target.type=='cuda' else torch.float32
    model=Model.from_pretrained(path,local_files_only=True,dtype=dtype,attn_implementation='sdpa').eval().to(target)
    processor=Processor.from_pretrained(path,local_files_only=True)
    candidates=[]
    try:
        with torch.inference_mode():
            inputs=processor(images=Image.fromarray(rgb),text=queries[0],return_tensors='pt').to(target)
            inputs['pixel_values']=inputs['pixel_values'].to(dtype)
            vision=model.get_vision_features(pixel_values=inputs['pixel_values'])
            sizes=inputs['original_sizes'].tolist()
            for query in queries:
                text_inputs=processor(text=query,return_tensors='pt').to(target)
                outputs=model(vision_embeds=vision,**text_inputs)
                result=processor.post_process_instance_segmentation(outputs,threshold=confidence,mask_threshold=mask_threshold,target_sizes=sizes)[0]
                for mask,score,box in zip(result['masks'],result['scores'],result['boxes']):
                    candidates.append(dict(mask=mask.detach().cpu().numpy().astype(bool),score=float(score),box=box.detach().cpu().tolist(),concept=query))
    finally:
        del model,processor
        gc.collect()
        if target.type=='cuda':torch.cuda.empty_cache()
    return candidates

def instance_layout(rgb,candidates,width,height,expected_people=0,nms_iou=.85,minimum_pixels=24,contact_width=8):
    """掩码 IoU 去重；只在真实重叠像素按分数竞争，不裁切跨人肢体。"""
    selected=[]
    for candidate in sorted(candidates,key=lambda x:x['score'],reverse=True):
        mask=np.asarray(candidate['mask'],bool)
        if mask.shape!=rgb.shape[:2] or not np.isfinite(candidate['score']):raise ValueError('SAM3 returned an invalid mask or score')
        if int(mask.sum())<minimum_pixels:continue
        duplicate=False
        for kept in selected:
            union=(mask|kept['mask']).sum();iou=float((mask&kept['mask']).sum()/max(1,union))
            if iou>=nms_iou:duplicate=True;break
        if not duplicate:selected.append(dict(candidate,mask=mask))
    if not selected:raise ValueError('SAM3 found no matching instances. Review the concept phrases and confidence threshold.')
    if len(selected)>16:raise ValueError('SAM3 found more than 16 instances. Narrow the concepts or increase the confidence threshold.')
    if expected_people and len(selected)!=expected_people:raise ValueError(f'SAM3 found {len(selected)} instances; expected {expected_people}. Review the masks and concept phrases. Instances are not invented or silently discarded.')
    def center(c):
        y,x=np.nonzero(c['mask']);return float(x.mean()),float(y.mean())
    selected.sort(key=center)
    scores=np.stack([np.where(c['mask'],c['score'],-1.) for c in selected]);labels=(scores.argmax(0)+1).astype(np.uint8);labels[scores.max(0)<0]=0
    missing=[i+1 for i in range(len(selected)) if not np.any(labels==i+1)]
    if missing:raise ValueError(f'SAM3 instances {missing} have no exclusive visible pixels after overlap resolution. Refine concept phrases rather than reassigning hidden regions.')
    mapping=[dict(char_id=f'char{i+1}',source='sam3_concept',concept=c['concept'],confidence=c['score'],box=c['box']) for i,c in enumerate(selected)]
    image=torch.from_numpy(rgb.copy()).float()[None]/255
    layout=label_layout(labels,width,height,contact_width,reference=image,mapping=mapping)
    layout['layout_id']=reference_key(image,width,height,len(selected))+'-'+layout['layout_id']
    union=torch.from_numpy((labels>0).astype(np.float32))[None,None]
    union=torch.nn.functional.interpolate(union,size=(height,width),mode='nearest')[0]
    return layout,union,mapping
