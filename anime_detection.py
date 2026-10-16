"""二次元专用YOLOv8 ONNX人物检测，不加载通用YOLOX或伪造多人骨架。"""
from __future__ import annotations
import ast,json,os
import numpy as np
from PIL import Image,ImageDraw
import torch
from scipy import ndimage
from .auto_regions import sam2_config,sample_points,partition_foreground,label_layout,to_rgb

MODEL_REPO='deepghs/anime_person_detection'
MODEL_REVISION='e39c744c22432ad01f91dd254fe2b02c8d878b8c'

def nms(boxes,scores,iou_threshold=.75):
    order=np.argsort(-scores,kind='stable');keep=[]
    area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    while len(order):
        i=int(order[0]);keep.append(i);rest=order[1:]
        intersection=np.maximum(0,np.minimum(boxes[i,2:],boxes[rest,2:])-np.maximum(boxes[i,:2],boxes[rest,:2])).prod(1)
        iou=intersection/np.maximum(area[i]+area[rest]-intersection,1e-9)
        order=rest[iou<=iou_threshold]
    return keep

def decode_yolov8(output,width,height,input_width,input_height,confidence=.30,iou=.75):
    a=np.asarray(output)
    if a.ndim==3:
        if a.shape[0]!=1:raise ValueError('检测一次只接受一张参考图')
        a=a[0]
    if a.ndim!=2:raise ValueError('ONNX检测输出必须是二维候选矩阵')
    if a.shape[0]==5:a=a.T
    elif a.shape[1]!=5:raise ValueError('请选择单类人物YOLOv8原始ONNX权重；不支持COCO80类或端到端NMS格式')
    a=a[np.isfinite(a).all(1)&(a[:,4]>=confidence)]
    if not len(a):return []
    # 上游YOLOv8导出的是cx,cy,w,h,person_score，而不是xyxy或带objectness的v5。
    boxes=np.c_[a[:,:2]-a[:,2:4]/2,a[:,:2]+a[:,2:4]/2]
    indices=nms(boxes,a[:,4],iou)
    scale=np.array([width/input_width,height/input_height]*2)
    result=[]
    for i in indices:
        box=np.clip(boxes[i]*scale,[0,0,0,0],[width,height,width,height])
        if np.any(box[2:]-box[:2]<2):continue
        result.append({'box':box.tolist(),'confidence':float(a[i,4])})
    return sorted(result,key=lambda p:(p['box'][0]+p['box'][2])/2)

def detect_anime(image,model_path,confidence=.30,iou=.75,resolution=640):
    try:import onnxruntime as ort
    except ImportError as e:raise RuntimeError('Install onnxruntime in the ComfyUI Python environment.') from e
    ort.disable_telemetry_events()
    # 单次CPU会话，退出后不保留多模型缓存，也不与Anima抢GPU。
    opts=ort.SessionOptions();opts.intra_op_num_threads=min(4,os.cpu_count() or 1)
    session=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
    try:
        names=session.get_modelmeta().custom_metadata_map.get('names')
        if names:
            classes=ast.literal_eval(names)
            values=list(classes.values()) if isinstance(classes,dict) else classes
            if values!=['person']:raise ValueError('请选择单类动漫person模型；头部/人脸检测器不能当整个人物实例检测器')
        inputs=session.get_inputs()
        if len(inputs)!=1 or inputs[0].type!='tensor(float)':raise ValueError('检测模型应只有一个float32 RGB输入')
        shape=inputs[0].shape
        if len(shape)!=4 or shape[1]!=3:raise ValueError('检测模型输入须为NCHW RGB三通道')
        w=int(shape[3]) if isinstance(shape[3],int) else int(resolution)
        h=int(shape[2]) if isinstance(shape[2],int) else int(resolution)
        # 与imgutils的默认非动态预处理一致：RGB缩放到目标宽高，再独立缩回坐标。
        im=Image.fromarray(image).resize((w,h),Image.Resampling.BILINEAR)
        tensor=np.asarray(im,np.float32).transpose(2,0,1)[None]/255
        raw=session.run(None,{inputs[0].name:np.ascontiguousarray(tensor)})[0]
        return decode_yolov8(raw,image.shape[1],image.shape[0],w,h,confidence,iou)
    finally:del session

def box_preview(image,records):
    im=Image.fromarray(image);d=ImageDraw.Draw(im)
    for i,r in enumerate(records):
        color=('#55a9ff','#ffb05b','#72d9a1','#ec86b8')[i%4]
        b=r['box'];d.rectangle(b,outline=color,width=max(2,round(min(im.size)/200)))
        d.text((b[0]+5,b[1]+5),f'char{i+1}  {r["confidence"]:.3f}',fill=color,stroke_width=1,stroke_fill='white')
    return torch.from_numpy(np.array(im).copy()).float()[None]/255

def sam_boxes(rgb,records,checkpoint,manual_labels=None,refine=False):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:raise RuntimeError('自动轮廓需要官方SAM2；先运行检测检查节点，安装方法见文档。') from e
    model=build_sam2(sam2_config(checkpoint),checkpoint,device='cpu',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model);masks=[]
    try:
        with torch.inference_mode():
            predictor.set_image(rgb)
            for i,r in enumerate(records):
                # 框是真实检测结果；不用人造骨架推断手臂归属。
                kwargs={'box':np.asarray(r['box'],np.float32),'multimask_output':True}
                if manual_labels is not None:
                    positive=sample_points(manual_labels==i+1,24)
                    negative=sample_points((manual_labels>0)&(manual_labels!=i+1),32)
                    points=np.concatenate([positive,negative])
                    if len(points):kwargs.update(point_coords=points,point_labels=np.r_[np.ones(len(positive)),np.zeros(len(negative))].astype(np.int32))
                pred,quality,logits=predictor.predict(**kwargs)
                scores=np.asarray(quality).copy()
                if manual_labels is not None and len(points):
                    xy=points.astype(int);coverage=pred[:,xy[:,1],xy[:,0]]
                    if len(positive):scores+=3*coverage[:,:len(positive)].mean(1)
                    if len(negative):scores-=3*coverage[:,len(positive):].mean(1)
                best=int(np.argmax(scores))
                if refine:
                    # 官方交互精修接口：上一轮低分辨率logits作为mask提示，保留框/人工种子。
                    refined,rquality,_=predictor.predict(**dict(kwargs,multimask_output=False,mask_input=logits[best:best+1]))
                    accept=float(rquality[0])>=float(quality[best])-.05
                    if manual_labels is not None and len(points):
                        rc=refined[0,xy[:,1],xy[:,0]]
                        accept=accept and (not len(positive) or rc[:len(positive)].mean()>=coverage[best,:len(positive)].mean()) and (not len(negative) or rc[len(positive):].mean()<=coverage[best,len(positive):].mean())
                    if accept:pred[best]=refined[0]
                masks.append(pred[best].astype(bool))
    finally:
        if hasattr(predictor,'reset_predictor'):predictor.reset_predictor()
        del predictor,model
    return np.stack(masks)

def regions_from_boxes(rgb,records,checkpoint,width,height,analysis_resolution=512,contact_width=10,manual_labels=None,refine=False,foreground=None):
    ratio=analysis_resolution/max(width,height);w=max(16,round(width*ratio));h=max(16,round(height*ratio))
    small=to_rgb(rgb,w,h);sx=w/rgb.shape[1];sy=h/rgb.shape[0]
    scaled=[dict(r,box=(np.array(r['box'])*[sx,sy,sx,sy]).tolist()) for r in records]
    manual=None if manual_labels is None else np.array(Image.fromarray(manual_labels).resize((w,h),Image.Resampling.NEAREST))
    masks=sam_boxes(small,scaled,checkpoint,manual,refine=True) if refine else sam_boxes(small,scaled,checkpoint,manual)
    # 独占轮廓的内核提供自动身份种子；重叠区才交给边缘感知路径。
    exclusive=masks.sum(0)==1;seeds=np.zeros((h,w),np.uint8)
    for i,m in enumerate(masks):
        core=ndimage.binary_erosion(m&exclusive,iterations=2)
        if not core.any():core=m&exclusive
        if not core.any() and manual is not None:core=manual==i+1
        if not core.any():raise ValueError(f'char{i+1}没有独立轮廓：SAM把两人合并或完全遮挡。请接原图种子以区分；不能只依赖检测框。')
        seeds[core]=i+1
    if manual is not None:seeds[manual>0]=manual[manual>0]
    fg=masks.any(0)
    if foreground is not None:
        alpha=np.array(Image.fromarray(np.asarray(foreground,np.float32)).resize((w,h),Image.Resampling.BILINEAR))
        # 只补检测人物框内的语义前景；不扩展到整幅场景或改掉已有SAM归属。
        supports=np.zeros((h,w),bool)
        for r in scaled:
            x0,y0,x1,y1=np.rint(r['box']).astype(int);supports[max(0,y0):min(h,y1+1),max(0,x0):min(w,x1+1)]=True
        fg=fg|((alpha>.5)&supports)
    labels,_,uncertain=partition_foreground(small,fg.astype(np.float32),[],foreground=fg,
        contact_px=max(0,round(contact_width*ratio)),manual_labels=seeds,manual_count=len(records),sam_prior=masks)
    mapping=[dict(char_id=f'char{i+1}',source='anime_detector_sam2',confidence=r['confidence'],box=r['box']) for i,r in enumerate(records)]
    layout=label_layout(labels,width,height,contact_width,mapping=mapping)
    return layout,uncertain
