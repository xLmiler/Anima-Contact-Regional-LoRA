"""离散身份标签清理：同一像素只属于一个人，保留明确画笔修正。"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from .auto_regions import label_layout

def clean_labels(labels, island_pixels=24, hole_pixels=64, smooth_radius=.6, protect=None):
    src=np.asarray(labels,dtype=np.uint8)
    out=src.copy(); ids=[int(x) for x in np.unique(src) if x>0]
    if not ids or max(ids)>16:raise ValueError('清理需要1~16个连续人物ID，背景为0')
    protected=np.zeros(src.shape,bool) if protect is None else np.asarray(protect,bool)
    if protected.shape!=src.shape:raise ValueError('画笔保护区尺寸不符')
    # 最大连通块始终保留，避免小人物被当作噪点删除；独立发饰等可调低阈值保留。
    remove=np.zeros(src.shape,bool)
    for cid in ids:
        cc,n=ndi.label(out==cid)
        sizes=np.bincount(cc.ravel()); sizes[0]=0
        if n:
            bad=(sizes<=int(island_pixels));bad[0]=False;bad[int(sizes.argmax())]=False
            remove|=bad[cc]&~protected
    if remove.any():
        stable=~remove
        nearest=ndi.distance_transform_edt(~stable,return_distances=False,return_indices=True)
        out[remove]=out[tuple(nearest[:,remove])]
    # 仅补封闭小孔，不填贯通背景，也不把两人间的大缝隙强行连起来。
    cc,n=ndi.label(out==0);sizes=np.bincount(cc.ravel())
    edge_ids=np.unique(np.r_[cc[0],cc[-1],cc[:,0],cc[:,-1]])
    filled=0
    for k in np.flatnonzero((sizes>0)&(sizes<=int(hole_pixels))):
        if k==0 or k in edge_ids:continue
        hole=cc==k
        if (hole&protected).any():continue
        rim=ndi.binary_dilation(hole)&~hole
        votes=np.bincount(out[rim],minlength=17);votes[0]=0
        cid=int(votes.argmax())
        if cid and votes[cid]>=.75*votes.sum():out[hole]=cid;filled+=int(hole.sum())
    if smooth_radius>0:
        # 在统一标签上竞争，避免分别膨胀各人物产生重叠和白缝。
        best=np.full(src.shape,-1.,np.float32);winner=out.copy()
        for cid in [0]+ids:
            score=ndi.gaussian_filter((out==cid).astype(np.float32),float(smooth_radius))
            choose=score>best;winner[choose]=cid;best[choose]=score[choose]
        for cid in ids:
            if not np.any(winner==cid):winner[out==cid]=cid
        out=winner
    out[protected]=src[protected]
    return out,dict(changed_pixels=int((out!=src).sum()),removed_pixels=int(remove.sum()),filled_pixels=filled)

def layout_labels(layout):
    from PIL import Image
    h,w=layout['height'],layout['width'];a=np.zeros((h,w),np.uint8)
    for ch in layout['characters']:
        mask=layout['raster_masks'][ch['id']][0].cpu().numpy()
        if mask.shape!=(h,w):mask=np.asarray(Image.fromarray(mask).resize((w,h),Image.Resampling.NEAREST))
        a[mask>.5]=int(ch['id'][4:])
    return a

def clean_layout(layout,island_pixels=24,hole_pixels=64,smooth_radius=.6,protect=None):
    if 'raster_masks' not in layout:raise ValueError('清理节点需要实例轮廓，不接受矩形布局')
    labels,stats=clean_labels(layout_labels(layout),island_pixels,hole_pixels,smooth_radius,protect)
    out=label_layout(labels,layout['width'],layout['height'],layout.get('contact_pixels',10),layout.get('reference_image'),layout.get('person_mapping'))
    if len(out['characters'])!=len(layout['characters']):raise ValueError('清理导致人物丢失，请调低阈值')
    out['cleanup_stats']=stats
    return out
