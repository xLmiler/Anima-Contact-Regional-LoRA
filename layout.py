"""归一化布局：同一组选框可用于首采、二采和不同长宽的画布。"""
from __future__ import annotations
import copy
import json
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

COLORS=["#55a9ff","#ffb05b","#72d9a1","#ec86b8","#a99bff","#f2d46d","#77d5de","#e89478","#d5ec86","#b775dc","#e76c64","#68b8b1","#c69f70","#8eabdb","#c8cfdf","#b2bf61"]
DEFAULT_DATA={"version":1,"characters":[{"id":"char1","label":"char1"},{"id":"char2","label":"char2"}],
 "regions":[
  {"id":"a-body","kind":"body","char_id":"char1","x":.06,"y":.04,"w":.50,"h":.92},
  {"id":"b-body","kind":"body","char_id":"char2","x":.44,"y":.04,"w":.50,"h":.92},
  {"id":"a-face","kind":"face","char_id":"char1","x":.16,"y":.025,"w":.36,"h":.40},
  {"id":"b-face","kind":"face","char_id":"char2","x":.50,"y":.025,"w":.36,"h":.40},
  {"id":"contact","kind":"contact","char_id":"","x":.43,"y":.40,"w":.14,"h":.38}]}
DEFAULT_JSON=json.dumps(DEFAULT_DATA,ensure_ascii=False,separators=(",",":"))

def parse_layout(raw,width,height):
    if width%16 or height%16 or not (64<=width<=8192 and 64<=height<=8192):
        raise ValueError("画布宽高须为 16 的倍数，范围 64~8192；Anima patch 与二采才能对齐")
    data=json.loads(raw) if isinstance(raw,str) else copy.deepcopy(raw)
    chars=data.get("characters",[])
    if not chars or len(chars)>16:raise ValueError("布局需要 1~16 个角色")
    ids=[c.get("id") for c in chars]
    if len(set(ids))!=len(ids) or any(i not in {f"char{x}" for x in range(1,17)} for i in ids):
        raise ValueError("角色 ID 必须为唯一的 char1~char16")
    regions=data.get("regions",[])
    if len(regions)>128:raise ValueError("区域数量超过 128")
    seen=set()
    for r in regions:
        if not r.get("id") or r["id"] in seen:raise ValueError("区域 ID 为空或重复")
        seen.add(r["id"])
        if r.get("kind") not in {"body","face","ownership","contact"}:raise ValueError("未知区域类型")
        if r["kind"]!="contact" and r.get("char_id") not in ids:raise ValueError("区域所属角色不存在")
        for k in ("x","y","w","h"):
            if not isinstance(r.get(k),(int,float)) or isinstance(r[k],bool) or not math.isfinite(r[k]):
                raise ValueError("区域坐标必须是有限数字")
        if r["w"]<=0 or r["h"]<=0 or r["x"]<0 or r["y"]<0 or r["x"]+r["w"]>1.000001 or r["y"]+r["h"]>1.000001:
            raise ValueError("区域超出画布或大小为零")
    data["width"],data["height"]=int(width),int(height)
    return data

def rect_mask(layout,char_id,kind,width=None,height=None):
    w,h=width or layout["width"],height or layout["height"]
    if kind=="contact" and "raster_contacts" in layout:
        import torch.nn.functional as F
        return F.interpolate(layout["raster_contacts"][None].float(),size=(h,w),mode="nearest")[0]
    out=torch.zeros(1,h,w,dtype=torch.float32)
    for r in layout["regions"]:
        if r["kind"]!=kind or (kind!="contact" and r["char_id"]!=char_id):continue
        x0,y0=int(r["x"]*w),int(r["y"]*h)
        x1,y1=math.ceil((r["x"]+r["w"])*w),math.ceil((r["y"]+r["h"])*h)
        out[:,y0:y1,x0:x1]=1
    return out

def role_masks(layout,char_id):
    if char_id not in [c["id"] for c in layout["characters"]]:raise ValueError(f"画布未建立 {char_id}")
    if "raster_masks" in layout:
        import torch.nn.functional as F
        src=layout["raster_masks"][char_id]
        mask=F.interpolate(src[None].float(),size=(layout["height"],layout["width"]),mode="nearest")[0]
        if not mask.any():raise ValueError(f"{char_id} 实例区域为空")
        yy,xx=torch.where(mask[0]>.5)
        return dict(mask=mask,face=torch.zeros_like(mask),ownership=torch.zeros_like(mask),
                    center=(float(xx.float().mean()/layout['width']),float(yy.float().mean()/layout['height'])))
    mask=rect_mask(layout,char_id,"body")
    face=rect_mask(layout,char_id,"face")
    own=rect_mask(layout,char_id,"ownership")
    if not torch.maximum(mask,face).any():raise ValueError(f"{char_id} 没有角色区域，请在画布拖出选框")
    rects=[r for r in layout["regions"] if r["char_id"]==char_id and r["kind"]=="body"]
    rects=rects or [r for r in layout["regions"] if r["char_id"]==char_id and r["kind"]=="face"]
    area=sum(r["w"]*r["h"] for r in rects)
    center=(sum((r["x"]+r["w"]/2)*r["w"]*r["h"] for r in rects)/area,
            sum((r["y"]+r["h"]/2)*r["w"]*r["h"] for r in rects)/area)
    return dict(mask=mask,face=face,ownership=own,center=center)

def render_layout(layout,masks=None,roles=None):
    w,h=layout["width"],layout["height"]
    im=Image.new("RGB",(w,h),(25,31,44));d=ImageDraw.Draw(im,"RGBA")
    for x in range(0,w,max(16,w//16)):d.line((x,0,x,h),fill=(255,255,255,18))
    for y in range(0,h,max(16,h//16)):d.line((0,y,w,y),fill=(255,255,255,18))
    if masks is not None:
        import torch.nn.functional as F
        m=F.interpolate(masks[:,None].float(),size=(h,w),mode="nearest")[:,0].numpy()
        arr=np.asarray(im).copy().astype(np.float32)
        for i,role in enumerate(roles):
            c=COLORS[int(role["char_id"][4:])-1]
            rgb=np.array([int(c[j:j+2],16) for j in (1,3,5)])
            a=m[i,:,:,None]*.60
            arr=arr*(1-a)+rgb*a
        im=Image.fromarray(arr.clip(0,255).astype(np.uint8))
        d=ImageDraw.Draw(im,"RGBA")
    for r in layout["regions"]:
        x,y,rw,rh=r["x"]*w,r["y"]*h,r["w"]*w,r["h"]*h
        c="#d591ff" if r["kind"]=="contact" else COLORS[int(r["char_id"][4:])-1]
        rgb=tuple(int(c[j:j+2],16) for j in (1,3,5))
        if masks is None:d.rectangle((x,y,x+rw,y+rh),fill=rgb+(35,))
        d.rectangle((x,y,x+rw,y+rh),outline=rgb+(230,),width=max(1,w//400))
        label=(r["char_id"]+" "+r["kind"]).strip()
        d.text((x+5,y+5),label,fill=rgb+(255,))
    return torch.from_numpy(np.array(im).copy()).float()[None]/255
