"""姿态拓扑＋深度边缘＋可选 SAM2 的人物归属，不把 CN 图误称为实例标签。"""
from __future__ import annotations
import copy, hashlib, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .layout import COLORS
SAM2_CONFIGS = {'sam2.1_hiera_tiny.pt': 'configs/sam2.1/sam2.1_hiera_t.yaml', 'sam2.1_hiera_small.pt': 'configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_base_plus.pt': 'configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_large.pt': 'configs/sam2.1/sam2.1_hiera_l.yaml'}

def sam2_config(checkpoint):
    name = str(checkpoint).replace('\\', '/').split('/')[-1]
    if name not in SAM2_CONFIGS:
        raise ValueError('不支持的SAM2权重文件名；请选择官方SAM2.1 tiny/small/base_plus/large，不能只重命名其他架构权重')
    return SAM2_CONFIGS[name]

def to_rgb(image, width, height):
    if torch.is_tensor(image):
        a = image[0].detach().cpu().numpy() if image.ndim == 4 else image.detach().cpu().numpy()
    else:
        a = np.asarray(image)
    if a.dtype != np.uint8:
        a = (a.clip(0, 1) * 255).astype(np.uint8)
    return np.array(Image.fromarray(a[:, :, :3]).resize((width, height), Image.Resampling.BILINEAR))

def reference_key(image, width, height, count):
    a = to_rgb(image, width, height)
    return hashlib.sha256(a.tobytes() + f'{width},{height},{count}'.encode()).hexdigest()[:24]

def seed_labels(image, width, height, count, raw):
    data = json.loads(raw) if isinstance(raw, str) else raw
    key = reference_key(image, width, height, count)
    if data.get('strokes') and data.get('source_id') != key:
        raise ValueError('参考图或宽高/人数已变化，请清空旧种子笔画，运行载入新参考后再标注')
    blank = {f'char{i}': torch.zeros(1, height, width) for i in range(1, count + 1)}
    layout = dict(width=width, height=height, raster_masks=blank, layout_id=key)
    return (raster_strokes(layout, data, allow_background=True), key)

def sample_points(mask, limit=32):
    y, x = np.nonzero(mask)
    if not len(x):
        return np.empty((0, 2), np.float32)
    xy = np.c_[x, y].astype(np.float32)
    if len(xy) > 8192:
        xy = xy[::int(np.ceil(len(xy) / 8192))]
    selected = [int(np.argmin(((xy - xy.mean(0)) ** 2).sum(1)))]
    distance = np.full(len(xy), np.inf)
    for _ in range(min(limit, len(xy)) - 1):
        distance = np.minimum(distance, ((xy - xy[selected[-1]]) ** 2).sum(1))
        selected.append(int(distance.argmax()))
    return xy[selected]

def sam2_masks(rgb, points, checkpoint, background_points=None):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise RuntimeError('Install the official SAM2 package in the ComfyUI Python environment.') from e
    model = build_sam2(sam2_config(checkpoint), checkpoint, device='cpu', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)
    masks = []
    try:
        with torch.inference_mode():
            predictor.set_image(rgb)
            for i, pos in enumerate(points):
                if not len(pos):
                    raise ValueError(f'人物 {i + 1} 缺少正种子，无法提示 SAM2')
                pos = pos[::max(1, len(pos) // 32)]
                neg = np.concatenate([p[::max(1, len(p) // 24)] for j, p in enumerate(points) if j != i]) if len(points) > 1 else np.empty((0, 2))
                if background_points is not None and len(background_points):
                    neg = np.concatenate([neg, background_points])
                xy = np.concatenate([pos, neg])
                labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int32)
                pred, quality, _ = predictor.predict(point_coords=xy, point_labels=labels, multimask_output=True)
                xs = np.clip(xy[:, 0].astype(int), 0, rgb.shape[1] - 1)
                ys = np.clip(xy[:, 1].astype(int), 0, rgb.shape[0] - 1)
                coverage = pred[:, ys, xs]
                scores = quality + 3 * coverage[:, :len(pos)].mean(1)
                if len(neg):
                    scores -= 3 * coverage[:, len(pos):].mean(1)
                masks.append(pred[int(np.argmax(scores))].astype(bool))
    finally:
        if hasattr(predictor, 'reset_predictor'):
            predictor.reset_predictor()
        del predictor, model
    return np.stack(masks)

def contact_ring(labels, radius):
    if labels.max() < 2:
        return np.zeros_like(labels, dtype=bool)
    near = np.zeros_like(labels, np.uint8)
    for i in range(1, int(labels.max()) + 1):
        near += ndimage.maximum_filter((labels == i).astype(np.uint8), size=2 * radius + 1) > 0
    return (near >= 2) & (labels > 0)

def partition_foreground(rgb, depth, people, backend='pose-depth', sam_checkpoint=None, foreground=None, threshold=0.3, depth_threshold=0.1, edge_weight=12.0, contact_px=5, manual_labels=None, manual_count=None, sam_prior=None):
    h, w = depth.shape
    background = None
    if manual_labels is None:
        raise ValueError('Ownership allocation requires explicit identity seeds')
    else:
        seeds = np.stack([manual_labels == i for i in range(1, manual_count + 1)])
        missing = [f'char{i + 1}' for i, s in enumerate(seeds) if not s.any()]
        if missing:
            raise ValueError('缺少人物种子：' + ', '.join(missing) + '；请先在原图种子画笔中分别标注每个人，再重新运行。笔画太细时请加粗。')
        background = manual_labels == 255
        supports = seeds.copy()
        points = [sample_points(s) for s in seeds]
    sam = sam_prior if sam_prior is not None else sam2_masks(rgb, points, sam_checkpoint, background_points=sample_points(background) if background is not None else None) if backend == 'sam2' else None
    if foreground is not None:
        fg = foreground > 0.5
    elif sam is not None:
        fg = sam.any(0)
    else:
        fg = depth > depth_threshold
    fg |= supports.any(0) & (depth > depth_threshold)
    if manual_labels is not None:
        fg |= seeds.any(0)
        fg[background] = False
    seeds &= fg[None]
    overlap = seeds.sum(0) > 1
    seeds[:, overlap] = False
    if any((not m.any() for m in seeds)):
        raise ValueError('身份种子未落入可用前景，请检查前景遮罩及种子位置')
    index = np.arange(h * w).reshape(h, w)
    edges = []
    costs = []
    gray = rgb.astype(np.float32).mean(-1) / 255
    for dy, dx in [(0, 1), (1, 0)]:
        a = (slice(0, h - dy or h), slice(0, w - dx or w))
        b = (slice(dy, h), slice(dx, w))
        good = fg[a] & fg[b]
        u = index[a][good]
        v = index[b][good]
        weight = 1 + edge_weight * np.abs(depth[a][good] - depth[b][good]) + 2 * np.abs(gray[a][good] - gray[b][good])
        edges.append((u, v))
        costs.append(weight)
    distances = []
    for i, seed in enumerate(seeds):
        us = []
        vs = []
        cs = []
        for (u, v), base in zip(edges, costs):
            if sam is not None:
                allow = sam[i].reshape(-1)
                penalty = 1 + 5 * (~allow[u] | ~allow[v])
                cost = base * penalty
            else:
                cost = base
            us.extend((u, v))
            vs.extend((v, u))
            cs.extend((cost, cost))
        graph = coo_matrix((np.concatenate(cs), (np.concatenate(us), np.concatenate(vs))), shape=(h * w, h * w)).tocsr()
        dist = dijkstra(graph, directed=False, indices=np.flatnonzero(seed), min_only=True)
        distances.append(dist.reshape(h, w))
    dist = np.stack(distances)
    best = np.min(dist, 0)
    labels = (np.argmin(dist, 0) + 1).astype(np.uint8)
    labels[~fg | ~np.isfinite(best)] = 0
    detached = fg & ~np.isfinite(best)
    if detached.any():
        proximity = np.stack([ndimage.distance_transform_edt(~s) for s in seeds])
        if sam is not None:
            proximity = proximity + np.where(sam, 0, 1000)
        labels[detached] = (np.argmin(proximity, axis=0)[detached] + 1).astype(np.uint8)
    for i, s in enumerate(seeds):
        labels[s] = i + 1
    uncertain = overlap | detached
    if len(dist) > 1:
        d = np.sort(dist, axis=0)[:2]
        finite = np.isfinite(d).all(0)
        margin = np.zeros((h, w))
        margin[finite] = (d[1, finite] - d[0, finite]) / (d[1, finite] + 1)
        uncertain |= finite & (margin < 0.12) & (labels > 0)
    return (labels, contact_ring(labels, contact_px), uncertain)

def label_layout(labels, width, height, contact_pixels=10, reference=None, mapping=None):
    ids = [int(i) for i in np.unique(labels) if i > 0]
    if len(ids) > 16 or ids != list(range(1, len(ids) + 1)):
        raise ValueError('人物 ID 必须连续，且每个人都要有可见区域')
    full = np.array(Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST))
    masks = {f'char{i}': torch.from_numpy((full == i).astype(np.float32))[None] for i in ids}
    contact = contact_ring(full, int(contact_pixels))
    key = hashlib.sha256(full.tobytes()).hexdigest()[:20]
    return dict(version=2, width=width, height=height, characters=[dict(id=f'char{i}', label=f'char{i}') for i in ids], regions=[], raster_masks=masks, raster_contacts=torch.from_numpy(contact.astype(np.float32))[None], layout_id=key, person_mapping=mapping or [], reference_image=reference, contact_pixels=int(contact_pixels))

def label_image(layout):
    h, w = (layout['height'], layout['width'])
    a = np.zeros((h, w, 3), np.uint8)
    for cid, mask in layout['raster_masks'].items():
        c = COLORS[int(cid[4:]) - 1]
        a[mask[0].numpy() > 0.5] = [int(c[j:j + 2], 16) for j in (1, 3, 5)]
    return torch.from_numpy(a).float()[None] / 255

def overlay_image(layout, image, uncertain=None):
    rgb = to_rgb(image, layout['width'], layout['height']).astype(np.float32) / 255
    labels = label_image(layout)[0].numpy()
    active = np.any(labels > 0, axis=-1)
    out = rgb.copy()
    out[active] = rgb[active] * 0.55 + labels[active] * 0.45
    ring = layout['raster_contacts'][0].numpy() > 0.5
    out[ring] = out[ring] * 0.35 + np.array([0.75, 0.3, 1.0]) * 0.65
    if uncertain is not None:
        u = np.array(Image.fromarray(uncertain).resize((layout['width'], layout['height']), Image.Resampling.NEAREST))
        out[u] = out[u] * 0.45 + np.array([1.0, 0.18, 0.2]) * 0.55
    return torch.from_numpy(out.copy()).float()[None]

def raster_strokes(layout, data, allow_background=False):
    h, w = (layout['height'], layout['width'])
    base = np.zeros((h, w), np.uint8)
    for cid, m in layout['raster_masks'].items():
        base[m[0].numpy() > 0.5] = int(cid[4:])
    im = Image.fromarray(base)
    draw = ImageDraw.Draw(im)
    strokes = data.get('strokes', [])
    if len(strokes) > 5000:
        raise ValueError('笔画超过5000条，请清理不需要的笔画后重新运行')
    for stroke in strokes:
        cid = stroke.get('char_id', 'erase')
        if cid == 'background' and allow_background:
            value = 255
        elif cid == 'erase':
            value = 0
        elif cid in layout['raster_masks']:
            value = int(cid[4:])
        else:
            raise ValueError('笔画的 char ID 不存在')
        radius = float(stroke.get('radius', 0.01))
        if not 0 < radius <= 0.2:
            raise ValueError('画笔半径超出范围')
        points = stroke.get('points', [])
        if not points or len(points) > 20000:
            raise ValueError('笔画点数不合法')
        xy = []
        for x, y in points:
            if not np.isfinite([x, y]).all() or not 0 <= x <= 1 or (not 0 <= y <= 1):
                raise ValueError('笔画坐标越界')
            xy.append((x * w, y * h))
        r = max(1, int(radius * min(w, h)))
        draw.line(xy, fill=value, width=2 * r)
        for x, y in (xy[0], xy[-1]):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=value)
    return np.array(im)

def paint_layout(layout, raw):
    data = json.loads(raw) if isinstance(raw, str) else raw
    if data.get('source_id') and data['source_id'] != layout.get('layout_id') and data.get('strokes'):
        raise ValueError('参考区域已变化，请在涂改节点清空旧笔画后重新运行，避免把旧手臂位置应用到新动作')
    h, w = (layout['height'], layout['width'])
    out = label_layout(raster_strokes(layout, data), w, h, layout.get('contact_pixels', 10), layout.get('reference_image'), layout.get('person_mapping'))
    if len(out['characters']) != len(layout['characters']):
        raise ValueError('涂改移除了整个人物，请撤销或补回该角色区域')
    return out
