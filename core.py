"""Anima 接触友好的区域 LoRA 引擎（实验版）。

不修改 ComfyUI 源码或复制原生带权模块；采样前向期间使用可撤销钩子，
按 Cosmos/Anima 的 [B,T,H,W,D] 和独立 cross-attn 进行区域增量。
"""
from __future__ import annotations
import contextvars
import threading
import weakref
from contextlib import contextmanager
from collections import OrderedDict
import logging
import math
import re
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F
LOG = logging.getLogger('AnimaContactRegional')
BUNDLE_KEY = 'anima_contact_regional_lora_v06'
_ACTIVE = contextvars.ContextVar('anima_regional_bundle', default=None)
_BRANCH = contextvars.ContextVar('anima_regional_branch', default=None)
_CAPTURE = contextvars.ContextVar('anima_regional_capture', default=None)
_MODEL_LOCKS = weakref.WeakKeyDictionary()
_LOCK_GUARD = threading.Lock()

class AdapterCache:
    """采样内有界缓存，不在原生模块上保存 GPU Tensor，退出外层采样即释放。"""

    def __init__(self, limit_mb=64):
        self.limit = int(limit_mb * 1024 * 1024)
        self.bytes = 0
        self.items = OrderedDict()
        self.peak = 0

    def get(self, adapter, x):
        key = (id(adapter.a), id(adapter.b), str(x.device), str(x.dtype))
        if key in self.items:
            item = self.items.pop(key)
            self.items[key] = item
            return item[0]
        pair = (adapter.a.to(device=x.device, dtype=x.dtype), adapter.b.to(device=x.device, dtype=x.dtype))
        size = sum((t.numel() * t.element_size() for t in pair))
        if size <= self.limit:
            while self.items and self.bytes + size > self.limit:
                _, (_, old) = self.items.popitem(last=False)
                self.bytes -= old
            self.items[key] = (pair, size)
            self.bytes += size
            self.peak = max(self.peak, self.bytes)
        return pair

    def clear(self):
        self.items.clear()
        self.bytes = 0

class RuntimeState:
    """采样间连续状态：故意不是 dict/list，以抵抗 ComfyUI merge_nested_dicts 的递归复制。

    ComfyUI 只浅复制未知对象；每个 Apply 节点新建一份状态，不能跨实例污染。
    """

    def __init__(self, adaptive):
        self.settings = dict(adaptive)
        self.runtime = {}
        self.reset()

    def release_gpu(self):
        cache = self.runtime.get('adapter_cache')
        if cache is not None:
            cache.clear()
        self.runtime.get('context_cache', {}).clear()

    def reset(self):
        self.release_gpu()
        self.adaptive = dict(self.settings, calls=0, affinity=None, mask=None, last_sigma=None)
        self.runtime = {'calls': 0, 'log_first': True, 'context_cache': {}, 'last_gates': None, 'adapter_cache': AdapterCache(64)}

@dataclass
class Adapter:
    kind: str
    a: torch.Tensor
    b: torch.Tensor
    scale: float
    region: int
    text_projection: bool = False
    cross_projection: bool = False
    self_projection: bool = False

def normalized_key(key: str) -> str:
    """对 PEFT/Kohya/LyCORIS 命名安全归一，不盲目替换路径分隔符。"""
    prefixes = ('base_model.model.transformer.', 'base_model.model.', 'model.diffusion_model.', 'diffusion_model.', 'diffusion_model_', 'model.', 'transformer.', 'net.', 'net_', 'lycoris_', 'lora_unet_', 'lora_transformer_', 'unet.')
    for _ in range(8):
        old = key
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):].lstrip('_')
                break
        if key == old:
            break
    return key

def parse_lora(sd: dict[str, torch.Tensor]) -> tuple[dict[str, dict], list[str]]:
    """返回可区域化的 Linear LoRA/LoKr 和明确的 unsupported 报告。

    LoKr 支持 direct/direct、direct/factorized、factorized/factorized；
    对 DoRA、Tucker/Conv、未知扩展 fail-closed，不允许悄悄当成标准 LoKr。
    """
    raw: dict[str, dict] = {}
    supported = ('lokr_w1', 'lokr_w1_a', 'lokr_w1_b', 'lokr_w2', 'lokr_w2_a', 'lokr_w2_b', 'lokr_t1', 'lokr_t2', 'alpha', 'dora_scale', 'scalar', 'lora_A.weight', 'lora_B.weight', 'lora_A.default.weight', 'lora_B.default.weight', 'lora_down.weight', 'lora_up.weight')
    for key, value in sd.items():
        k = str(key)
        for suffix in sorted(supported, key=len, reverse=True):
            token = '.' + suffix
            if k.endswith(token):
                stem = k[:-len(token)]
                raw.setdefault(stem, {})[suffix] = value
                break
    result = {}
    unsupported = []
    for raw_key, group in raw.items():
        norm = normalized_key(raw_key)
        if 'dora_scale' in group:
            unsupported.append(f'{raw_key}: DoRA weight decomposition 暂不支持安全区域化')
            continue
        if 'lokr_t1' in group or 'lokr_t2' in group:
            unsupported.append(f'{raw_key}: Tucker LoKr 暂不支持')
            continue
        if any((x.startswith('lokr_') for x in group)):

            def fac(name):
                if name in group:
                    w = group[name]
                    return (w, None) if w.ndim == 2 else (None, None)
                a, b = (group.get(name + '_a'), group.get(name + '_b'))
                if a is None or b is None or a.ndim != 2 or (b.ndim != 2) or (a.shape[1] != b.shape[0]):
                    return (None, None)
                return (a.float() @ b.float(), int(b.shape[0]))
            w1, r1 = fac('lokr_w1')
            w2, r2 = fac('lokr_w2')
            if w1 is None or w2 is None:
                unsupported.append(f'{raw_key}: 不完整或非二维的 LoKr 因子')
                continue
            alpha = group.get('alpha')
            rank = r1 or r2
            scale = float(alpha) / float(rank) if alpha is not None and rank else 1.0
            scalar = group.get('scalar')
            if scalar is not None:
                if not torch.is_tensor(scalar) or scalar.numel() != 1:
                    unsupported.append(f'{raw_key}: 非标量 scalar')
                    continue
                scale *= float(scalar.detach().float().item())
            result[norm] = dict(kind='lokr', a=w1, b=w2, scale=scale, raw=raw_key)
            continue
        down = group.get('lora_A.weight', group.get('lora_A.default.weight', group.get('lora_down.weight')))
        up = group.get('lora_B.weight', group.get('lora_B.default.weight', group.get('lora_up.weight')))
        if down is None and up is None:
            continue
        if down is None or up is None or down.ndim != 2 or (up.ndim != 2) or (down.shape[0] != up.shape[1]):
            unsupported.append(f'{raw_key}: LoRA down/up 维度不匹配')
            continue
        rank = down.shape[0]
        alpha = float(group['alpha']) if 'alpha' in group else rank
        result[norm] = dict(kind='lora', a=down, b=up, scale=alpha / rank, raw=raw_key)
    return (result, unsupported)

def _capture_affinity(q, k, st, region):
    """Anima/Cosmos ComfyUI attention container 的非破坏性注意力观测。

    观测仅是粗空间相关性，不等于语义分割。使用非破坏性的 peek()
    不消耗 ComfyUI 的单次所有权容器；恢复二维 token 网格，不做 1D 插值。
    """
    with torch.no_grad():
        if not torch.is_tensor(q) and hasattr(q, 'peek'):
            q = q.peek()
        if not torch.is_tensor(k) and hasattr(k, 'peek'):
            k = k.peek()
        if not torch.is_tensor(q) or not torch.is_tensor(k) or q.ndim < 4 or (k.ndim != 4):
            st['capture_incompatible'] = st.get('capture_incompatible', 0) + 1
            return
        b, nh, hd = (q.shape[0], q.shape[-2], q.shape[-1])
        h, w, frames = (st['gates'].shape[-2], st['gates'].shape[-1], st['frames'])
        sq = q.reshape(b, -1, nh, hd)
        sk = k.reshape(b, -1, nh, hd)
        if sq.shape[1] != frames * h * w:
            st['capture_incompatible'] = st.get('capture_incompatible', 0) + 1
            return
        stride = max(1, math.ceil(math.sqrt(h * w / 576)))
        if frames != 1:
            sq = sq.reshape(b, frames, h, w, nh, hd).mean(1).reshape(b, h, w, nh, hd)
        else:
            sq = sq.reshape(b, h, w, nh, hd)
        qs = sq[:, ::stride, ::stride].reshape(b, -1, nh, hd).float()
        keys = sk[:, :min(256, sk.shape[1])].float()
        if keys.shape[1] == 0:
            return
        sim = torch.einsum('bqhd,bkhd->bhqk', qs, keys) * hd ** (-0.5)
        sim = sim.amax(-1).mean(1)
        sim = sim - sim.mean(dim=-1, keepdim=True)
        rows = st['batch_rows'][:b]
        valid = torch.as_tensor(rows, dtype=torch.bool, device=sim.device)
        if not valid.any():
            return
        sim = sim[valid].mean(0).reshape(1, 1, len(range(0, h, stride)), len(range(0, w, stride)))
        full = F.interpolate(sim, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        if not torch.isfinite(full).all():
            st['capture_incompatible'] = st.get('capture_incompatible', 0) + 1
            return
        stored = st['observations']
        if stored[region] is None:
            stored[region] = full.detach().cpu()
        else:
            stored[region] += full.detach().cpu()
        st['counts'][region] += 1

def refine_gates(affinities, prior, *, mode='refine', threshold=0.45, blend=0.65, grow=2):
    """K2 FreeFuse 风格早期区域亲和度修正，保留人物手工区域先验。"""
    if mode == 'off' or not affinities or any((a is None for a in affinities)):
        return prior
    r, h, w = prior.shape
    aff = torch.stack([a.float().reshape(h, w) for a in affinities]).to(prior)
    sm = F.avg_pool2d(aff[:, None], 3, 1, 1)
    sm = F.avg_pool2d(sm, 3, 1, 1)[:, 0]
    sm = (sm - sm.amin(dim=(-1, -2), keepdim=True)) / (sm.amax(dim=(-1, -2), keepdim=True) - sm.amin(dim=(-1, -2), keepdim=True)).clamp_min(1e-06)
    if mode == 'refine':
        pr = F.max_pool2d(prior[:, None], 2 * int(grow) + 1, 1, int(grow))[:, 0]
        sm = sm * (pr > 0.01)
        sm = sm / sm.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-06)
    discovered = torch.where(sm >= threshold, sm, 0.0)
    empty = discovered.sum(dim=(1, 2)) <= 0.001
    discovered[empty] = prior[empty]
    return torch.lerp(prior, discovered, float(blend)).clamp(0, 1)

def _kron_apply(x, w1, w2):
    a1, b1 = w1.shape
    a2, b2 = w2.shape
    y = torch.einsum('...ij,pj->...ip', x.reshape(*x.shape[:-1], b1, b2), w2)
    y = torch.einsum('...ip,qi->...qp', y, w1)
    return y.reshape(*x.shape[:-1], a1 * a2)

def resize_mask(mask, h, w, device=None, grow=0, feather=0):
    """输入 Comfy MASK [H,W]/[B,H,W]，输出 [Htoken,Wtoken]。"""
    if mask is None:
        return torch.zeros(h, w, dtype=torch.float32, device=device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError('MASK 应为 [H,W] 或 [B,H,W]')
    m = mask[:1].float().unsqueeze(1).to(device=device)
    m = F.interpolate(m, size=(h, w), mode='area')
    if grow > 0:
        k = int(grow) * 2 + 1
        m = F.max_pool2d(m, k, stride=1, padding=k // 2)
    if feather > 0:
        k = int(feather) * 2 + 1
        m = F.avg_pool2d(m, k, stride=1, padding=k // 2, count_include_pad=False)
    return m[0, 0].clamp(0, 1)

def build_gates(regions: list[dict], contacts: list[dict], h, w, *, device=None, exclusive=True, contact_mix=0.0, overlap_mode='nearest', raw_masks=None, preserve_mass=False):
    """重叠按数值强度+优先级分配；接触区降低 LoRA 增量但保留底模注意力。"""
    masks = []
    owns = []
    coverage = []
    for region in regions:
        i = len(masks)
        m = resize_mask(raw_masks[i], h, w, device) if raw_masks is not None else resize_mask(region['mask'], h, w, device, region.get('grow', 0), region.get('feather', 0))
        own = resize_mask(region.get('ownership'), h, w, device)
        face = resize_mask(region.get('face'), h, w, device)
        m = torch.maximum(torch.maximum(m, own), face)
        masks.append(m)
        owns.append(own + 0.5 * face)
        coverage.append(torch.maximum(resize_mask(raw_masks[i] if raw_masks is not None else region['mask'], h, w, device, region.get('grow', 0)), torch.maximum(own, face)))
    if not masks:
        raise ValueError('至少需要一个角色区域')
    gates = torch.stack(masks, dim=0)
    if exclusive and gates.shape[0] > 1:
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=gates.device), torch.linspace(0, 1, w, device=gates.device), indexing='ij')
        centers = []
        for i, region in enumerate(regions):
            if region.get('center') is not None:
                cx, cy = region['center']
            else:
                total = gates[i].sum().clamp_min(1e-08)
                cx, cy = ((gates[i] * xx).sum() / total, (gates[i] * yy).sum() / total)
            centers.append((xx - cx) ** 2 + (yy - cy) ** 2)
        distance = torch.stack(centers)
        scores = gates + 2.0 * torch.stack(owns, dim=0) - distance * 0.05
        scores = torch.where(gates > 0, scores, torch.full_like(scores, -1000000.0))
        peak = gates.max(dim=0).values
        winner = scores.argmax(dim=0)
        mask = torch.stack([(winner == i) & (peak > 0) for i in range(len(regions))])
        gates = gates * mask
        if preserve_mass:
            gates = mask.to(gates) * torch.stack(coverage).sum(0).clamp(0, 1)
    elif gates.shape[0] > 1:
        gates = gates / gates.sum(dim=0, keepdim=True).clamp(min=1.0)
    if contacts and contact_mix > 0:
        contact_masks = [resize_mask(c['mask'], h, w, device, feather=c.get('feather', 0)) for c in contacts]
        united = torch.stack(contact_masks).amax(dim=0)
        gates *= (1.0 - float(contact_mix) * united).clamp(0, 1)
    return gates

def _spatial_gate(st: dict, adapter: Adapter, x: torch.Tensor):
    """Anima 的真实 block 中可能存在 5D 图像、3D flatten 图像、3D 文本。"""
    active = _BRANCH.get()
    if active == -1:
        return None
    if adapter.self_projection and st.get('local_self_attention'):
        if active != adapter.region:
            return None
        return torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
    if adapter.cross_projection:
        if active != adapter.region:
            return None
        return torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
    if adapter.text_projection:
        return None
    if active is not None and active != adapter.region:
        return None
    gates = st.get('lora_gates', st['gates'])
    if adapter.region >= gates.shape[0]:
        return None
    m = gates[adapter.region]
    batch_rows = st['batch_rows']
    if x.ndim == 5 and x.shape[-3:-1] == m.shape:
        out = m[None, None, :, :, None].to(x)
        if len(batch_rows) == x.shape[0]:
            row = torch.tensor(batch_rows, device=x.device, dtype=x.dtype).view(-1, 1, 1, 1, 1)
            return out * row
        return out
    if x.ndim == 3 and x.shape[1] == st['frames'] * m.numel():
        out = m.reshape(1, 1, *m.shape).expand(1, st['frames'], *m.shape).reshape(1, -1, 1).to(x)
        if len(batch_rows) == x.shape[0]:
            row = torch.tensor(batch_rows, device=x.device, dtype=x.dtype).view(-1, 1, 1)
            return out * row
        return out
    return None

def linear_delta(module, x, y, adapters):
    """对原生 Linear 的输出加增量；模块本身、参数和 AIMDO 元数据保持原样。"""
    st = _ACTIVE.get()
    if st is None or not torch.is_tensor(x):
        return y
    if not adapters:
        return y
    cast_cache = st.setdefault('adapter_cache', AdapterCache(64))
    for adapter in adapters:
        gate = _spatial_gate(st, adapter, x)
        if gate is None:
            continue
        a, b = cast_cache.get(adapter, x)
        if adapter.kind == 'lora':
            if a.shape[1] != x.shape[-1] or b.shape[0] != y.shape[-1]:
                raise RuntimeError('LoRA shape mismatch in Anima projection')
            delta = F.linear(F.linear(x, a), b)
        else:
            if a.shape[1] * b.shape[1] != x.shape[-1] or a.shape[0] * b.shape[0] != y.shape[-1]:
                raise RuntimeError('LoKr shape mismatch in Anima projection')
            delta = _kron_apply(x, a, b)
        y = y + delta.to(y) * (float(adapter.scale) * gate.to(y))
        st.setdefault('delta_calls', [0] * len(st['gates']))[adapter.region] += 1
    return y

class _RegionalCrossMixin:

    def compute_attention(self, q, k, v, transformer_options=None):
        cap = _CAPTURE.get()
        st = _ACTIVE.get()
        if cap is not None and st is not None and st.get('capture'):
            index = int(getattr(self, '_anima_block_idx', -1))
            if index in st['capture_blocks']:
                _capture_affinity(q, k, st, cap)
        return super().compute_attention(q, k, v, transformer_options=transformer_options or {})

    def forward(self, x, context=None, rope_emb=None, transformer_options=None, _original_forward=None):
        st = _ACTIVE.get()
        opts = transformer_options or {}
        parent_forward = _original_forward if _original_forward is not None else super().forward
        if st is None:
            return parent_forward(x, context=context, rope_emb=rope_emb, transformer_options=opts)

        def run(ctx, branch, capture=None):
            tk = _BRANCH.set(branch)
            cap = _CAPTURE.set(capture)
            try:
                return parent_forward(x, context=ctx, rope_emb=rope_emb, transformer_options=opts)
            finally:
                _CAPTURE.reset(cap)
                _BRANCH.reset(tk)
        baseline = run(context, -1)
        identity_baseline = run(st['identity_baseline_context'], -1) if st.get('identity_baseline_context') is not None else baseline
        out = baseline
        for idx in range(len(st['gates'])):
            if st['gates'][idx].amax().item() <= 0:
                continue
            regional_context = st['region_contexts'][idx]
            capture = idx if st.get('capture') else None
            text_only = run(regional_context, -1, capture)
            text_gate = _output_gate(st['gates'][idx], text_only, st, st.get('text_batch_rows'))
            gain = st.get('identity_gains', [1.0] * len(st['gates']))[idx]
            out = out + (text_only - identity_baseline) * text_gate * st['identity_strength'] * gain
            if st['has_adapters'][idx] and st['cross_lora_strength'] > 0:
                adapted = run(regional_context, idx)
                lora_gate = _output_gate(st['lora_gates'][idx], adapted, st)
                delta = adapted - text_only
                if st.get('separate_uncond') and (not all(st['text_batch_rows'])):
                    neg_adapted = run(context, idx)
                    rows = torch.tensor(st['text_batch_rows'], device=delta.device, dtype=torch.bool).view(-1, *[1] * (delta.ndim - 1))
                    delta = torch.where(rows, delta, neg_adapted - baseline)
                out = out + delta * lora_gate * st['cross_lora_strength']
            st.setdefault('region_cross_calls', [0] * len(st['gates']))[idx] += 1
        return out

def _output_gate(mask, y, st, rows=None):
    rows = st['batch_rows'] if rows is None else rows
    if y.ndim == 5 and tuple(y.shape[-3:-1]) == tuple(mask.shape):
        g = mask[None, None, :, :, None].to(y)
        if len(rows) == y.shape[0]:
            row = torch.tensor(rows, device=y.device, dtype=y.dtype).view(-1, 1, 1, 1, 1)
            return g * row
        return g
    if y.ndim == 3 and y.shape[1] == st['frames'] * mask.numel():
        g = mask.reshape(1, 1, *mask.shape).expand(1, st['frames'], *mask.shape).reshape(1, -1, 1).to(y)
        if len(rows) == y.shape[0]:
            row = torch.tensor(rows, device=y.device, dtype=y.dtype).view(-1, 1, 1)
            return g * row
        return g
    raise RuntimeError(f'Anima cross-attn 输出形状 {tuple(y.shape)} 与 token mask {tuple(mask.shape)} 不匹配')

def regional_self_forward(original, x, *args, **kwargs):
    """完整角色self-attention差分只在输出归属区合成，K/V不直接写入其他人的输出。"""
    st = _ACTIVE.get()
    if st is None or not st.get('local_self_attention'):
        return original(x, *args, **kwargs)

    def run(branch):
        token = _BRANCH.set(branch)
        try:
            return original(x, *args, **kwargs)
        finally:
            _BRANCH.reset(token)
    base = run(-1)
    out = base
    for i, has in enumerate(st['has_self_adapters']):
        if not has or st['lora_gates'][i].amax().item() <= 0:
            continue
        value = run(i)
        out = out + (value - base) * _output_gate(st['lora_gates'][i], value, st)
    return out

def _layer_lookup(diffusion_model):
    lookup = {}
    for path, mod in diffusion_model.named_modules():
        if path.startswith('blocks.') and getattr(mod, 'weight', None) is not None and (getattr(mod.weight, 'ndim', 0) == 2):
            lookup[path] = path
            lookup[path.replace('.', '_')] = path
    return lookup

def install_regional_loras(model_patcher, regions, *, self_kv_mode='base', **unused):
    """仅生成 CPU 适配器计划，不替换任何原生模块，也不分配 VBAR/GPU Tensor。"""
    dit = model_patcher.get_model_object('diffusion_model')
    if not all((hasattr(dit, n) for n in ('blocks', 'patch_spatial', 'llm_adapter'))):
        raise TypeError('只支持原生 Anima Cosmos 模型')
    lookup = _layer_lookup(dit)
    plan = {}
    stats = []
    try:
        import comfy.lora
        root = getattr(model_patcher, 'model', None)
        aliases = comfy.lora.model_lora_keys_unet(root, {}) if root is not None else {}
        for alias, target in aliases.items():
            if isinstance(target, str):
                canonical = normalized_key(target.removesuffix('.weight'))
                if canonical in lookup:
                    lookup[normalized_key(alias)] = lookup[canonical]
    except ImportError:
        pass
    for idx, role in enumerate(regions):
        for entry in role['loras']:
            if entry.get('missing'):
                stats.append(dict(region=idx + 1, char_id=role.get('char_id', str(idx + 1)), lora=entry['label'], matched=0, unmatched=[], skipped_global=0, policy_skipped=0, skipped_unsupported=0, status='skipped_with_warning', sample_keys=['文件不存在，未加载']))
                LOG.warning('[AnimaContact] %s 文件不存在：%s；显式回退文字身份', role.get('char_id'), entry['label'])
                continue
            sd = entry.get('sd')
            if sd is None:
                import comfy.utils
                sd = comfy.utils.load_torch_file(entry['path'], safe_load=True)
            parsed, issues = parse_lora(sd)
            counts = dict(matched=0, unmatched=[], skip_text=0, skipped_global=0, policy_skipped=0, skipped_unsupported=len(issues), unsupported=issues)
            for key, info in parsed.items():
                path = lookup.get(key) or lookup.get(key.replace('.', '_'))
                if path is None:
                    if key.startswith(('blocks.', 'blocks_')):
                        counts['unmatched'].append(key)
                    else:
                        counts['skipped_global'] += 1
                    continue
                if '.adaln_modulation_' in path:
                    counts['skipped_global'] += 1
                    continue
                if self_kv_mode == 'base' and path.endswith(('.self_attn.k_proj', '.self_attn.v_proj')):
                    counts['policy_skipped'] += 1
                    continue
                mod = dit.get_submodule(path)
                if hasattr(mod, '_anima_regional_adapters'):
                    raise RuntimeError('Stale native module patches detected. Restart ComfyUI before sampling.')
                outdim, indim = mod.weight.shape
                a, b = (info['a'], info['b'])
                ok = a.shape[1] == indim and b.shape[0] == outdim if info['kind'] == 'lora' else a.shape[1] * b.shape[1] == indim and a.shape[0] * b.shape[0] == outdim
                if not ok:
                    counts['unmatched'].append(key + ' [shape mismatch]')
                    continue
                family = 'self_kv' if path.endswith(('.self_attn.k_proj', '.self_attn.v_proj')) else 'self_q_out' if '.self_attn.' in path else 'cross' if '.cross_attn.' in path else 'mlp'
                gain = role.get('projection_gains', {}).get(family, 1.0)
                adapter = Adapter(info['kind'], a.detach().cpu(), b.detach().cpu(), info['scale'] * entry['strength'] * gain, idx, path.endswith(('.cross_attn.k_proj', '.cross_attn.v_proj')), '.cross_attn.' in path, '.self_attn.' in path)
                plan.setdefault(path, []).append(adapter)
                counts['matched'] += 1
            counts['sample_keys'] = list(sd)[:8]
            if counts['matched'] == 0:
                message = f"{role.get('char_id', idx + 1)} / {entry['label']} 没有可用区域投影：{counts}。这属于权重命名/架构匹配问题，不是角色连线或遮罩。请核对Anima版本和权重键名。"
                if entry.get('on_unmatched') == 'skip_with_report':
                    counts['status'] = 'skipped_with_warning'
                    LOG.warning('[AnimaContact] %s；显式按设置跳过此LoRA，保留该身份文字', message)
                else:
                    raise RuntimeError(message)
            else:
                counts['status'] = 'loaded'
            stats.append(dict(region=idx + 1, char_id=role.get('char_id', f'char{idx + 1}'), lora=entry['label'], **counts))
    return (stats, plan)

@contextmanager
def scoped_modules(dit, plan):
    """模型加载后才挂接，finally 移除。相同共享模型串行进入，不更改 module identity。"""
    with _LOCK_GUARD:
        lock = _MODEL_LOCKS.setdefault(dit, threading.RLock())
    with lock:
        handles = []
        restores = []
        missing = object()

        def replace(obj, name, value):
            restores.append((obj, name, obj.__dict__.get(name, missing)))
            setattr(obj, name, value)
        try:
            for path, adapters in plan.items():
                mod = dit.get_submodule(path)

                def hook(module, args, kwargs, output, adapters=adapters):
                    x = args[0] if args else kwargs.get('input', kwargs.get('x'))
                    return linear_delta(module, x, output, adapters)
                handles.append(mod.register_forward_hook(hook, with_kwargs=True))
            for idx, block in enumerate(dit.blocks):
                self_attn = block.self_attn
                self_original = self_attn.forward
                state = _ACTIVE.get()
                isolation = float((state or {}).get('self_region_isolation', 0))
                if isolation > 0 and idx >= int(len(dit.blocks) * state.get('isolation_start_layer', 0.5)) and (len(state['gates']) > 1):
                    if not hasattr(self_attn, 'attn_op'):
                        raise RuntimeError('Spatial isolation requires the native Anima attn_op interface')
                    original_op = self_attn.attn_op

                    def isolated_op(q, k, v, *args, original_op=original_op, **kwargs):
                        active = _ACTIVE.get()
                        if active is None:
                            return original_op(q, k, v, *args, **kwargs)
                        from .attention_isolation import isolated_attention
                        return isolated_attention(q, k, v, active['gates'], active['self_region_isolation'], active['frames'])
                    replace(self_attn, 'attn_op', isolated_op)

                def self_routed(x, *a, original=self_original, **kw):
                    return regional_self_forward(original, x, *a, **kw)
                replace(self_attn, 'forward', self_routed)
                cross = block.cross_attn
                original = cross.forward

                def routed(x, context=None, rope_emb=None, transformer_options=None, cross=cross, original=original):
                    return _RegionalCrossMixin.forward(cross, x, context, rope_emb, transformer_options, _original_forward=original)
                replace(cross, 'forward', routed)
                if hasattr(cross, 'compute_attention'):
                    compute = cross.compute_attention

                    def observed(q, k, v, transformer_options=None, compute=compute, idx=idx):
                        st = _ACTIVE.get()
                        region = _CAPTURE.get()
                        if st is not None and region is not None and st.get('capture') and (idx in st['capture_blocks']):
                            _capture_affinity(q, k, st, region)
                        return compute(q, k, v, transformer_options=transformer_options or {})
                    replace(cross, 'compute_attention', observed)
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()
            for obj, name, old in reversed(restores):
                if old is missing:
                    obj.__dict__.pop(name, None)
                else:
                    setattr(obj, name, old)

def _get_context_tensor(cond, model, ref):
    if not isinstance(cond, list) or not cond:
        raise ValueError('角色 CONDITIONING 为空')
    if len(cond) != 1:
        raise ValueError('角色条件暂不支持时间表/多段 CONDITIONING；请在角色节点中使用普通完整提示词')
    src, meta = cond[0]
    if not torch.is_tensor(src):
        raise TypeError('角色 CONDITIONING[0][0] 必须是 Tensor')
    if src.ndim == 4 and src.shape[1] == 1:
        src = src[:, 0]
    if src.ndim != 3:
        raise ValueError(f'不支持的 Anima CONDITIONING 形状: {tuple(src.shape)}')
    src = src.to(device=ref.device, dtype=ref.dtype)
    ids = meta.get('t5xxl_ids') if isinstance(meta, dict) else None
    if ids is not None:
        ids = torch.as_tensor(ids, device=src.device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids[None]
        weights = meta.get('t5xxl_weights')
        if weights is not None:
            weights = weights.to(src)
            if weights.ndim == 1:
                weights = weights[None, :, None]
            elif weights.ndim == 2:
                weights = weights[:, :, None]
        src = model.preprocess_text_embeds(src, ids, t5xxl_weights=weights)
    if src.shape[-1] != ref.shape[-1]:
        raise RuntimeError('角色条件维度不匹配：请检查 Anima 文本编码器以及 T5 token 元数据')
    if src.shape[0] == 1 and ref.shape[0] > 1:
        src = src.expand(ref.shape[0], -1, -1)
    if src.shape[0] != ref.shape[0]:
        raise ValueError('角色 CONDITIONING batch 与采样 batch 不匹配')
    return src

def regional_wrapper(executor, *args, **kwargs):
    """所有区域都按本次真实 latent 的 patch 网格重建，不绑定第一次采样尺寸。"""
    to = kwargs.get('transformer_options', {})
    bundle = to.get(BUNDLE_KEY) if isinstance(to, dict) else None
    if not bundle:
        return executor(*args, **kwargs)
    x = args[0] if args else kwargs.get('x')
    ctx = args[2] if len(args) > 2 else kwargs.get('context')
    if x is None or ctx is None:
        raise RuntimeError('Anima 执行器参数缺失')
    model = executor.class_obj
    if not bundle['regions']:
        bundle['runtime_handle'].runtime['last_gates'] = torch.empty((0, math.ceil(x.shape[-2] / int(model.patch_spatial)), math.ceil(x.shape[-1] / int(model.patch_spatial))))
        if bundle.get('shape_control') is not None:
            from .shape_control import scoped_lllite
            with scoped_lllite(model, bundle['shape_control'], args, kwargs):
                return executor(*args, **kwargs)
        return executor(*args, **kwargs)
    h = math.ceil(x.shape[-2] / int(model.patch_spatial))
    w = math.ceil(x.shape[-1] / int(model.patch_spatial))
    frames = math.ceil(x.shape[-3] / int(model.patch_temporal)) if x.ndim == 5 else 1
    bs = x.shape[0]
    co = to.get('cond_or_uncond', [0])
    if bs % max(1, len(co)):
        raise RuntimeError('无法确定 CFG batch 行，拒绝将角色 LoRA 意外写入错误分支')
    rows = [int(c) == 0 or bundle['apply_uncond'] for c in co for _ in range(bs // len(co))]
    text_rows = [int(c) == 0 for c in co for _ in range(bs // len(co))] if bundle.get('separate_uncond') else rows
    handle = bundle['runtime_handle']
    adaptive = handle.adaptive
    sigma = to.get('sigmas')
    sigma = float(sigma.max().detach().cpu()) if torch.is_tensor(sigma) and sigma.numel() else None
    grid = (frames, h, w)
    new_round = adaptive['last_sigma'] is not None and sigma is not None and (sigma > adaptive['last_sigma'] * 1.01)
    if new_round or handle.runtime.get('grid') not in (None, grid):
        handle.reset()
        adaptive = handle.adaptive
        LOG.info('[AnimaContact][RESET] 新采样或网格变化，已清空动态观测和文字缓存')
    adaptive['last_sigma'] = sigma
    runtime = handle.runtime
    runtime['adapter_cache'].limit = int(bundle.get('adapter_cache_mb', 64) * 1024 * 1024)
    runtime['grid'] = grid
    exclusive = bundle['overlap_mode'] == 'nearest'
    raw = adaptive.get('mask')
    if raw is None:
        raw = bundle.get('inherited_masks')
    gates = build_gates(bundle['regions'], [], h, w, device=x.device, exclusive=exclusive, raw_masks=raw, preserve_mass=bundle.get('preserve_gate_mass', False))
    lora_gates = gates.clone()
    if bundle['contacts'] and bundle['contact_mix'] > 0:
        contact = torch.stack([resize_mask(c['mask'], h, w, x.device, feather=c.get('feather', 0)) for c in bundle['contacts']]).amax(0)
        lora_gates *= 1.0 - bundle['contact_mix'] * contact
    contexts = []
    cache = runtime['context_cache']
    for i, ch in enumerate(bundle['regions']):
        key = (i, str(ctx.device), str(ctx.dtype))
        if key not in cache:
            cache[key] = _get_context_tensor(ch['conditioning'], model, ctx[:1]).detach()
        contexts.append(cache[key].expand(bs, -1, -1))
    nblocks = len(model.blocks)
    capture_set = {min(nblocks - 1, nblocks // 4 + i * max(1, nblocks // 8)) for i in range(4)}
    count = len(gates)
    has_adapters = [any((abs(e['strength']) > 0 for e in ch['loras'])) for ch in bundle['regions']]
    observe = [None] * count
    st = {'gates': gates, 'lora_gates': lora_gates, 'frames': frames, 'batch_rows': rows, 'region_contexts': contexts, 'identity_strength': bundle['identity_strength'], 'cross_lora_strength': bundle['cross_lora_strength'], 'has_adapters': has_adapters, 'capture': adaptive['mode'] != 'off' and adaptive['calls'] < adaptive['steps'] and any(rows), 'capture_blocks': capture_set, 'observations': observe, 'counts': [0] * count, 'delta_calls': [0] * count, 'region_cross_calls': [0] * count, 'capture_incompatible': 0, 'adapter_cache': runtime['adapter_cache']}
    st.update(self_region_isolation=bundle.get('self_region_isolation', 0.0), isolation_start_layer=bundle.get('isolation_start_layer', 0.5))
    st.update(text_batch_rows=text_rows, separate_uncond=bundle.get('separate_uncond', False), local_self_attention=bundle.get('local_self_attention', False), identity_gains=[ch.get('identity_gain', 1.0) for ch in bundle['regions']], has_self_adapters=[any((a.region == i and a.self_projection and (abs(a.scale) > 0) for adapters in bundle['patch_plan'].values() for a in adapters)) for i in range(count)])
    ref = bundle['regions'][0].get('identity_reference_conditioning')
    if ref is not None:
        key = ('identity_reference', str(ctx.device), str(ctx.dtype))
        if key not in cache:
            cache[key] = _get_context_tensor(ref, model, ctx[:1]).detach()
        st['identity_baseline_context'] = cache[key].expand(bs, -1, -1)
    runtime['calls'] += 1
    if runtime['log_first']:
        LOG.info('[AnimaContact][%s] latent=%s, token=%sx%s, CFG=%s, 身份文字=%.2f, 交叉LoRA=%.2f, self-KV=%s, 覆盖=%s', bundle['stage'], tuple(x.shape), h, w, rows, bundle['identity_strength'], bundle['cross_lora_strength'], bundle['self_kv_mode'], [round(float(m.mean()), 4) for m in gates])
    tk = _ACTIVE.set(st)
    branch = _BRANCH.set(None)
    try:
        with scoped_modules(model, bundle['patch_plan']):
            if bundle.get('shape_control') is not None:
                from .shape_control import scoped_lllite
                with scoped_lllite(model, bundle['shape_control'], args, kwargs):
                    result = executor(*args, **kwargs)
            else:
                result = executor(*args, **kwargs)
        if any(rows):
            inactive = [i + 1 for i, yes in enumerate(has_adapters) if yes and st['delta_calls'][i] == 0]
            if inactive:
                raise RuntimeError(f'角色 {inactive} 有 LoRA，但本次没有执行任何区域增量')
        if runtime['log_first']:
            LOG.info('[AnimaContact][RUNTIME] 各角色增量=%s, 条件重算=%s, 观测=%s, 不兼容=%s', st['delta_calls'], st['region_cross_calls'], st['counts'], st['capture_incompatible'])
            runtime['log_first'] = False
        runtime['delta_calls'] = st['delta_calls']
        runtime['cross_calls'] = st['region_cross_calls']
        runtime['last_gates'] = gates.detach().cpu()
        return result
    finally:
        _BRANCH.reset(branch)
        _ACTIVE.reset(tk)
        if st['capture']:
            if adaptive['affinity'] is None:
                adaptive['affinity'] = [None] * count
            for i, a in enumerate(observe):
                if a is not None:
                    adaptive['affinity'][i] = a if adaptive['affinity'][i] is None else adaptive['affinity'][i] + a
            adaptive['calls'] += 1
            if adaptive['calls'] >= adaptive['steps'] and all((a is not None for a in adaptive['affinity'])):
                prior = build_gates(bundle['regions'], [], h, w, device='cpu', exclusive=False, raw_masks=bundle.get('inherited_masks'))
                discovered = refine_gates(adaptive['affinity'], prior, mode='refine', threshold=adaptive['threshold'], blend=adaptive['blend'], grow=adaptive['grow'])
                adaptive['mask'] = torch.where(discovered >= 0.5, torch.ones_like(discovered), torch.zeros_like(discovered)).cpu()
                LOG.info('[AnimaContact][ADAPTIVE] 已更新 %s；手工面部和肢体锁定仍优先', tuple(discovered.shape))
