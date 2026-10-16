"""Internal operations shared by the public functional nodes."""
from __future__ import annotations
import copy, hashlib, json, logging, base64, io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .layout import DEFAULT_JSON, parse_layout, role_masks, rect_mask, render_layout, COLORS
from .core import BUNDLE_KEY, RuntimeState, install_regional_loras, regional_wrapper
from .auto_regions import paint_layout, label_layout, label_image, overlay_image, raster_strokes, to_rgb, seed_labels, reference_key
from .mask_cleanup import clean_layout, layout_labels
from .anime_detection import detect_anime, box_preview, regions_from_boxes
LOG = logging.getLogger('AnimaContactRegional')
LAYOUT = 'ANIMA_LAYOUT'
ROLES = 'ANIMA_IDENTITIES'
PIPE = 'ANIMA_PIPELINE'
SEEDS = 'ANIMA_SEEDS'
BOXES = 'ANIMA_PEOPLE'
LLLITE = 'ANIMA_LLLITE'
POLICY = 'ANIMA_SCENE_POLICY'

def encode(clip, text):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)

def tip(default, text, **kw):
    return dict(default=default, tooltip=text, **kw)

class _RegionCanvas:
    RETURN_TYPES = (LAYOUT, 'IMAGE', 'INT', 'INT')
    FUNCTION = 'make'

    def make(self, canvas_width, canvas_height, layout_json):
        layout = parse_layout(layout_json, canvas_width, canvas_height)
        return (layout, render_layout(layout), canvas_width, canvas_height)

class _IdentityCore:
    RETURN_TYPES = (ROLES,)
    FUNCTION = 'make'

    def make(self, clip, layout, scene_text, char_id, lora_name, lora_strength, identity_prompt, edge_softness, previous_roles=None):
        roles = list(previous_roles or [])
        if char_id in [r['char_id'] for r in roles]:
            raise ValueError(f'重复绑定 {char_id}')
        data = role_masks(layout, char_id)
        loras = []
        if lora_name != 'None':
            import folder_paths
            loras = [dict(path=folder_paths.get_full_path_or_raise('loras', lora_name), strength=float(lora_strength), label=lora_name)]
        if 'solo' in identity_prompt.lower().replace(',', ' ').split():
            LOG.warning('%s 提示词包含 solo，与多人构图可能冲突', char_id)
        combined = ', '.join((x.strip().strip(',') for x in (scene_text, identity_prompt) if x.strip()))
        data.update(char_id=char_id, name=char_id, loras=loras, grow=0, feather=int(edge_softness), conditioning=encode(clip, combined), combined_prompt=combined, layout_source=layout)
        return (roles + [data],)

def patch_pipeline(source, roles, layout, settings, stage='首采', inherited_masks=None):
    from comfy.patcher_extension import WrappersMP
    if BUNDLE_KEY in source.model_options.get('transformer_options', {}):
        raise ValueError('输入已经应用区域 LoRA。请接原始底模；二采请使用专门的二采中转节点，避免重复叠加')
    m = source.clone()
    stats, plan = install_regional_loras(m, roles, self_kv_mode=settings['self_kv_mode'])
    handle = RuntimeState(dict(mode=settings['adaptive_mode'], steps=settings['adaptive_calls'], threshold=0.55, blend=settings['adaptive_blend'], grow=2))
    contacts = [dict(mask=rect_mask(layout, None, 'contact'), feather=1)]
    bundle = dict(settings, regions=roles, contacts=contacts, patch_plan=plan, runtime_handle=handle, inherited_masks=inherited_masks, stage=stage)
    m.model_options.setdefault('transformer_options', {})[BUNDLE_KEY] = bundle
    m.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, BUNDLE_KEY, regional_wrapper)
    if hasattr(WrappersMP, 'OUTER_SAMPLE'):

        def reset_sample(executor, *args, **kwargs):
            handle.reset()
            if bundle.get('shape_control') is not None:
                bundle['shape_control'].runtime.clear()
            try:
                return executor(*args, **kwargs)
            except MemoryError as e:
                if 'VBAR OOM' in str(e):
                    LOG.error('[AnimaContact][VBAR] 原生加载器的VBAR槽位耗尽。本版不替换加权模块；请重启运行环境以释放旧槽位。若重启后仍出现，请提供ComfyUI版本及动态VRAM日志；不自动修改核心或无限重试。')
                raise
            finally:
                handle.release_gpu()
        m.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, BUNDLE_KEY, reset_sample)
    pipe = dict(source_model=source, roles=roles, layout=layout, settings=dict(settings), bundle=bundle, stage=stage, lora_report=stats)
    lines = [f"{stage} / {layout['width']}×{layout['height']}", f"身份文字={settings['identity_strength']}；交叉LoRA={settings['cross_lora_strength']}；self-KV={settings['self_kv_mode']}", '文字与LoRA差分分离；原生加权模块身份保持；采样内临时钩子，结束即移除；GPU增量缓存有界。']
    for stat in stats:
        lines.append(f"{stat['char_id']} {stat['lora']}: 可用={stat['matched']}, 策略跳过={stat['policy_skipped']}, 全局跳过={stat['skipped_global']}, 未匹配={len(stat['unmatched'])}, 不支持={stat['skipped_unsupported']}")
        if stat.get('status') == 'skipped_with_warning':
            lines.append('警告：此LoRA未加载，已按显式设置保留文字身份；原始键例=' + str(stat.get('sample_keys')))
    report = '\n'.join(lines)
    LOG.info('[AnimaContact] %s', report)
    return (m, pipe, report)

def upscale_latent(samples, target_h, target_w, method):
    x = samples['samples']
    if x.ndim not in (4, 5):
        raise ValueError('仅支持 4D/5D LATENT')
    frames = x.shape[2] if x.ndim == 5 else None
    src = x.movedim(2, 1).flatten(0, 1) if frames is not None else x
    options = {} if method == 'nearest-exact' else {'align_corners': False}
    dst = F.interpolate(src, size=(target_h, target_w), mode=method, **options)
    if frames is not None:
        dst = dst.reshape(x.shape[0], frames, x.shape[1], target_h, target_w).movedim(1, 2)
    out = dict(samples, samples=dst)
    if 'noise_mask' in samples:
        mask = samples['noise_mask']
        shape = mask.shape
        if mask.ndim < 2:
            raise ValueError('noise_mask 缺少空间维度')
        out['noise_mask'] = F.interpolate(mask.float().reshape(-1, 1, *shape[-2:]), size=(target_h, target_w), mode='bilinear', align_corners=False).reshape(*shape[:-2], target_h, target_w).to(mask)
    return out

class _LatentUpscale:
    RETURN_TYPES = ('MODEL', 'LATENT', PIPE, 'IMAGE', 'INT', 'INT', 'STRING')
    FUNCTION = 'prepare'

    def prepare(self, pipeline, samples, scale_factor, upscale_method, mask_source, second_adaptive):
        x = samples['samples']
        if x.ndim not in (4, 5):
            raise ValueError('首采 LATENT 维度不支持')
        h, w = x.shape[-2:]
        expected = pipeline['layout']
        if (w * 8, h * 8) != (expected['width'], expected['height']):
            raise ValueError('LATENT 尺寸与区域管线不一致，请将同一采样器的 LATENT 和其对应管线接入中转')
        th, tw = (max(2, int(h * scale_factor / 2 + 0.5) * 2), max(2, int(w * scale_factor / 2 + 0.5) * 2))
        if th * 8 > 8192 or tw * 8 > 8192:
            raise ValueError('二采目标超出 8192 像素')
        enlarged = upscale_latent(samples, th, tw, upscale_method)
        layout = copy.deepcopy(expected)
        layout.update(width=tw * 8, height=th * 8)
        roles = []
        for old in pipeline['roles']:
            ch = dict(old)
            ch.update(role_masks(layout, ch['char_id']), layout_source=layout)
            roles.append(ch)
        inherited = None
        if mask_source == 'auto':
            cached = pipeline['bundle']['runtime_handle'].adaptive.get('mask')
            if cached is not None:
                inherited = cached.detach().cpu().clone()
        settings = dict(pipeline['settings'], adaptive_mode=second_adaptive)
        model, pipe, diag = patch_pipeline(pipeline['source_model'], roles, layout, settings, stage='二采', inherited_masks=inherited)
        if pipeline.get('shape_control') is not None:
            from .shape_control import rebuild_shape, attach_shape
            spec = rebuild_shape(pipeline['shape_control'], pipeline)
            if spec is None:
                diag += '\n二采已关闭LLLite，仅继承人物区域与LoRA身份；不再输入参考结构特征。'
            else:
                attach_shape(model, pipe, spec)
                diag += '\n二采LLLite按显式强度设置继承，条件图与效果遮罩按实际token网格缩放。'
        from .core import build_gates
        gates = build_gates(roles, [], th // 2, tw // 2, device='cpu', exclusive=settings['overlap_mode'] == 'nearest', raw_masks=inherited, preserve_mass=settings.get('preserve_gate_mass', False)) if roles else torch.empty((0, th // 2, tw // 2))
        preview = render_layout(layout, gates, roles)
        report = f"{w * 8}×{h * 8} → {tw * 8}×{th * 8}；来源={('上一采动态区域' if inherited is not None else '实例布局' if 'raster_masks' in layout else '手工布局')}；人物区域与交融边带已缩放；LoRA 从原底模重建、未叠加；动态统计与文字缓存独立。\n" + diag
        LOG.info('[AnimaContact][HIRES] %s', report)
        return (model, enlarged, pipe, preview, tw * 8, th * 8, report)

class _RegionalInspect:
    RETURN_TYPES = ('IMAGE', 'STRING')
    FUNCTION = 'inspect'

    def inspect(self, pipeline, samples, image=None):
        runtime = pipeline['bundle']['runtime_handle'].runtime
        gates = runtime.get('last_gates')
        if gates is None:
            raise RuntimeError('管线尚未执行采样，请连接同一模型对应的采样输出')
        if tuple(samples['samples'].shape[-2:]) != (pipeline['layout']['height'] // 8, pipeline['layout']['width'] // 8):
            raise ValueError('检查节点的 LATENT 与管线画布尺寸不一致')
        preview = render_layout(pipeline['layout'], gates, pipeline['roles'])
        if image is not None:
            bg = F.interpolate(image[:1].movedim(-1, 1), size=preview.shape[1:3], mode='bilinear', align_corners=False).movedim(1, -1).cpu()
            preview = bg * 0.65 + preview * 0.35
        report = f"{pipeline['stage']}；实际前向次数={runtime['calls']}；token网格={tuple(gates.shape[-2:])}；最近一次各角色增量调用={runtime.get('delta_calls')}；交叉注意力重算={runtime.get('cross_calls')}。预览为实际身份门控（不含交融区额外的 LoRA 释放）；不是自动人物分割图。"
        if pipeline.get('shape_control') is not None:
            cr = pipeline['shape_control'].runtime
            report += f" LLLite实际执行={cr.get('calls', 0)}次，退出/跳过={cr.get('skipped', 0)}次，最近强度={cr.get('last_strength', 0):.3f}。"
        return {'ui': {'text': [report]}, 'result': (preview, report)}

def thumbnail(image, max_side=640, nearest=False):
    a = (image[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    im = Image.fromarray(a)
    im.thumbnail((max_side, max_side), Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC)
    b = io.BytesIO()
    im.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

class _IdentitySeeds:
    RETURN_TYPES = (SEEDS, 'IMAGE', 'INT', 'INT', 'STRING')
    FUNCTION = 'prepare'
    OUTPUT_NODE = True

    def prepare(self, image, canvas_width, canvas_height, expected_people, strokes_json):
        if image.shape[0] != 1:
            raise ValueError('人物种子仅支持单张参考图')
        if canvas_width % 16 or canvas_height % 16:
            raise ValueError('生成宽高必须为16倍数')
        labels, key = seed_labels(image, canvas_width, canvas_height, expected_people, strokes_json)
        rgb = to_rgb(image, canvas_width, canvas_height).astype(np.float32) / 255
        colors = np.zeros_like(rgb)
        from .layout import COLORS
        characters = [dict(id=f'char{i}', label=f'char{i}') for i in range(1, expected_people + 1)]
        for i in range(1, expected_people + 1):
            colors[labels == i] = [int(COLORS[i - 1][j:j + 2], 16) / 255 for j in (1, 3, 5)]
        colors[labels == 255] = [0.9, 0.15, 0.65]
        active = labels > 0
        rgb[active] = rgb[active] * 0.5 + colors[active] * 0.5
        missing = [c['id'] for i, c in enumerate(characters, 1) if not np.any(labels == i)]
        report = '缺少 ' + ', '.join(missing) + '；请先标注再运行分割。' if missing else '所有人物已有种子；可运行分割，检查轮廓后再采样。'
        ui = dict(mode='seeds', background=thumbnail(image), labels=thumbnail(torch.zeros_like(image)), characters=characters, width=canvas_width, height=canvas_height, source_id=key)
        seeds = dict(labels=labels, count=expected_people, source_id=key, width=canvas_width, height=canvas_height)
        return {'ui': {'anima_mask_editor': [ui]}, 'result': (seeds, torch.from_numpy(rgb.copy())[None], canvas_width, canvas_height, report)}

class _PersonDetect:
    RETURN_TYPES = (BOXES, 'IMAGE', 'INT', 'STRING')
    FUNCTION = 'detect'
    OUTPUT_NODE = True

    def detect(self, image, detector, expected_people, confidence, nms_iou, resolution):
        if image.shape[0] != 1:
            raise ValueError('二次元检测只接受单张参考图')
        import folder_paths
        rgb = to_rgb(image, image.shape[2], image.shape[1])
        path = folder_paths.get_full_path_or_raise('anima_contact_detectors', detector)
        records = detect_anime(rgb, path, confidence, nms_iou, resolution)
        actual = len(records)
        if actual > 16:
            raise ValueError(f'检测到{actual}人，超过当前16身份上限；请先排除重复框或裁切参考图，不会静默截掉人物')
        target = expected_people or actual
        report = f'二次元专用检测：实际{len(records)}人，期望{expected_people}人；conf={confidence:g}，NMS IoU={nms_iou:g}。框按中心从左到右绑定char。'
        if expected_people and len(records) != expected_people:
            report += '请检查重复框/漏框，尝试阈值或其他动漫检测权重，仍失败可用种子路线；不会按画面左右虚构人物。'
        data = dict(records=records, expected_people=target, requested_people=expected_people, source_id=reference_key(image, image.shape[2], image.shape[1], target), width=image.shape[2], height=image.shape[1])
        logging.info('[AnimaContact][ANIME] %s', report)
        return {'ui': {'text': [report]}, 'result': (data, box_preview(rgb, records), len(records), report)}

class _PersonSegment:
    RETURN_TYPES = (LAYOUT, 'IMAGE', 'IMAGE', 'INT', 'INT', 'STRING')
    FUNCTION = 'analyze'

    def analyze(self, image, people, sam2_checkpoint, canvas_width, canvas_height, analysis_resolution, contact_width, manual_seeds=None, refinement='off', foreground_mask=None):
        if canvas_width % 16 or canvas_height % 16:
            raise ValueError('目标宽高必须是16倍数')
        count = people['expected_people']
        records = people['records']
        if count < 1:
            raise ValueError('没有检测到人物，请检查检测预览或使用种子路线')
        if len(records) != count:
            raise ValueError(f'动漫专用检测实际{len(records)}人，期望{count}人；先看检测节点的框预览，不能凭空补齐人')
        if image.shape[0] != 1 or people['source_id'] != reference_key(image, image.shape[2], image.shape[1], count):
            raise ValueError('人物框与当前原图不同，禁止套用旧人物位置')
        manual = None
        if manual_seeds is not None:
            if manual_seeds['source_id'] != reference_key(image, canvas_width, canvas_height, count):
                raise ValueError('手工种子与原图/画布不一致')
            manual = manual_seeds['labels']
        import folder_paths
        checkpoint = folder_paths.get_full_path_or_raise('sam2', sam2_checkpoint)
        rgb = to_rgb(image, image.shape[2], image.shape[1])
        if refinement not in ('off', 'mask_refine'):
            raise ValueError('未知分割精修模式')
        foreground = None
        if foreground_mask is not None:
            alpha = foreground_mask.detach().float().cpu()
            if alpha.ndim == 3 and alpha.shape[0] == 1:
                alpha = alpha[0]
            if alpha.ndim != 2 or not torch.isfinite(alpha).all():
                raise ValueError('Expected one finite foreground probability map')
            foreground = alpha.numpy().clip(0, 1)
        layout, uncertain = regions_from_boxes(rgb, records, checkpoint, canvas_width, canvas_height, analysis_resolution, contact_width, manual, refine=refinement == 'mask_refine', foreground=foreground)
        layout['reference_image'] = image.detach().cpu()
        layout['layout_id'] = reference_key(image, canvas_width, canvas_height, count) + '-' + layout['layout_id']
        report = '动漫人物框＋SAM2轮廓；未用深度或通用人体骨架。重叠肢体仍可能错分，请复核并涂改；报告颜色不是角色发色。'
        report += f' 分割分析长边={analysis_resolution}，精修={refinement}。'
        return (layout, overlay_image(layout, image, uncertain), label_image(layout), canvas_width, canvas_height, report)

class _IdentityMapImport:
    RETURN_TYPES = (LAYOUT, 'IMAGE', 'INT', 'INT')
    FUNCTION = 'load'

    def load(self, id_image, canvas_width, canvas_height, expected_people, contact_width, reference=None):
        if canvas_width % 16 or canvas_height % 16 or id_image.shape[0] != 1:
            raise ValueError('单张ID图，目标宽高16倍数')
        from .layout import COLORS
        a = to_rgb(id_image, id_image.shape[2], id_image.shape[1]).astype(np.float32)
        labels = np.zeros(a.shape[:2], np.uint8)
        best = np.full(a.shape[:2], np.inf)
        for i in range(expected_people):
            color = np.array([int(COLORS[i][j:j + 2], 16) for j in (1, 3, 5)])
            dist = np.linalg.norm(a - color, axis=-1)
            select = (dist < 45) & (dist < best)
            labels[select] = i + 1
            best[select] = dist[select]
        if set(np.unique(labels)) - {0} != set(range(1, expected_people + 1)):
            raise ValueError('未找到全部固定人物ID颜色，请使用案例标签图或检查颜色。不要把普通参考RGB当作ID图。')
        layout = label_layout(labels, canvas_width, canvas_height, contact_width, reference)
        return (layout, label_image(layout), canvas_width, canvas_height)

class _SeedSegment:
    RETURN_TYPES = _PersonSegment.RETURN_TYPES
    FUNCTION = 'analyze'

    def analyze(self, image, seeds, sam2_checkpoint, analysis_resolution, contact_width):
        from .auto_regions import sam2_masks, sample_points, partition_foreground
        w0, h0 = (seeds['width'], seeds['height'])
        count = seeds['count']
        if image.shape[0] != 1 or seeds['source_id'] != reference_key(image, w0, h0, count):
            raise ValueError('种子与原图/画布不同')
        ratio = analysis_resolution / max(w0, h0)
        w = max(16, round(w0 * ratio))
        h = max(16, round(h0 * ratio))
        rgb = to_rgb(image, w, h)
        manual = np.array(Image.fromarray(seeds['labels']).resize((w, h), Image.Resampling.NEAREST))
        points = [sample_points(manual == i) for i in range(1, count + 1)]
        missing = [f'char{i + 1}' for i, p in enumerate(points) if not len(p)]
        if missing:
            raise ValueError('缺少种子：' + ', '.join(missing) + '；请先在身份种子节点标注每个人物')
        import folder_paths
        checkpoint = folder_paths.get_full_path_or_raise('sam2', sam2_checkpoint)
        masks = sam2_masks(rgb, points, checkpoint, sample_points(manual == 255))
        labels, _, uncertain = partition_foreground(rgb, masks.any(0).astype(np.float32), [], foreground=masks.any(0), manual_labels=manual, manual_count=count, sam_prior=masks, contact_px=max(0, round(contact_width * ratio)))
        layout = label_layout(labels, w0, h0, contact_width, image.detach().cpu(), [dict(char_id=f'char{i}', source='manual_sam2') for i in range(1, count + 1)])
        return (layout, overlay_image(layout, image, uncertain), label_image(layout), w0, h0, '仅人物种子＋SAM2，未使用深度图或自动骨架。')

class _LayoutControl:
    RETURN_TYPES = ('IMAGE', 'MASK')
    FUNCTION = 'make'

    def make(self, layout, mode, smooth_pixels, line_width, effect_margin, release_mask=None):
        from .shape_control import layout_control
        return layout_control(layout, mode, smooth_pixels, line_width, effect_margin, release_mask)

class _LLLiteApply:
    RETURN_TYPES = ('MODEL', PIPE, 'STRING')
    FUNCTION = 'apply'

    def apply(self, pipeline, control, image, strength, start_percent, end_percent, fade_out, second_pass_strength, effect_mask=None):
        from .shape_control import make_spec, attach_shape
        m, pipe, report = patch_pipeline(pipeline['source_model'], pipeline['roles'], pipeline['layout'], pipeline['settings'], stage=pipeline['stage'])
        spec = make_spec(control, image, effect_mask, pipeline['source_model'], strength, start_percent, end_percent, fade_out, second_pass_strength)
        attach_shape(m, pipe, spec)
        report += f'\nLLLite {start_percent:g}→{end_percent:g}，尾段{fade_out:g}渐退，二采强度{second_pass_strength:g}；控制简图不应输入原图RGB。'
        return (m, pipe, report)

class _PaintCore:
    RETURN_TYPES = (LAYOUT, 'IMAGE', 'IMAGE', 'STRING')
    FUNCTION = 'paint'

    def paint(self, layout, strokes_json, island_pixels, hole_pixels, smooth_radius):
        base = clean_layout(layout, island_pixels, hole_pixels, smooth_radius)
        base['layout_id'] = layout['layout_id']
        out = paint_layout(base, strokes_json)
        out['cleanup_stats'] = base['cleanup_stats']
        ids = label_image(out)
        image = out.get('reference_image')
        preview = overlay_image(out, image) if image is not None else ids
        stats = base['cleanup_stats']
        report = f"全部{len(out['characters'])}个身份已载入；清理改变{stats['changed_pixels']}像素，其中小岛{stats['removed_pixels']}、小孔{stats['filled_pixels']}；随后应用手工笔画。色块仅表示人物归属，预览RGB不用于身份编码。"
        ui = dict(background=thumbnail(image) if image is not None else thumbnail(ids), labels=thumbnail(label_image(base), nearest=True), characters=out['characters'], width=out['width'], height=out['height'], source_id=layout['layout_id'])
        return {'ui': {'anima_mask_editor': [ui]}, 'result': (out, preview, ids, report)}

class _IdentityGains(_IdentityCore):

    def make(self, clip, layout, scene_text, char_id, lora_name, lora_strength, identity_prompt, edge_softness, lora_mode, identity_gain, self_q_out_gain, self_kv_gain, cross_gain, mlp_gain, previous_roles=None):
        if lora_mode == 'selected' and lora_name == 'None':
            raise ValueError('启用LoRA后请选择权重；只用底模请选disabled')
        result = super().make(clip, layout, scene_text, char_id, 'None' if lora_mode == 'disabled' else lora_name, lora_strength, identity_prompt, edge_softness, previous_roles)
        ch = result[0][-1]
        ch['identity_gain'] = float(identity_gain)
        ch['projection_gains'] = dict(self_q_out=float(self_q_out_gain), self_kv=float(self_kv_gain), cross=float(cross_gain), mlp=float(mlp_gain))
        return result

class _TileUpscale:
    RETURN_TYPES = ('MODEL', 'LATENT', PIPE, 'IMAGE', 'INT', 'INT', 'STRING')
    FUNCTION = 'prepare'

    def prepare(self, pipeline, samples, vae, tile_control, scale_factor, tile_strength, start_percent, end_percent):
        from .shape_control import make_spec, attach_shape
        if tile_control.profile != 'tile-repair-v20':
            raise ValueError('请接专用TileRepair v2加载器，不接轮廓CN')
        p = dict(pipeline)
        p.pop('shape_control', None)
        p.pop('structure_control', None)
        model, enlarged, pipe, _, w, h, report = _LatentUpscale().prepare(p, samples, scale_factor, 'bilinear', 'auto', 'off')
        decoded = vae.decode(samples['samples'])
        if decoded.ndim == 5 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        if decoded.ndim != 4 or decoded.shape[0] != 1:
            raise ValueError('Tile桥接当前只支持单张图片；不自动拆批次/视频')
        if decoded.shape[1:3] != (pipeline['layout']['height'], pipeline['layout']['width']):
            raise ValueError('首采解码图与管线尺寸不同')
        rgb = F.interpolate(decoded[..., :3].movedim(-1, 1), size=(h, w), mode='bicubic', align_corners=False).movedim(1, -1).clamp(0, 1)
        latent = dict(enlarged, samples=vae.encode(rgb))
        if tuple(latent['samples'].shape[-2:]) != (h // 8, w // 8):
            raise ValueError('VAE编码尺寸与目标不一致')
        spec = make_spec(tile_control, rgb, None, pipeline['source_model'], tile_strength, start_percent, end_percent, 0.0, 0.0)
        attach_shape(model, pipe, spec)
        report += '\n首采生成RGB图放大并VAE编码；Tile为此生成图提供细节约束；首采参考轮廓/深度已关闭。二采建议denoise=.20，steps=20；不能保证逐像素不变。'
        return (model, latent, pipe, rgb, w, h, report)

def normalize_text(text):
    return ', '.join((x.strip() for x in str(text).replace('，', ',').replace('、', ',').split(',') if x.strip()))

def identity_cards(layout, names=None):
    names = names or {}
    cards = []
    image = layout.get('reference_image')
    for ch in layout['characters']:
        cid = ch['id']
        mask = layout['raster_masks'][cid][0].cpu()
        yy, xx = torch.where(mask > 0.5)
        card = dict(id=cid, label=names.get(cid) or ch.get('label') or cid, color=COLORS[int(cid[4:]) - 1])
        if len(xx):
            x0, x1 = (int(xx.min()), int(xx.max()) + 1)
            y0, y1 = (int(yy.min()), int(yy.max()) + 1)
            card['center'] = [float(xx.float().mean() / layout['width']), float(yy.float().mean() / layout['height'])]
            if image is not None:
                import torch.nn.functional as F
                rgb = F.interpolate(image[:1, ..., :3].movedim(-1, 1).float(), size=(layout['height'], layout['width']), mode='bilinear', align_corners=False).movedim(1, -1).cpu()
                crop = rgb[:, y0:y1, x0:x1] * (0.15 + 0.85 * mask[y0:y1, x0:x1][None, :, :, None])
                card['thumbnail'] = thumbnail(crop, max_side=128)
        cards.append(card)
    return cards

class _OwnershipEditor(_PaintCore):

    def paint(self, layout, strokes_json, island_pixels, hole_pixels, smooth_radius, identity_names_json):
        names = json.loads(identity_names_json or '{}')
        ids = {c['id'] for c in layout['characters']}
        names = {k: v.strip()[:100] for k, v in names.items() if k in ids and isinstance(v, str)} if isinstance(names, dict) else {}
        strokes = json.loads(strokes_json or '{}')
        if not isinstance(strokes, dict) or not isinstance(strokes.get('strokes', []), list):
            raise ValueError('笔画数据格式无效，请清空笔画后重试')
        reset = bool(strokes.get('strokes') and strokes.get('source_id') and (strokes['source_id'] != layout.get('layout_id')))
        if reset:
            strokes = dict(version=1, strokes=[])
        strokes['strokes'] = [s for s in strokes.get('strokes', []) if s.get('char_id') in ids | {'erase'}]
        strokes['source_id'] = layout['layout_id']
        result = super().paint(layout, json.dumps(strokes), island_pixels, hole_pixels, smooth_radius)
        out = result['result'][0]
        for ch in out['characters']:
            ch['label'] = names.get(ch['id']) or ch['id']
        cards = identity_cards(out, names)
        ui = result['ui']['anima_mask_editor'][0]
        ui.update(characters=cards, version=3, strokes_state=strokes, names_state=names, reset_reason='参考区域变化，已自动清空旧笔画' if reset else '')
        if reset:
            result['result'] = (*result['result'][:3], result['result'][3] + '\n参考区域变化，已自动清空旧笔画。')
        out['identity_cards'] = cards
        return result

class _SceneConditioning:
    FUNCTION = 'make'
    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING', 'STRING', POLICY)

    def make(self, clip, scene_prompt, quality_prompt, negative_prompt):
        scene = normalize_text(scene_prompt)
        quality = normalize_text(quality_prompt)
        full = ', '.join((x for x in [quality, scene] if x))
        reference = encode(clip, scene)
        return (encode(clip, full), encode(clip, normalize_text(negative_prompt)), scene, dict(scene_text=scene, reference_conditioning=reference, quality=quality))

class _Identity(_IdentityGains):

    def make(self, clip, layout, scene_text, char_id, identity_prompt, edge_softness, identity_gain, self_q_out_gain, self_kv_gain, cross_gain, mlp_gain, lora_stack_json, unmatched_policy, previous_roles=None, scene_policy=None):
        if char_id not in {c['id'] for c in layout['characters']}:
            return {'ui': {'anima_identity': [dict(id=char_id, label='当前参考无此人物 · 已跳过', color=COLORS[int(char_id[4:]) - 1])]}, 'result': (list(previous_roles or []),)}
        raw = json.loads(lora_stack_json or '{}')
        entries = raw.get('entries', [])
        if not isinstance(entries, list) or len(entries) > 12:
            raise ValueError('每身份LoRA组最多12条')
        import folder_paths
        available = set(folder_paths.get_filename_list('loras'))
        loras = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError('LoRA条目格式无效')
            if not entry.get('enabled', True):
                continue
            name = entry.get('name', '')
            strength = float(entry.get('strength', 1.0))
            if not torch.isfinite(torch.tensor(strength)) or not -2 <= strength <= 2:
                raise ValueError('LoRA倍率范围-2~2')
            if not name or strength == 0:
                continue
            if name not in available:
                if unmatched_policy == 'skip_with_report':
                    loras.append(dict(path=None, label=name, strength=strength, on_unmatched=unmatched_policy, missing=True))
                    continue
                raise ValueError(f'{char_id} 的LoRA文件不存在：{name}；请选择本地实际文件')
            loras.append(dict(path=folder_paths.get_full_path_or_raise('loras', name), label=name, strength=strength, on_unmatched=unmatched_policy, purpose=entry.get('purpose', 'identity')))
        result = super().make(clip, layout, normalize_text(scene_text), char_id, 'None', 1.0, normalize_text(identity_prompt), edge_softness, 'disabled', identity_gain, self_q_out_gain, self_kv_gain, cross_gain, mlp_gain, previous_roles)
        role = result[0][-1]
        role['loras'] = loras
        if scene_policy is not None:
            if normalize_text(scene_text) != scene_policy['scene_text']:
                raise ValueError('身份文字和分离质量策略应连接同一个共享场景节点')
            role['identity_reference_conditioning'] = scene_policy['reference_conditioning']
        card = next((x for x in layout.get('identity_cards', []) if x['id'] == char_id), dict(id=char_id, label=char_id, color=COLORS[int(char_id[4:]) - 1]))
        role['name'] = card['label']
        return {'ui': {'anima_identity': [card]}, 'result': result}

class _RegionalApply:

    def apply(self, model, roles=None, layout=None, identity_strength=1.0, cross_lora_strength=1.0, self_kv_mode='regional', overlap_mode='nearest', contact_lora_release=0.0, adaptive_mode='off', adaptive_calls=2, adaptive_blend=0.25, self_attention_mode='local_output', cfg_lora_mode='both', gate_mode='partition'):
        roles = list(roles or [])
        refs = [r.get('identity_reference_conditioning') for r in roles]
        if any((r is not None for r in refs)) and (not all((r is refs[0] for r in refs))):
            raise ValueError('全部身份应连接同一个场景质量策略，或全部不连接；不能混用不同场景基线')
        ids = {c['id'] for c in layout['characters']}
        bound = {r['char_id'] for r in roles}
        if bound - ids:
            raise ValueError(f'身份节点引用不存在的人物：{sorted(bound - ids)}')
        if len(bound) != len(roles):
            raise ValueError('角色ID重复')
        for r in roles:
            previous = r['layout_source']
            if previous is not layout and (not (previous.get('layout_id') == layout.get('layout_id') and previous['width'] == layout['width'] and (previous['height'] == layout['height']))):
                raise ValueError('所有身份和路由必须接同一布局')
        settings = dict(identity_strength=float(identity_strength), cross_lora_strength=float(cross_lora_strength), self_kv_mode=self_kv_mode, overlap_mode=overlap_mode, contact_mix=float(contact_lora_release), adaptive_mode=adaptive_mode, adaptive_calls=int(adaptive_calls), adaptive_blend=float(adaptive_blend), apply_uncond=cfg_lora_mode == 'both', local_self_attention=self_attention_mode == 'local_output', separate_uncond=True, preserve_gate_mass=gate_mode == 'partition')
        m, pipe, report = patch_pipeline(model, roles, layout, settings)
        report += f'\n生效身份：{sorted(bound)}；未绑定人物：{sorted(ids - bound)}（保留底模与全局场景，不重排char ID）。'
        report += '\n质量/画风只在全局基线保留一次；身份差分以同场景无质量条件为基准（连接场景基线策略时）。'
        return {'ui': {'text': [report]}, 'result': (m, pipe, report)}
