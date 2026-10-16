# Node reference / 节点参考

Node IDs and input keys are stable API identifiers. Display names follow the ComfyUI interface language. / 节点 ID 与输入键为稳定接口标识，显示名称随 ComfyUI 界面语言切换。

## Anima Region Canvas / Anima 区域画布

`AnimaRegionCanvas` · `Anima Contact/Masks`

Edit normalized spatial ownership geometry. 编辑归一化空间归属几何数据。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `canvas_width` | Target image width in pixels; must be divisible by 16.<br>目标图像宽度，单位为像素，必须为16的倍数。 |
| `canvas_height` | Target image height in pixels; must be divisible by 16.<br>目标图像高度，单位为像素，必须为16的倍数。 |
| `layout_json` | Normalized body, face, ownership and contact geometry maintained by the visual editor.<br>由可视化编辑器维护的归一化躯干、面部、归属和接触区域几何数据。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `preview` (IMAGE), `width` (INT), `height` (INT)

## Anima Scene Conditioning / Anima 场景条件

`AnimaSceneConditioning` · `Anima Contact/Conditioning`

Encode shared scene, global style and negative conditioning. 编码共享场景、全局画风与负面条件。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `clip` | Anima-compatible text encoder used to encode shared and regional conditioning.<br>用于编码共享和区域条件的 Anima 兼容文本编码器。 |
| `scene_prompt` | Scene-level composition, interaction, environment and camera conditioning shared by all identities.<br>由所有身份共享的构图、交互、环境和镜头条件。 |
| `quality_prompt` | Style and quality conditioning applied to the global baseline, separate from identity residuals.<br>作用于全局基线并与身份残差分离的画风和质量条件。 |
| `negative_prompt` | Global negative conditioning for classifier-free guidance.<br>用于无分类器引导的全局负面条件。 |

Outputs / 输出：`positive` (CONDITIONING), `negative` (CONDITIONING), `scene_text` (STRING), `scene_policy` (ANIMA_SCENE_POLICY)

## Anima Identity / Anima 身份条件

`AnimaIdentity` · `Anima Contact/Conditioning`

Bind regional text and optional local MODEL adapters to one ownership identifier. 将区域文本和可选局部 MODEL 适配器绑定到归属标识。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `clip` | Anima-compatible text encoder used to encode shared and regional conditioning.<br>用于编码共享和区域条件的 Anima 兼容文本编码器。 |
| `layout` | Spatial ownership masks, region identifiers and contact boundaries.<br>包含空间归属遮罩、区域标识和接触边界的数据。 |
| `scene_text` | Shared semantic context encoded together with each identity description.<br>与每个身份描述共同编码的共享语义上下文。 |
| `char_id` | Identifier linking an identity conditioning branch to an ownership region; independent of display names.<br>将身份条件分支与归属区域关联的标识，与显示名称无关。 |
| `identity_prompt` | Appearance, clothing and distinguishing attributes for one regional conditioning branch.<br>单个区域条件分支的外观、服装和辨识属性。 |
| `edge_softness` | Smoothing radius of ownership gates in model image tokens; larger values reduce edge specificity.<br>以模型图像 token 为单位的归属门控平滑半径；增大后边界特异性降低。 |
| `identity_gain` | Multiplier for this identity's conditioning residual, independent of LoRA scale.<br>当前身份条件残差的倍率，与 LoRA 倍率相互独立。 |
| `self_q_out_gain` | Multiplier for local LoRA increments on self-attention query and output projections.<br>局部 LoRA 在自注意力查询和输出投影上的增量倍率。 |
| `self_kv_gain` | Multiplier for local LoRA increments on self-attention key and value projections.<br>局部 LoRA 在自注意力键和值投影上的增量倍率。 |
| `cross_gain` | Per-identity multiplier for supported cross-attention LoRA increments.<br>当前身份受支持交叉注意力 LoRA 增量的倍率。 |
| `mlp_gain` | Multiplier for local LoRA increments on supported feed-forward projections.<br>局部 LoRA 在受支持前馈网络投影上的增量倍率。 |
| `lora_stack_json` | Serialized local MODEL adapters with independent enable state and scale; an empty stack uses the base model.<br>包含独立开关和倍率的局部 MODEL 适配器序列化数据；空组使用底模。 |
| `previous_roles` | Optional identity chain extended by this node.<br>由此节点扩展的可选身份链。 |
| `scene_policy` | Shared unstyled reference conditioning used to compute identity residuals consistently.<br>用于一致计算身份残差的共享无画风参考条件。 |

Outputs / 输出：`roles` (ANIMA_IDENTITIES)

## Anima Regional Apply / Anima 区域路由

`AnimaRegionalApply` · `Anima Contact/Routing`

Build spatial text and adapter residual routing while retaining the shared model baseline. 构建空间文本与适配器残差路由，同时保留共享模型基线。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `model` | Unmodified source diffusion model used to construct a regional sampling pipeline.<br>用于构建区域采样管线的原始扩散模型。 |
| `layout` | Spatial ownership masks, region identifiers and contact boundaries.<br>包含空间归属遮罩、区域标识和接触边界的数据。 |
| `identity_strength` | Global scale of regional text residuals relative to the shared conditioning baseline.<br>区域文本相对共享条件基线的残差全局倍率。 |
| `cross_lora_strength` | Global scale of cross-attention adapter residuals; does not scale identity text residuals.<br>交叉注意力适配器残差的全局倍率，不缩放身份文本残差。 |
| `self_kv_mode` | Whether regional adapters affect self-attention key and value projections.<br>是否让区域适配器影响自注意力的键和值投影。 |
| `overlap_mode` | Policy used to allocate tokens covered by multiple ownership regions.<br>用于分配被多个归属区域覆盖的 token 的策略。 |
| `contact_lora_release` | Fraction of adapter influence removed inside contact boundaries; identity text remains active.<br>在接触边界内移除的适配器影响比例；身份文本仍保持有效。 |
| `adaptive_mode` | Optional attention-based ownership refinement; requires explicit spatial priors.<br>可选的基于注意力的归属精修，需要明确空间先验。 |
| `adaptive_calls` | Number of initial model forward calls used for adaptive ownership observations.<br>用于自适应归属观测的初始模型前向调用次数。 |
| `adaptive_blend` | Interpolation weight between observed affinity and the supplied ownership prior.<br>观测亲和度与输入归属先验之间的插值权重。 |
| `self_attention_mode` | Local output residual routing or shared projection routing for regional adapters.<br>区域适配器采用局部输出残差路由或共享投影路由。 |
| `cfg_lora_mode` | Applies adapters to both guidance branches or only the conditional branch; negative text is not replaced.<br>适配器作用于两个引导分支或仅条件分支；不替换负面文本。 |
| `gate_mode` | Partition gates preserve covered-token magnitude after exclusive ownership allocation.<br>分区门控在独占归属分配后保留已覆盖 token 的幅度。 |
| `self_region_isolation` | Experimental suppression of attention between different identity regions; zero preserves native attention.<br>实验性抑制不同身份区域之间的注意力；零保留原生注意力。 |
| `isolation_start_layer` | Fractional transformer depth after which spatial self-attention isolation is applied.<br>开始应用空间自注意力隔离的 Transformer 层深度比例。 |
| `roles` | Active identity branches assigned to this sampling pipeline.<br>分配到当前采样管线的有效身份分支。 |

Outputs / 输出：`model` (MODEL), `pipeline` (ANIMA_PIPELINE), `report` (STRING)

## Anima Person Detection / Anima 人物检测

`AnimaPersonDetect` · `Anima Contact/Segmentation`

Detect anime person instances and assign deterministic spatial identifiers. 检测二次元人物实例并分配确定的空间标识。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |
| `detector` | Anime person detector filename in the registered model directory.<br>已注册模型目录中的二次元人物检测模型文件名。 |
| `expected_people` | Zero accepts the detected count; a positive value verifies the count before segmentation.<br>零表示采用检测人数；正值在分割前核对人数。 |
| `confidence` | Minimum person detection confidence accepted before non-maximum suppression.<br>非极大值抑制前接受的人物检测最低置信度。 |
| `nms_iou` | Intersection-over-union threshold for suppressing duplicate person boxes.<br>用于抑制重复人物框的交并比阈值。 |
| `resolution` | Input spatial resolution used by dynamically shaped detection models.<br>动态输入形状检测模型使用的空间分辨率。 |
| `auto_download` | Download missing catalogued weights on execution, verify SHA256 and install atomically.<br>执行时下载目录清单中的缺失权重，验证 SHA256 并原子安装。 |

Outputs / 输出：`people` (ANIMA_PEOPLE), `preview` (IMAGE), `count` (INT), `report` (STRING)

## Anima Instance Segmentation / Anima 实例分割

`AnimaPersonSegment` · `Anima Contact/Segmentation`

Combine detection boxes, SAM2 and optional foreground priors to allocate visible pixels. 结合检测框、SAM2 与可选前景先验分配可见像素。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |
| `people` | Validated detection boxes and their source image association.<br>已验证的人物检测框及其源图像关联信息。 |
| `sam2_checkpoint` | Official SAM2.1 checkpoint filename matching the registered model configuration.<br>与已注册模型配置匹配的官方 SAM2.1 权重文件名。 |
| `canvas_width` | Target image width in pixels; must be divisible by 16.<br>目标图像宽度，单位为像素，必须为16的倍数。 |
| `canvas_height` | Target image height in pixels; must be divisible by 16.<br>目标图像高度，单位为像素，必须为16的倍数。 |
| `analysis_resolution` | Longest side used for instance segmentation and edge-aware ownership allocation.<br>实例分割和边缘感知归属分配使用的最长边分辨率。 |
| `contact_width` | Radius in target pixels used to mark adjacent identity boundaries.<br>用于标记相邻身份边界的目标图像像素半径。 |
| `manual_seeds` | Optional positive identity and background annotations constraining instance segmentation.<br>用于约束实例分割的可选身份正标注和背景标注。 |
| `refinement` | Optional second SAM2 prediction conditioned on the selected low-resolution mask logits.<br>可选的第二次 SAM2 预测，使用已选低分辨率遮罩 logits 作为条件。 |
| `foreground_mask` | Optional union foreground probability map used to recover missing pixels without defining instance identity.<br>用于补回漏选像素的可选联合前景概率图，不定义逐人身份。 |
| `auto_download` | Download missing catalogued weights on execution, verify SHA256 and install atomically.<br>执行时下载目录清单中的缺失权重，验证 SHA256 并原子安装。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `preview` (IMAGE), `identity_map` (IMAGE), `width` (INT), `height` (INT), `report` (STRING)

## Anima Ownership Editor / Anima 归属编辑器

`AnimaOwnershipEditor` · `Anima Contact/Masks`

Clean discrete ownership labels and edit identity assignment with persistent strokes. 清理离散归属标签，并通过持久化笔画编辑身份分配。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `layout` | Spatial ownership masks, region identifiers and contact boundaries.<br>包含空间归属遮罩、区域标识和接触边界的数据。 |
| `strokes_json` | Serialized normalized brush strokes bound to the current reference layout.<br>与当前参考布局绑定的归一化画笔笔画序列化数据。 |
| `island_pixels` | Maximum component area removed as isolated noise; each identity's largest component is retained.<br>作为孤立噪点移除的最大连通块面积；每个身份的最大连通块保留。 |
| `hole_pixels` | Maximum enclosed background area filled when the surrounding identity is sufficiently unambiguous.<br>周围身份归属足够明确时允许填补的封闭背景区域最大面积。 |
| `smooth_radius` | Gaussian competition radius in target pixels for mutually exclusive identity labels.<br>用于互斥身份标签竞争的高斯半径，单位为目标像素。 |
| `identity_names_json` | Optional region identifier to display-name mapping; labels do not alter spatial ownership.<br>可选的区域标识与显示名称映射；名称不改变空间归属。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `preview` (IMAGE), `identity_map` (IMAGE), `report` (STRING), `width` (INT), `height` (INT)

## Anima Identity Map Import / Anima 身份图导入

`AnimaIdentityMapImport` · `Anima Contact/Masks`

Import palette-encoded identity labels as a spatial ownership layout. 将调色板编码的身份标签导入为空间归属布局。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `id_image` | Discrete ownership map encoded with the documented identity palette; background is black or white.<br>使用固定身份调色板编码的离散归属图；黑色或白色为背景。 |
| `canvas_width` | Target image width in pixels; must be divisible by 16.<br>目标图像宽度，单位为像素，必须为16的倍数。 |
| `canvas_height` | Target image height in pixels; must be divisible by 16.<br>目标图像高度，单位为像素，必须为16的倍数。 |
| `expected_people` | Number of identity palette entries expected in the imported map.<br>导入身份图中应当出现的身份调色板数量。 |
| `contact_width` | Radius in target pixels used to mark adjacent identity boundaries.<br>用于标记相邻身份边界的目标图像像素半径。 |
| `reference` | Optional RGB reference associated with the imported ownership map.<br>与导入归属图关联的可选 RGB 参考图像。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `identity_map` (IMAGE), `width` (INT), `height` (INT)

## Anima Identity Seeds / Anima 身份种子

`AnimaIdentitySeeds` · `Anima Contact/Masks`

Annotate positive identity seeds and background exclusions. 标注身份正种子和背景排除区域。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |
| `canvas_width` | Target image width in pixels; must be divisible by 16.<br>目标图像宽度，单位为像素，必须为16的倍数。 |
| `canvas_height` | Target image height in pixels; must be divisible by 16.<br>目标图像高度，单位为像素，必须为16的倍数。 |
| `expected_people` | Zero accepts the detected count; a positive value verifies the count before segmentation.<br>零表示采用检测人数；正值在分割前核对人数。 |
| `strokes_json` | Serialized normalized brush strokes bound to the current reference layout.<br>与当前参考布局绑定的归一化画笔笔画序列化数据。 |

Outputs / 输出：`seeds` (ANIMA_SEEDS), `preview` (IMAGE), `width` (INT), `height` (INT), `report` (STRING)

## Anima Seed Segmentation / Anima 种子分割

`AnimaSeedSegment` · `Anima Contact/Segmentation`

Allocate instance ownership using SAM2 and explicit identity seeds. 使用 SAM2 与明确身份种子分配实例归属。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |
| `seeds` | Per-identity positive labels and optional background exclusions tied to one reference.<br>与同一参考绑定的逐身份正标签及可选背景排除区域。 |
| `sam2_checkpoint` | Official SAM2.1 checkpoint filename matching the registered model configuration.<br>与已注册模型配置匹配的官方 SAM2.1 权重文件名。 |
| `analysis_resolution` | Longest side used for instance segmentation and edge-aware ownership allocation.<br>实例分割和边缘感知归属分配使用的最长边分辨率。 |
| `contact_width` | Radius in target pixels used to mark adjacent identity boundaries.<br>用于标记相邻身份边界的目标图像像素半径。 |
| `auto_download` | Download missing catalogued weights on execution, verify SHA256 and install atomically.<br>执行时下载目录清单中的缺失权重，验证 SHA256 并原子安装。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `preview` (IMAGE), `identity_map` (IMAGE), `width` (INT), `height` (INT), `report` (STRING)

## Anima Layout Control Image / Anima 布局控制图

`AnimaLayoutControl` · `Anima Contact/Control`

Extract structural control images and effect masks from ownership labels. 从归属标签提取结构控制图和效果遮罩。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `layout` | Spatial ownership masks, region identifiers and contact boundaries.<br>包含空间归属遮罩、区域标识和接触边界的数据。 |
| `mode` | Outline or identity-color representation generated from ownership labels.<br>从归属标签生成轮廓或身份色块控制表示。 |
| `smooth_pixels` | Pixel radius used to smooth discrete ownership boundaries before extracting control geometry.<br>提取控制几何前平滑离散归属边界的像素半径。 |
| `line_width` | Width in output pixels of the extracted outline representation.<br>提取轮廓表示的输出像素线宽。 |
| `effect_margin` | Pixel dilation of the foreground control-effect mask.<br>前景控制效果遮罩的像素膨胀距离。 |
| `release_mask` | White pixels suppress structural control locally without changing identity ownership.<br>白色像素在局部抑制结构控制，不改变身份归属。 |

Outputs / 输出：`control_image` (IMAGE), `effect_mask` (MASK)

## Anima Regional LLLite Apply / Anima 区域轻量控制

`AnimaContactLLLiteApply` · `Anima Contact/Control`

Apply scheduled structural residuals to a regional pipeline. 向区域管线应用具有时间表的结构残差。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `pipeline` | Source model, identities, ownership layout and runtime settings for one sampling stage.<br>包含原始模型、身份、归属布局和运行设置的单个采样阶段管线。 |
| `control` | Loaded Anima LLLite weights matching the supplied control representation.<br>与输入控制表示匹配的已加载 Anima LLLite 权重。 |
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |
| `strength` | Scale of the structural control residual during its active diffusion interval.<br>结构控制在有效扩散区间内的残差倍率。 |
| `start_percent` | Start position on the model's full diffusion schedule.<br>模型完整扩散时间表上的控制起始位置。 |
| `end_percent` | End position on the model's full diffusion schedule.<br>模型完整扩散时间表上的控制结束位置。 |
| `fade_out` | Length of the terminal schedule interval over which control strength decreases to zero.<br>控制强度在末段下降至零所使用的时间表区间长度。 |
| `second_pass_strength` | Explicit control strength retained when rebuilding a later sampling stage.<br>重建后续采样阶段时明确保留的控制强度。 |
| `effect_mask` | White preserves structural control and black disables it; independent of identity gates.<br>白色保留结构控制，黑色关闭结构控制；与身份门控独立。 |

Outputs / 输出：`model` (MODEL), `pipeline` (ANIMA_PIPELINE), `report` (STRING)

## Anima Tile Upscale / Anima Tile放大

`AnimaTileUpscale` · `Anima Contact/Sampling`

Resize generated RGB content, re-encode latents and prepare Tile-guided refinement. 缩放生成的 RGB 内容、重新编码潜空间并准备 Tile 引导精修。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `pipeline` | Source model, identities, ownership layout and runtime settings for one sampling stage.<br>包含原始模型、身份、归属布局和运行设置的单个采样阶段管线。 |
| `samples` | Latent samples produced by the sampling stage associated with the supplied pipeline.<br>与输入管线关联的采样阶段产生的潜空间样本。 |
| `vae` | Compatible encoder and decoder used for pixel-space resizing and latent reconstruction.<br>用于像素空间缩放和潜空间重建的兼容编码器与解码器。 |
| `tile_control` | Anima TileRepair control weights used with generated RGB content.<br>与生成 RGB 内容共同使用的 Anima TileRepair 控制权重。 |
| `scale_factor` | Spatial image scale with target dimensions aligned to the model patch grid.<br>图像空间缩放倍率；目标尺寸对齐模型 patch 网格。 |
| `tile_strength` | Residual control scale for the generated-image Tile conditioning.<br>生成图像 Tile 条件的残差控制倍率。 |
| `start_percent` | Start position on the model's full diffusion schedule.<br>模型完整扩散时间表上的控制起始位置。 |
| `end_percent` | End position on the model's full diffusion schedule.<br>模型完整扩散时间表上的控制结束位置。 |

Outputs / 输出：`model` (MODEL), `samples` (LATENT), `pipeline` (ANIMA_PIPELINE), `control_image` (IMAGE), `width` (INT), `height` (INT), `report` (STRING)

## Anima Regional Inspection / Anima 区域诊断

`AnimaRegionalInspect` · `Anima Contact/Diagnostics`

Inspect spatial gates and runtime adapter activity after sampling. 采样后检查空间门控和运行时适配器活动。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `pipeline` | Source model, identities, ownership layout and runtime settings for one sampling stage.<br>包含原始模型、身份、归属布局和运行设置的单个采样阶段管线。 |
| `samples` | Latent samples produced by the sampling stage associated with the supplied pipeline.<br>与输入管线关联的采样阶段产生的潜空间样本。 |
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |

Outputs / 输出：`preview` (IMAGE), `report` (STRING)

## Anima Foreground Segmentation / Anima 前景分割

`AnimaForeground` · `Anima Contact/Segmentation`

Predict union foreground using ISNet, ToonOut or a probability ensemble. 使用 ISNet、ToonOut 或概率集成预测联合前景。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB image used as the spatial or segmentation reference.<br>用于空间或分割参考的单张 RGB 图像。 |
| `backend` | Foreground model or model ensemble; produces a union mask without instance identifiers.<br>前景模型或模型集成；输出不含实例标识的联合前景遮罩。 |
| `auto_download` | Download missing catalogued weights on execution, verify SHA256 and install atomically.<br>执行时下载目录清单中的缺失权重，验证 SHA256 并原子安装。 |
| `fusion` | Consensus uses geometric mean; coverage uses maximum; intersection uses minimum probability.<br>共识使用几何平均，覆盖使用最大值，交集使用最小概率。 |

Outputs / 输出：`foreground_mask` (MASK), `preview` (IMAGE), `report` (STRING)

## Anima Mask Combine / Anima 遮罩融合

`AnimaMaskCombine` · `Anima Contact/Masks`

Fuse aligned foreground probability masks from independent segmentation models. 融合来自独立分割模型的对齐前景概率遮罩。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `mask_a` | First foreground probability mask in the range zero to one.<br>取值为零到一的第一张前景概率遮罩。 |
| `mask_b` | Second foreground probability mask; resized to the first mask before fusion.<br>第二张前景概率遮罩；融合前缩放到第一张遮罩的尺寸。 |
| `fusion` | Consensus uses geometric mean; coverage uses maximum; intersection uses minimum probability.<br>共识使用几何平均，覆盖使用最大值，交集使用最小概率。 |

Outputs / 输出：`foreground_mask` (MASK)

## Anima LLLite Loader / Anima LLLite加载器

`AnimaLLLiteLoader` · `Anima Contact/Models`

Load validated Anima structural or Tile control weights. 加载已验证的 Anima 结构或 Tile 控制权重。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `model_name` | Catalogued model with a fixed source revision, installation directory and SHA256 checksum.<br>具有固定来源版本、安装目录和 SHA256 校验值的模型预设。 |
| `auto_download` | Download missing catalogued weights on execution, verify SHA256 and install atomically.<br>执行时下载目录清单中的缺失权重，验证 SHA256 并原子安装。 |

Outputs / 输出：`control` (ANIMA_LLLITE), `report` (STRING)

## Anima Model Download / Anima 模型下载

`AnimaModelDownload` · `Anima Contact/Models`

Install a catalogued model with fixed-revision integrity verification. 安装目录清单中的模型，并执行固定来源版本的完整性验证。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `model_name` | Catalogued model with a fixed source revision, installation directory and SHA256 checksum.<br>具有固定来源版本、安装目录和 SHA256 校验值的模型预设。 |

Outputs / 输出：`path` (STRING)

## Anima SAM3 Semantic Segmentation / Anima SAM3 语义分割

`AnimaContactSAM3Segment` · `Anima Contact/Segmentation`

Segment all matching instances with SAM3 concept prompts. Preserve predicted masks, deduplicate queries and assign stable spatial ownership identifiers. 通过概念短语分割全部匹配实例，保留预测掩码，对重复查询去重并按空间顺序分配稳定归属标识。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB image to segment with concept prompts.<br>根据概念提示进行分割的单张 RGB 图像。 |
| `semantic_mode` | Automatic probes person, girl and boy concepts and deduplicates instance masks. Custom uses the supplied phrases. This is concept probing, not automatic caption generation.<br>自动模式依次探测 person、girl、boy 并对实例掩码去重；自定义模式使用输入短语。这是概念探测，不是自动图像描述。 |
| `semantic_text` | One concise concept phrase per line, used in custom mode. Each phrase can return multiple instances.<br>自定义模式使用的简短概念短语，每行一条；每条短语可以返回多个实例。 |
| `expected_people` | Zero accepts the result count. A positive count checks all returned instances without fabricating or truncating objects.<br>0 接受实际数量；正数核对结果数量，不虚构或截断实例。 |
| `confidence` | Minimum instance confidence after the SAM3 presence and detection score calculation.<br>SAM3 目标存在性与检测评分计算后的最低实例置信度。 |
| `mask_threshold` | Foreground probability cutoff used by the official instance postprocessor.<br>官方实例后处理用于区分前景的概率阈值。 |
| `nms_iou` | Mask IoU threshold for suppressing duplicate instances across concept queries. Larger values preserve more overlapping objects.<br>跨语义查询抑制重复实例的掩码 IoU 阈值；较大数值保留更多重叠目标。 |
| `minimum_pixels` | Discard instance masks smaller than this number of source-image pixels.<br>移除在原图中像素数小于此值的实例掩码。 |
| `canvas_width` | Target layout width in pixels; divisible by 16. Match the reference aspect ratio to avoid distortion.<br>布局输出宽度，须为 16 的倍数；保持参考图比例可避免拉伸。 |
| `canvas_height` | Target layout height in pixels; divisible by 16.<br>布局输出高度，须为 16 的倍数。 |
| `contact_width` | Boundary contact radius in target pixels; does not change the predicted instance silhouette.<br>以目标画布像素计算的边界接触半径，不改变预测的实例轮廓。 |
| `device` | Auto follows the ComfyUI device when available. CPU uses float32; CUDA uses float16. The model is released after inference.<br>自动模式跟随 ComfyUI 设备；CPU 使用 float32，CUDA 使用 float16；推理结束后释放模型。 |
| `auto_download` | Download fixed-revision Transformers weights and configuration from the public bodhicitta/sam3 repository, with upstream-matching integrity checks. No Hugging Face login is required.<br>从公开的 bodhicitta/sam3 仓库下载固定修订的 Transformers 权重与配置，并核对上游一致的校验值；无需 Hugging Face 登录。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `overlay` (IMAGE), `identity_map` (IMAGE), `foreground` (MASK), `count` (INT), `report` (STRING)

## Anima Foreground Ownership / Anima 前景身份分区

`AnimaContactForegroundOwnership` · `Anima Contact/Segmentation`

Assign union foreground pixels to explicit identity seeds using image-edge geodesic distances. Does not load SAM or a person detector. 通过图像边缘感知的测地距离，将联合前景分配给明确身份种子；不加载 SAM 或人物检测模型。

| Input / 输入 | Meaning / 含义 |
|---|---|
| `image` | Single RGB reference associated with the foreground mask and identity seeds.<br>与前景遮罩及身份种子对应的单张 RGB 参考图。 |
| `foreground_mask` | Foreground probability map. It contains subject coverage without per-person identity labels.<br>前景概率图，只表示主体覆盖范围，不包含逐人身份标签。 |
| `seeds` | Positive identity and optional background annotations from the same reference and canvas.<br>来自同一参考图与画布的身份正种子及可选背景标注。 |
| `analysis_resolution` | Longest analysis edge. Larger values preserve finer mask geometry at greater CPU and memory cost.<br>归属分析长边；更大数值保留更细轮廓，同时增加 CPU 耗时及内存占用。 |
| `mask_threshold` | Probability cutoff for foreground allocation. Explicit identity seeds remain protected.<br>分配前景所用的概率阈值，明确身份种子受到保护。 |
| `edge_weight` | Geodesic cost multiplier for foreground-probability changes; RGB intensity edges also constrain propagation.<br>前景概率变化的测地路径代价倍率，RGB 亮度边缘也约束归属传播。 |
| `contact_width` | Contact boundary radius in target-canvas pixels.<br>以目标画布像素计算的身份接触边界半径。 |

Outputs / 输出：`layout` (ANIMA_LAYOUT), `overlay` (IMAGE), `identity_map` (IMAGE), `report` (STRING)
