# Model files / 模型文件

Plugin startup creates missing registered directories. Downloads occur on node execution, not at startup. / 插件启动时创建缺失目录，权重在节点执行时按需下载。

| Preset | Path under ComfyUI/models | Source | SHA256 |
|---|---|---|---|
| `isnet` | `anima_contact_segmentation/isnetis.onnx` | [fixed revision](https://huggingface.co/skytnt/anime-seg/resolve/493cb60893f47441b26ec4fb9a306bce9e342982/isnetis.onnx) | `f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99` |
| `toonout` | `anima_contact_segmentation/birefnet-toonout-fp16.onnx` | [fixed revision](https://huggingface.co/sprited/birefnet-toonout-onnx/resolve/2ded6fe6063fe146b7b30d8b4cfc5636df322525/birefnet-toonout-fp16.onnx) | `213a8a98ee426ef8f02d247eb5a5a9889359e37c2e1e7e31e282d61034d08a83` |
| `detector-small` | `anima_contact_detectors/person_detect_v1.3_s.onnx` | [fixed revision](https://huggingface.co/deepghs/anime_person_detection/resolve/e39c744c22432ad01f91dd254fe2b02c8d878b8c/person_detect_v1.3_s/model.onnx) | `6da88929438cd442e31e45ff4f934dd2d7eb9cf7a423c22885d650ee52550f90` |
| `detector-medium` | `anima_contact_detectors/person_detect_v1.1_m.onnx` | [fixed revision](https://huggingface.co/deepghs/anime_person_detection/resolve/e39c744c22432ad01f91dd254fe2b02c8d878b8c/person_detect_v1.1_m/model.onnx) | `c80365a98ca8388fd2f01fd5abea0439e92265bceceb048bef38b0849303467c` |
| `sam2-tiny` | `sam2/sam2.1_hiera_tiny.pt` | [fixed revision](https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/de431c4043854a71d8101e17995dfe596bf101a5/sam2.1_hiera_tiny.pt) | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |
| `outline` | `model_patches/anima-lllite-any-test-like-v2.safetensors` | [fixed revision](https://huggingface.co/kohya-ss/Anima-LLLite/resolve/36ba7f2f498a1ca63cc77fc7a1298652f72d4524/anima-lllite-any-test-like-v2.safetensors) | `8863424f04a2445815a827b22cbda557a085d2752e05fab54a2b303821a9e3b2` |
| `tile` | `controlnet/animaTileRepair_v20.safetensors` | [fixed revision](https://huggingface.co/LeFeujitif/sandbox/resolve/00729a0ce2caf54a10c3268979d2bc5ecda7d1b8/animaTileRepair_v20.safetensors) | `be5a9ead6527329237a66a0eb234ccea4cddb140c1d1e1b7d0776e43f5ff1330` |
| `regional` | `model_patches/anima-lllite-regional-exp-v3.safetensors` | [fixed revision](https://huggingface.co/Sen-sou/Anima-LLLite-Regional-Controlnet/resolve/f1b7beeddb4175d2d20b14e10538db62bbb441a6/anima-lllite-regional-exp-v3.safetensors) | `f8cdc9897a4eaa50b0f9bc18dda098fb096eeb64df001589492b9774d0cf5cc5` |

## Base architecture / 底模架构

The compatible Anima base, Qwen text encoder and VAE are loaded with native ComfyUI nodes from https://huggingface.co/circlestone-labs/Anima. Their directories are models/diffusion_models, models/text_encoders and models/vae. Native CLIPLoader uses its supported stable_diffusion type with weight-based Anima encoder detection; no unsupported anima enum is introduced.

Anima 底模、Qwen 文本编码器和 VAE 由原生节点加载，目录分别为 models/diffusion_models、models/text_encoders 和 models/vae。原生 CLIPLoader 使用其受支持的 stable_diffusion 类型及权重识别，不添加不受支持的 anima 枚举。

## Foreground models / 前景模型

ISNet uses centered padding and is the lighter default. ToonOut is a BiRefNet model fine-tuned for anime content; its ONNX export uses ImageNet normalization with direct square resizing. Its FP16-weight export retains FP32 input/output and is approximately 492 MB. Ensemble inference executes both models sequentially and fuses probabilities.

ISNet 采用居中补边，是较轻的默认选项。ToonOut 为针对二次元微调的 BiRefNet，ONNX 导出采用 ImageNet 归一化和直接方形缩放，FP16 权重导出仍使用 FP32 输入输出，约492 MB。集成模式依次运行两个模型并融合概率。

ISNet source: https://github.com/SkyTNT/anime-segmentation

ToonOut source: https://github.com/MatteoKartoon/BiRefNet

ONNX export: https://huggingface.co/sprited/birefnet-toonout-onnx

## SAM2 checkpoints / SAM2权重

The automatic catalogue contains the official SAM2.1 tiny checkpoint. Other official SAM2.1 checkpoints can be installed manually in models/sam2; filenames are mapped to their matching official architecture configuration. Do not rename an unrelated checkpoint.

自动目录包含官方 SAM2.1 tiny。其他官方 SAM2.1 权重可手动放到 models/sam2，节点依据文件名选择匹配的官方架构配置，不能重命名无关权重冒充。

Source: https://github.com/facebookresearch/sam2

## Offline installation / 离线安装

Install the exact files at the listed paths, or use:

```bash
python tools/download_models.py --models-root "path/to/ComfyUI/models" --model toonout
```

Existing files with mismatched hashes are not overwritten. Downloaded partial files are removed on failure. Model source licenses remain independent from the plugin code license. / 已有文件校验不符时不覆盖，下载失败时移除临时文件；模型许可独立于插件代码许可。
## SAM3 concept segmentation / SAM3 概念分割

Download source / 下载来源：[bodhicitta/sam3](https://huggingface.co/bodhicitta/sam3), fixed revision `cba430d22f6fdc3f06ad3841274ec7bb55885f2f`. Required weight and configuration checksums match the upstream Meta export. / 所需权重及配置校验值与 Meta 上游导出一致。

Install the Transformers image-model bundle under `ComfyUI/models/sam3/facebook-sam3`. The weight file `model.safetensors` is 3,439,938,512 bytes (about 3.44 GB), SHA256 `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`. It requires `config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json` and `merges.txt` from the same revision. The node verifies the weight SHA256 and configuration Git blob hashes recorded in `model_store.py`. It does not load the separate native `sam3.pt` checkpoint.

Transformers 图像模型完整目录放入 `ComfyUI/models/sam3/facebook-sam3`。`model.safetensors` 约 3.44 GB，另需上述同修订配置与分词文件。节点验证权重 SHA256 及配置文件 Git blob 哈希，不使用原生实现的 `sam3.pt`。

Automatic download uses the public mirror anonymously (`token=False`); Hugging Face login and `HF_TOKEN` are not required. Model Download also accepts the `sam3` preset. Offline use requires the complete verified Transformers directory. Compatible files already present in the local directory are reused.

自动下载以匿名方式访问公开镜像，无需 Hugging Face 登录或 `HF_TOKEN`。“模型下载”节点也提供 `sam3` 预设。离线使用需要完整且校验一致的 Transformers 目录；本地已有匹配文件时直接复用。

SAM3 uses its own text encoder through Transformers, independently of Anima's Qwen encoder. Install the optional `requirements_sam3.txt` dependencies only in an environment whose ComfyUI/extensions support Transformers 5.x; the node imports them lazily. Model weights remain subject to the Meta SAM License: [license](https://huggingface.co/facebook/sam3/blob/main/LICENSE).

SAM3 的概念文本编码器与 Anima 的 Qwen 编码器相互独立。可选依赖通过 `requirements_sam3.txt` 安装，需要所在 ComfyUI 与扩展环境兼容 Transformers 5.x；节点按需导入。权重受 Meta SAM License 约束。
