# Validation / 验证范围

Checks were run on Linux with Python 3.12.14, PyTorch 2.9.1+cpu, ONNX Runtime 1.31.0, NumPy 2.3.5, Pillow 12.3.0 and Chromium 131.0.6778.204. No complete ComfyUI distribution, Anima generation weights or user identity adapters were available in this environment. These checks establish mathematical and interface behavior within the stated scope; they do not establish image quality or full installation compatibility.

检查环境为 Linux、CPU PyTorch 和真实无头 Chromium。环境中没有完整 ComfyUI 发行版、Anima 生成权重或用户身份适配器。以下结果用于验证限定范围内的数学行为和接口，不代表生成质量评测或完整安装兼容性验证。

| Check / 检查 | Result / 结果 | Scope / 范围 |
|---|---|---|
| Python regression suite | 75 passed | Regional residuals, CFG handling, adapter projections, native module identity, scoped cleanup, cross-stage state, empty adapters, seven-identity gate magnitude, public schemas, bilingual definitions, download lifecycle and isolation mathematics |
| Browser regression | 41 passed | Real Chromium with a simulated ComfyUI host: named widget restoration, persistent editing, stale-reference reset, workflow switching, Map links, identity deletion/muting/bypassing, LoRA groups and live language changes |
| ToonOut ONNX inference, retained record | Passed previously | Earlier standalone CPU integration record; the new foreground-only route is not covered by this record |
| Independent foreground ensemble, retained record | Passed previously | Earlier standalone ISNet/ToonOut fusion record; not rerun as part of the blocked foreground-route test |
| Workflow migration | 5 passed | Prior workflows migrated in memory; only counts and status were retained in the report |
| SAM3 Transformers CPU contract | Passed | Real processor and reduced random-weight Sam3Model: save/load, two concept queries, reused image embeddings and official postprocessing; no trained-weight accuracy claim |
| Legacy example contracts | 5 passed | All 12 plugin node IDs, original socket types and output positions remain registered and connected |
| SAM3 pretrained inference and anonymous download | Passed | Complete trained weights, eight verified files, two instances on a real anime reference; no ground-truth accuracy claim |
| SAM3 automatic semantic route | Passed | Public node with the example defaults: person/girl/boy queries, two instances after deduplication, full trained weights on CPU |
| Foreground-only route | Logic passed; ONNX test blocked | Synthetic foreground/seed tests verify geodesic ownership and no SAM calls. The real ONNX execution was blocked by automatic approval due to an unexplained Microsoft telemetry connection. |
| Focused example workflows | 7 verified | Official native IDs including ModelSamplingAuraFlow; separate dual-person routes and no alternative segmentation branches |
| Procedural reference assets | 12 verified | PNG integrity and ownership geometry for two, three, five and seven identities |

The complete trained SAM3 bundle was downloaded anonymously from bodhicitta/sam3 and all eight required files passed integrity verification. A real CPU inference with the `girl` concept on the supplied demo reference crop returned two instances. The left subject's cross-body arm remained in its instance mask during visual inspection. The public node's automatic semantic route also returned two instances after deduplicating person/girl/boy queries with the example defaults. No annotated ground truth was available; confidence scores are not segmentation accuracy. See `tests/sam3_download.json`, `tests/sam3_real_cpu.json` and `tests/sam3_default_route_cpu.json`.

完整 SAM3 训练权重已从公开镜像匿名下载，八个必需文件通过校验。真实 CPU 推理在提供的演示参考裁切中得到两个实例；目视检查时，左侧人物跨身手臂保留在其掩码内。公开节点使用示例默认的自动语义设置执行 person/girl/boy 查询，去重后也得到两个实例，CPU 耗时 56.5 秒。没有标注真值，不报告分割准确率，模型置信度也不等于准确率。纯前景路线的合成遮罩、身份种子和测地归属测试通过。真实 ONNX 检查被自动审批以无法确认载荷的 Microsoft 遥测连接为由拦截，即使已在创建会话前调用官方禁用遥测接口；该次运行结果不计为验证。见 `tests/foreground_route_cpu.json`。

Detailed records are in `tests/python_report.txt`, `tests/browser_report.json`, `tests/real_toonout_cpu.json`, `tests/real_foreground_ensemble_cpu.json` and `tests/migration_report.json`. The foreground checks used a screenshot crop as input without retaining it in the distribution. No annotated ground-truth masks were available, so no accuracy score is reported.

详细记录位于上述测试文件。前景模型接口检查使用过截图裁切，发行包不保留该图片。由于没有标注真值，不报告分割准确率。浏览器测试模拟了宿主接口，没有在秋叶整合包的完整前端中运行。

## Reproduction / 复现

Install the project requirements and a compatible PyTorch environment, then run:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

For the browser suite, install Playwright and its Chromium browser in a development environment:

```bash
npm install --no-save playwright
npx playwright install chromium
node tests/test_ui.cjs
```

`ANIMA_TEST_BROWSER` can select an existing Chromium executable. Browser tests use local HTTP fixtures and a simulated model list; they do not require model weights or logins. Run `python tests/smoke_sam3_cpu.py` with the optional SAM3 requirements to reproduce the reduced random-weight Transformers CPU contract. It does not download trained weights.

Real model inference additionally requires the model files listed in [MODELS.md](MODELS.md). Large weight files are not included in the distribution.

## Pending integration evidence / 尚缺的集成证据

- Full ComfyUI and the user's Aki distribution with their installed frontend and extensions.
- GPU sampling with actual Anima, LLLite and identity adapters, including VRAM and runtime measurements.
- Controlled six- and seven-person image comparisons for recognition, appearance leakage and interaction quality with optional self-attention isolation enabled and disabled.
- Annotated instance and union masks for quantitative foreground and ownership evaluation.

Self-attention isolation remains disabled by default. It suppresses direct cross-identity attention numerically, but its visual benefit and performance cost must be measured with real generation workloads.
