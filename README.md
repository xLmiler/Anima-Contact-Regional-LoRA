# Anima Contact Regional LoRA

Spatial identity conditioning, local LoRA routing and structural control for Anima in ComfyUI.

[简体中文](README.zh-CN.md) · [Algorithms](docs/ALGORITHMS.md) · [Models](docs/MODELS.md) · [Node reference](docs/NODES.md)

The plugin assigns separate text conditioning and optional MODEL adapters to visible ownership regions. A shared scene baseline preserves composition and interactions. Structural LLLite control guides geometry independently of identity appearance. No model training or modification of ComfyUI core files is required.

## Capabilities

- Separate shared scene, global style and per-identity conditioning.
- Multiple local LoRA or supported linear LoKr adapters per identity, with independent scales and switches. Empty stacks use base-model conditioning.
- Anime person detection, SAM2 and SAM3 instance segmentation, independent ISNet/ToonOut foreground segmentation and mask fusion.
- Persistent ownership editing with identity colors, labels and thumbnails; incompatible strokes clear automatically after the reference changes.
- Identity-chain synchronization when nodes are removed, bypassed or muted. Unbound regions retain the base model and shared scene.
- Scheduled Anima LLLite control and generated-image Tile refinement with pixel resizing and VAE re-encoding.
- English and Simplified Chinese node definitions, parameter labels and tooltips using ComfyUI's localization mechanism.

## Installation

Place the enclosing `Anima-Contact-Regional-LoRA` directory in `ComfyUI/custom_nodes`, then install `requirements.txt` with the Python environment used by ComfyUI. Restart ComfyUI and refresh the browser cache. Keep a single installed copy of the plugin.

The Anima model, its compatible Qwen text encoder and its VAE are loaded through native ComfyUI nodes. Do not replace them with unrelated text encoders or VAEs. SAM2 segmentation additionally requires the official [SAM2 package](https://github.com/facebookresearch/sam2); foreground-only and palette-map paths do not require SAM2. SAM3 uses the optional `requirements_sam3.txt` dependencies and its own model directory; see [MODELS.md](docs/MODELS.md#sam3-concept-segmentation--sam3-概念分割).

At startup, the plugin creates missing directories for its detectors, segmentation models, SAM2 checkpoints, model patches and control weights. It does not download all weights at startup. A supported loader downloads a missing catalogued model when executed with **Automatic Download** enabled, verifies SHA256 and installs it atomically. Existing files with an unexpected checksum are not overwritten.

## Workflow

For a real reference image:

1. Load an image and obtain person boxes with **Anima Person Detection**.
2. Optionally obtain a union foreground mask with **Anima Foreground Segmentation**, or supply a mask from another segmentation plugin.
3. Connect the image, boxes and optional foreground to **Anima Instance Segmentation**. Review identity ownership with **Anima Ownership Editor**.
4. Create **Anima Scene Conditioning** and one **Anima Identity** per active ownership ID. Chain the identities into **Anima Regional Apply**.
5. Generate a structural representation with **Anima Layout Control Image**, load matching weights with **Anima LLLite Loader**, then apply control to the regional pipeline.
6. Sample with native ComfyUI nodes. For a subsequent refinement stage, connect the generated latent, its pipeline, a compatible VAE and Tile weights to **Anima Tile Upscale**.

Region IDs link masks to conditioning branches. Display names and LoRA filenames do not need to match. Foreground segmentation predicts the union of subjects; it does not identify which person owns an overlapping limb. Correct those assignments with explicit seeds or ownership strokes.

## SAM3 concept segmentation

**Anima SAM3 Semantic Segmentation** accepts one RGB image. Automatic mode probes `person`, `girl` and `boy`, then suppresses duplicate masks across queries. Custom mode accepts one concise English concept phrase per line. Each concept may match multiple instances. Automatic mode is a fixed concept search, not a captioning model.

The node retains SAM3 instance silhouettes and resolves overlapping pixels by confidence. It does not rerun SAM2 or cut limbs by left/right distance. Instances are ordered by mask centroid; inspect the color legend before assigning identity conditioning. A positive expected count verifies the result instead of truncating or inventing people.

Install `requirements_sam3.txt` with the ComfyUI Python environment. **Automatic Download** uses the public [bodhicitta/sam3](https://huggingface.co/bodhicitta/sam3) repository without a login or token. Weight and configuration checksums match the upstream Meta export. CPU and CUDA inference use Transformers; the model is released after inference. [Model details](docs/MODELS.md) include the pinned revision, directory and disk size. Masks still require visual review, especially with occlusion or small subjects.

## Examples

The `example_workflows` directory contains seven focused workflows. All identities start with empty adapter stacks and instructions are in Chinese.

| Workflow | Segmentation route |
|---|---|
| Two people, SAM3 | SAM3 concept instances |
| Two people, SAM2 | Anime detector and SAM2 |
| Two people, foreground | Foreground model, explicit identity seeds and geodesic ownership; no SAM model |
| Three, five and seven people | One SAM3 workflow for each count |
| Two people, Tile refinement | SAM3 followed by generated-image Tile refinement |

Load a real subject reference into the native **Load Image** node. Each workflow connects only its chosen segmentation route. For the foreground workflow, execute the identity-seed node to load its canvas, annotate each person and overlapping limbs, then queue the complete workflow. Union foreground alone does not identify different people. Latent dimensions follow the edited ownership layout. No personal filenames, preselected LoRA files or embedded previews are included.

`examples/reference_guides` contains reference, identity-map and outline PNGs generated by Python. Rebuild the workflows and these assets with:

```bash
python tools/build_examples.py
```

These guides are geometric demonstrations of structural input, not artwork, segmentation benchmarks or generated-image quality comparisons.

## Node groups

| Group | Nodes |
|---|---|
| Segmentation | Person Detection, Instance Segmentation, SAM3 Semantic Segmentation, Foreground Segmentation, Foreground Ownership, Seed Segmentation |
| Masks | Region Canvas, Ownership Editor, Identity Map Import, Identity Seeds, Mask Combine |
| Conditioning | Scene Conditioning, Identity |
| Routing | Regional Apply |
| Control | Layout Control Image, LLLite Apply |
| Models | LLLite Loader, Model Download |
| Sampling | Tile Upscale |
| Diagnostics | Regional Inspection |

See [NODES.md](docs/NODES.md) for stable node identifiers and parameter documentation.

## Algorithms and model files

Person boxes are produced by a DeepGHS anime person detector. SAM2 predicts instance candidates from boxes or identity seeds. Foreground probability maps can supplement missed pixels before edge-aware ownership allocation. Small disconnected components, enclosed holes and mutually exclusive label boundaries are cleaned before protected manual strokes are applied.

The routing engine computes regional text residuals against a shared conditioning baseline and adapter residuals against the corresponding unadapted branch. Native weighted modules retain their identity; temporary hooks are scoped to a model call and restored afterward. Optional self-attention isolation suppresses communication between different owned regions in selected transformer layers while keeping background connectivity. It is experimental and disabled by default.

Supported control presets are Anima outline LLLite, experimental regional-color LLLite and TileRepair. Tile refinement uses the first generated RGB image as its condition, not the reference image or the colored identity map. Complete model locations, fixed download revisions, checksums and upstream licenses are documented in [MODELS.md](docs/MODELS.md).

## Limits and diagnostics

The routing/control integration targets the native 28-block Anima Base architecture. It does not make incompatible architectures or arbitrary global adapter targets spatially local. Adapter projection matching is strict; a missing file or a zero-match adapter fails instead of silently falling back.

More people divide a finite image into smaller visible regions. Missing mask coverage, downsampled ownership gates and shared attention can weaken local appearance even when nominal adapter strength is unchanged. Structural control does not replace semantic identity conditioning. Replacing the VAE cannot resolve identity mixing that has already occurred during denoising. See [ALGORITHMS.md](docs/ALGORITHMS.md) for the evidence and limits of the isolation mechanism.

Use **Anima Regional Inspection** to inspect actual gates and adapter activity after sampling. CPU mathematical tests, browser checks and actual foreground-model integration checks are recorded in [VALIDATION.md](docs/VALIDATION.md); their scope does not imply validation on every ComfyUI distribution or GPU.

## Compatibility and development

Menu-facing nodes use functional identifiers. Executable compatibility aliases preserve all plugin interfaces used by the five supplied legacy example graphs. [Migration guidance](docs/MIGRATION.md) explains the interface boundary and supported migration tool.

Documentation is maintained incrementally. Interface changes must update the relevant README sections, both localization files and node schemas together; release chronology is kept in [CHANGELOG.md](CHANGELOG.md).

## License and attribution

Code is distributed under GPL-3.0-only; retained original MIT notices are included. Model weights are downloaded separately and remain subject to their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE](LICENSE).
