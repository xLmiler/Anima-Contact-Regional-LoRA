# Algorithms and engineering limits

## Ownership segmentation

DeepGHS YOLOv8 ONNX detection establishes person boxes and a deterministic horizontal ordering. SAM2 supplies per-instance proposals. Manual positive and negative seeds can constrain proposals. An optional union foreground map supplements pixels missed by SAM2 within detected supports; it does not invent a new instance identity. Overlap allocation uses foreground connectivity, image/depth-proxy edges, instance priors and distance from reliable ownership seeds.

ISNet uses centered aspect-preserving padding with RGB normalization. ToonOut uses direct 1024-square resizing and ImageNet normalization, matching its ONNX export contract. The implementations must not share preprocessing. Consensus fusion is the geometric mean of aligned probabilities, coverage fusion is their maximum, and intersection fusion is their minimum. Consensus can discard a true foreground region missed by one model; coverage can admit false foreground. Neither is a proof of correct ownership.

Component cleanup preserves each identity's largest connected component. Small enclosed background cavities are filled only with sufficiently consistent surrounding ownership. Labels compete mutually exclusively during smoothing. Manual ownership strokes are applied after cleanup and are bound to their source layout.

Photoshop's Select Subject is a proprietary AI-assisted selection facility. Adobe distinguishes device/cloud processing and recommends object selection for individual targets in multi-subject images. Its public documentation does not provide weights or an inference algorithm that this plugin can reproduce exactly. Foreground extraction and per-person ownership remain different tasks.

## SAM3 semantic instances

SAM3 performs promptable concept segmentation through the Transformers image model and processor. One image embedding is reused across concept queries. Automatic mode probes a documented fixed set of person concepts; custom mode uses one text phrase per line. It is not a vision-language captioning pipeline.

The official postprocessor applies detection/presence confidence and mask thresholds. Cross-query duplicate suppression uses mask IoU. The retained silhouettes are not passed through SAM2 or distance-based repartitioning; only genuinely overlapping pixels compete by instance confidence. Spatial centroid ordering defines char IDs, while explicit expected counts validate the result. A mask or a named concept is not proof of semantic identity or limb ownership under every occlusion.

The pinned public bodhicitta/sam3 mirror supplies the Transformers bundle anonymously. Every required weight and configuration file is checked against the upstream-matching hash catalogue; local compatible files are reused.

## Foreground-only ownership

Foreground extraction predicts a union probability map, without assigning person identities. Explicit identity seeds and optional background exclusions constrain a geodesic allocation using foreground-probability and image-intensity edges. The result is an exclusive ownership layout, followed by the same persistent editor used by the other routes. This route loads no SAM or person detector. An empty foreground is rejected instead of presenting isolated seed dots as completed person masks. Incorrect or insufficient seeds can still assign an overlapping limb to the wrong identity.

## Regional conditioning and adapters

Each identity has its own encoded shared-scene-plus-identity context. Contexts are not concatenated into one growing prompt across all people. The cross-attention result is composed from a shared baseline and spatially gated identity residuals. Global style can remain in the shared baseline while identity residuals use an unstyled shared-scene reference. Negative CFG rows retain their negative text.

Supported local MODEL adapters are added to compatible projection targets through scoped hooks. Self-attention adapter branches can be recomputed independently and blended only at owned output tokens. Global time modulation or text-encoder adapters cannot generally be made spatially local by assigning an image mask; they are not treated as equivalent local adapters.

The engine does not divide each adapter's strength by the number of identities. Partition gates preserve covered-token magnitude. This numerical property does not ensure equal visual recognition when the number of people increases.

## Why dense scenes lose identity

A finite canvas gives smaller people fewer pixels and latent image tokens. Thin features and partially covered edge tokens can lose effective ownership after resampling. Incorrect foreground or overlapping limbs route the wrong features. Shared image self-attention can propagate appearance already introduced by earlier cross-attention or adapters into other regions. Unsupported projection families and conflicting source/target silhouettes introduce further differences from a single-person global LoRA baseline.

These are engineering mechanisms supported by the code and numerical tests, not a diagnosis of an unavailable set of user weights. There is no hardcoded six-person threshold. Real recognition quality depends on weights, masks, visible area, prompts, sampling and model architecture.

Changing a compatible VAE changes reconstruction, not the semantic conditioning already used during diffusion. Increasing text weights or replacing the Qwen encoder with an unrelated “smarter CLIP” is not an architecture-preserving identity fix. Structural LLLite residuals constrain shape but do not give a region a particular semantic identity by themselves.

## Optional self-attention isolation

The native Anima `attn_op` is temporarily wrapped after a selected transformer depth. Attention logits between two different owned IDs receive an additive `log(1-strength)` penalty; strength one uses a hard exclusion. Same-identity and background connections are retained. Queries are processed in chunks to avoid allocating a full token-by-token bias matrix.

This applies to the shared attention computation and corresponding adapter branches consistently. Weighted modules are not replaced, and the original operation is restored in a `finally` block. The native tensor and `AttentionTensorContainer.take()` interfaces are both supported. An unsupported attention interface raises an explicit error when isolation is enabled.

The strength defaults to zero. Numerical tests show reduced direct cross-region value leakage and equivalence at zero strength. They do not establish a GPU image-quality improvement for six or seven people. Excessive isolation can reduce interactions and cross-body coherence. It does not restore missing pixels or eliminate indirect propagation through shared background tokens.

## Structural and refinement control

Outline controls are derived from ownership boundaries; regional-color controls require their corresponding experimental model. Effect masks and release masks control structural residuals independently of identity gates. Control timing follows the model's full diffusion schedule.

Tile refinement decodes generated content, resizes pixels, re-encodes latents and conditions TileRepair on that generated RGB. It rebuilds regional routing from the original source model and does not stack a second copy of the identity adapters. Pixel-perfect preservation is not implied.

## Sources

- [ComfyUI native Anima implementation](https://github.com/Comfy-Org/ComfyUI/tree/master/comfy/ldm/anima)
- [Native Cosmos Predict2 attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py)
- [SAM3 official implementation](https://github.com/facebookresearch/sam3)
- [SAM3 Transformers interface](https://huggingface.co/docs/transformers/en/model_doc/sam3)
- [SAM2](https://github.com/facebookresearch/sam2)
- [SkyTNT Anime Segmentation](https://github.com/SkyTNT/anime-segmentation)
- [ToonOut paper](https://arxiv.org/abs/2509.06839)
- [ToonOut source](https://github.com/MatteoKartoon/BiRefNet)
- [ONNX exporter contract](https://huggingface.co/sprited/birefnet-toonout-onnx)
- [Anima Regional Conditioning: score-level spatial attention](https://github.com/Sen-sou/Comfyui-Anima-Regional-Conditioning)
- [Anima Regional: shared baseline and identity focus](https://github.com/An1X3R/ComfyUI-Anima-Regional)
- [Krea2 Regional: distinct joint-attention architecture](https://github.com/januspluto/ComfyUI-Krea2-Regional)
- [Adobe Select Subject](https://helpx.adobe.com/photoshop/desktop/quick-actions/select-subject-in-images.html)

External projects informed comparison and design. Their complete patch implementations are not installed or silently combined with this plugin.
