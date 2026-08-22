# MiniMax-H3 on Apple M3 Pro

Status: the real-weight C/Metal path covers the tokenizer, streamed Q8 Qwen
conditioner, affine-Q4 H3 transformer, BF16 residual stream, Video VAE, Audio
VAE and MP4 mux. The exact-attention 864×480×124 Turbo-4 path completes in
2,194.437 seconds with zero swaps, using a single-pass online-softmax
attention kernel — the same mathematics as the earlier two-pass kernel with
fp32 accumulation and at most one bf16 unit of rounding difference on 0.19
percent of outputs. Three faster attention variants are rejected on
frame-level review: the compiled tree (1,489.401 s) softens detail and ghosts
transitions; the frame-safe late-layer candidate (2,344.734 s) streaks the
highest-motion close-up; and a bf16-probability variant of the single-pass
kernel (2,051.492 s) duplicates line work in the shot-4 close-up. Attention
quality gates now include per-frame review of the highest-motion shot.
Official full-precision parity and the speed target remain open.

## Prompt budget

Line-work quality depends on the prompt respecting the model's shot
budget. Published MiniMax H3 guidance calls for one committed visual
style, action-first subject description, and one to two shots per six
seconds with at least 1.5 seconds per shot. A four-shot prompt packed
into 5.2 seconds renders duplicated, misaligned line work in every
attention path, including exact; the same runtime renders a
guide-conformant single-shot prompt with clean single-drawn lines
(`classroom-single-shot-prompt-gate` in [`results.json`](results.json)).
Quality gates therefore review prompts against this budget before
attributing artifacts to the runtime, and attention changes are
reviewed per-frame on the highest-motion shot.

## 360p Turbo-4 optimization round (2026-08-22)

A fixed-seed `640x352x124` text-to-video run exposed the same two dominant
boundaries as 480p: H3 denoise and Video VAE decode.  The initial native run
took 999.590614 seconds: 768.833793 seconds in denoise and 199.300129 seconds
in Video VAE.  A combined performance experiment took 783.832033 seconds,
but it is rejected: user visual review found a large composition regression,
including the subject's face and head being cropped from shots.  The release
default therefore retains the previous SIMD-reduction LoRA path while keeping
the bit-identical command-buffer pipeline and the quality-gated Video VAE tile.

| Change | Component A/B | Full-run effect | Decision and correctness boundary |
|---|---:|---:|---|
| BF16 simdgroup-matrix Turbo LoRA down/up | 128x128x124 denoise: 92.476261 → 52.163499 s (1.773x) | experimental 360p denoise, combined with submission pipelining: 768.833793 → 595.291946 s | **Rejected:** changed reduction order causes trajectory and composition regression; experimental opt-in only |
| Thirteen precommitted four-layer command buffers | 128x128x124 denoise: 93.332582 → 92.350155 s | included in the 595.291946-second denoise result | serial queue and hidden-state dependencies are retained; control media hashes are identical |
| Geometry-specific 256x160 Video VAE tiles at 352-pixel height | deterministic VAE: 198.378460 → 159.409562 s; peak 550.3 → 436.0 MiB | 199.300129 → 159.662679 s | required 64-pixel overlap retained; 124-frame deterministic-latent PSNR is 36.939526 dB and sampled seams pass visual review |

The release default uses the scalar SIMD-reduction adapter path.  The rejected
matrix experiment can be reproduced only with `MINIMAX_H3_LORA_MMA=1`, and
only when adapter dimensions are multiples of its 64-output/8-input tile.
`MINIMAX_H3_PIPELINE_LAYER_GROUPS=0` restores
synchronous four-layer submission.  `MINIMAX_H3_VAE_TILE_HEIGHT` and
`MINIMAX_H3_VAE_TILE_WIDTH` override the geometry default.  Release validation
checks the hidden state once per denoise step; setting
`MINIMAX_H3_VALIDATE_LAYER_GROUPS=1` restores validation after every four
layers.  `MINIMAX_H3_DENOISE_ONLY=1` is a benchmark-only boundary that stops
before both VAEs and mux.

### Rejected experiments and reusable lessons

* A 64-row Q4 activation batch changed no media bits but was slower:
  94.436593 versus 93.332582 seconds at 128x128x124.  Q4 dequantization,
  barriers and register pressure already dominate enough that doubling the
  activation tile does not improve this M3 Pro kernel.
* Reducing finite scans from 52 per run to four is the right release boundary,
  but it is not a performance optimization: 93.851104 versus 93.332582
  seconds is measurement noise.  Keep the change for validation cost and
  semantics, not as a speed claim.
* Staging a 64x32 BF16 LoRA weight tile in threadgroup memory and expanding the
  batch tile from 32 to 64 did not beat direct MMA: 52.254701 versus
  52.163499 seconds.  The adapter already benefits from device/L2 cache and
  the extra barriers cancel the saved reads.
* At 352 pixels, two 256-pixel vertical tiles overlap by 160 pixels.  Three
  160-pixel tiles use the required 64-pixel overlap, reduce redundant token
  work and make each quadratic attention task smaller.  Tile count alone is
  therefore a misleading cost metric; overlap area and per-tile attention
  rows must be evaluated together.
* BF16 MMA changes FP32 summation order.  Its same-seed encoded-video PSNR
  versus the prior reduction order is only 16.056232 dB.  User review found a
  large visual regression: composition drifted and the subject's head and face
  were cropped from shots.  The experiment failed the quality gate and is not
  the release default.
* Full resolution changes the bottleneck mix.  LoRA MMA improves the isolated
  128p denoise by 43.59 percent but complete 360p denoise by 22.57 percent,
  because exact dense attention consumes a larger share at 8,642 rows.

The exact prompt, baseline, optimized breakdown and experiment records are in
[`artifacts/walking-woman-640x352-turbo4/benchmark.json`](artifacts/walking-woman-640x352-turbo4/benchmark.json).

## N-to-N optimization timeline

| Runtime milestone | Affected stage | Affected-stage time before → after | 480p N-to-N time and change | Reason | Evidence |
|---|---|---:|---:|---|---|
| Correct seven-chunk baseline | Video VAE | **7,387.292038 s measured** | **9,294.869743 s**; 1.000× | Establish the released seven-chunk temporal graph; this is the valid starting point, not an optimization | measured complete N-to-N |
| Whole-image buffers + static bindings + RoPE + four-layer command grouping | Video VAE setup/execution; H3 resource binding | VAE: 7,387.292038 measured → 7,205.000040 s projected (**−182.291998 s**); H3 32×32 smoke: 17.122752 → 7.571732 s measured (**−9.551020 s**), excluded from the 480p projection | 9,112.577745 s; −182.291998; 1.020× | Reuse fixed Metal images, 438 VAE bindings and RoPE; encode four VAE layers per command buffer instead of recreating or synchronizing per use | VAE projected from 105 × 68.619048 s; H3 smoke measured separately; exact setup subchanges were not isolated |
| + 32-row simdgroup-matrix GEMM | Video VAE dense projections and MLP | 7,205.000040 → 716.561895 s projected (**−6,488.438145 s**) | 2,624.139600 s; −6,488.438145; 3.542× | Map scalar dense linear work to M3 simdgroup-matrix hardware with FP32 accumulation | projected from 105 × 6.824399 s |
| + eight-query exact tiled attention | Video VAE self-attention | 716.561895 → 659.524425 s projected (**−57.037470 s**) | 2,567.102130 s; −57.037470; 3.621× | Eight queries reuse 32 K/V rows in threadgroup memory while preserving the original key order | projected from 105 × 6.281185 s |
| + 32-row shared-weight GEMM | Video VAE dense projections and MLP | 659.524425 → 637.239960 s projected (**−22.284465 s**) | 2,544.817665 s; −22.284465; 3.652× | Share staged weights across neighboring activation rows to reduce repeated weight traffic | projected from 105 × 6.068952 s |
| + 64-row shared-weight GEMM | Video VAE dense projections and MLP | 637.239960 → 528.169950 s projected (**−109.070010 s**) | 2,435.747655 s; −109.070010; 3.816× | Amortize every staged tile across twice as many activation rows | projected from 105 × 5.030190 s |
| Full current N-to-N validation | Full graph measurement; no new optimization | VAE: 528.169950 projected → **487.273811 s measured** (−40.896139); all other stages: 1,907.577705 → 1,931.434426 s (+23.856721) | **2,418.708237 s**; −17.039418 vs projection; **3.843×** | Replace single-task linear scaling with a complete run; the net projection difference is not claimed as an optimization | measured complete N-to-N |
| Exact BF16 direct output + four-layer H3 command groups | H3 dense-attention storage and command submission | Attention: 5,024.974042 → 5,012.634167 ms/call; FP32 scratch: 448,401,408 → 0 B; 128² denoise: 17.407398 → 17.147884 s | full 480p N-to-N not remeasured; no projection claimed | Reuse the kernel's dead score-tile region as an FP32 fragment spill, emit BF16 directly and submit four layers per command buffer | all 112,100,352 irregular-input BF16 outputs and smoke media hashes identical |
| Experimental compiled tree attention | H3 dense attention | **1,898.120497 → 974.544700 s measured**; −923.575797 s; 1.948× | **1,489.401301 s**; −929.306936; **1.624×** vs exact; **6.241×** vs baseline | Compile geometry, row order, tree nodes, routes and query blocks once; rebuild only K/V summaries on Metal | measured complete N-to-N; functional pass, visual regression; opt-in only |
| Frame-safe late-layer attention | H3 attention in step 4, layers 40–49 only | same-run three late groups **92.597490 → 70.313368 s**; −22.284122 s; 1.317× | **2,344.734228 s**; −73.974009 s; **1.032×** vs earlier exact run | Keep conditioning exact; never merge K/V across frames; exact-anchor 190/200 attention calls | **rejected**: per-frame review of the violent-motion close-up (frame 45) shows vertical streak ghosting and eye-edge echoes that the whole-video luma gate averaged away |
| Single-pass online-softmax exact attention | H3 dense attention, all 200 calls | kernel **5,007 → 3,753 ms/call** (~1.33×): one QK pass instead of two, transposed key staging, one exponential per score, fp32 probabilities throughout | **2,194.437 s**; −224.271 s; **1.102×** vs prior exact; **4.236×** vs baseline | Same mathematics, fp32 accumulation; one bf16 ulp difference on 0.19% of kernel outputs | measured complete N-to-N; scene cuts identical (36/61/93), adjacent-luma 2.714 vs 2.668, per-frame review clean; a bf16-probability variant (2,051.492 s) was rejected after user review found duplicated line work in the shot-4 close-up |

### Latest measured run breakdown

| Stage | Seconds | Runtime share | Boundary |
|---|---:|---:|---|
| Tokenizer | 0.007698 | <0.001% | UTF-8 prompt → 240 token IDs |
| Metal setup | 0.043712 | 0.002% | offline metallib and pipelines |
| Text image access | 4.693886 | 0.194% | complete local 28.22 GB safetensors image |
| 50-layer text conditioner | 4.567422 | 0.189% | streamed affine-Q8 layers |
| Turbo AdaLN compile | 0.868117 | 0.036% | four fixed evaluations |
| H3 RoPE precompute | 0.000996 | <0.001% | fixed 15,639-row geometry |
| **H3 denoise** | **1,898.120497** | **78.477%** | 50 blocks × four dense evaluations |
| Video VAE precompute | 0.007911 | <0.001% | fixed tile/chunk geometry |
| **Video VAE decode** | **487.273811** | **20.146%** | 7 temporal chunks × 15 spatial tiles × 36 layers |
| Audio VAE decode | 13.760077 | 0.569% | 165,333 samples per channel |
| H.264/AAC mux | 0.671221 | 0.028% | synchronized MP4 |
| Other measured runtime | 8.692889 | 0.359% | allocation, Euler updates and stage transitions |
| **Model runtime** | **2,418.708237** | **100%** | tokenizer through original mux |
| Preflight hashes and shell overhead | 65.671763 | outside runtime | SHA-256 checks and shell work |
| **Command wall** | **2,484.380000** | runtime + preflight | outer `/usr/bin/time` wall |

The model-runtime rows add exactly to 2,418.708237 seconds. Post-run decode,
hash, media statistics and visual verification are excluded from both runtime
and command wall.

### Experimental compiled-tree run breakdown

| Stage | Seconds | Runtime share | Change from exact run |
|---|---:|---:|---:|
| Tokenizer | 0.008059 | 0.001% | +0.000361 s |
| Metal setup | 0.026800 | 0.002% | −0.016912 s |
| Text image access | 4.273334 | 0.287% | −0.420552 s |
| 50-layer text conditioner | 4.635396 | 0.311% | +0.067974 s |
| Turbo AdaLN compile | 0.877777 | 0.059% | +0.009660 s |
| H3 RoPE precompute | 0.000916 | <0.001% | −0.000080 s |
| **H3 denoise** | **974.544700** | **65.431%** | **−923.575797 s** |
| Video VAE precompute | 0.007417 | <0.001% | −0.000494 s |
| **Video VAE decode** | **483.809133** | **32.483%** | −3.464678 s |
| Audio VAE decode | 13.697438 | 0.920% | −0.062639 s |
| H.264/AAC mux | 0.648403 | 0.044% | −0.022818 s |
| Other measured runtime | 6.871928 | 0.461% | −1.820961 s |
| **Model runtime** | **1,489.401301** | **100%** | **−929.306936 s; 1.623947×** |
| Preflight hashes and shell overhead | 67.838699 | outside runtime | local SHA-256 checks |
| **Command wall** | **1,557.240000** | runtime + preflight | −927.140000 s |

The experimental run used 4,930,967,616 bytes (4.592 GiB) peak physical
footprint, 1,361,362,944 bytes maximum resident set and zero swaps. The extra
490,242,048 bytes over the exact run are the physical Q/K/V/output and tree
summary buffers. The stage timing passed; the quality gate did not. All 124
frames decode and no frame is flat, but the 0.25 scene detector reports 10
changes instead of the exact run's 3, adjacent-frame mean difference rises
from 4.494335 to 9.814484, and visual review finds softer detail and stronger
transition ghosting. The path therefore requires
`MINIMAX_H3_TREE_ATTENTION=conservative` and is not selected by default.

### Frame-safe late-layer run breakdown

| Stage | Seconds | Runtime share | Change from earlier exact run |
|---|---:|---:|---:|
| Tokenizer | 0.009389 | <0.001% | +0.001691 s |
| Metal setup | 0.040799 | 0.002% | −0.002913 s |
| Text image access | 4.308655 | 0.184% | −0.385231 s |
| 50-layer text conditioner | 4.639853 | 0.198% | +0.072431 s |
| Turbo AdaLN compile | 0.869813 | 0.037% | +0.001696 s |
| H3 RoPE precompute | 0.001027 | <0.001% | +0.000031 s |
| **H3 denoise** | **1,831.966125** | **78.131%** | **−66.154372 s** |
| Video VAE precompute | 0.007414 | <0.001% | −0.000497 s |
| **Video VAE decode** | **481.876037** | **20.551%** | −5.397774 s |
| Audio VAE decode | 13.649289 | 0.582% | −0.110788 s |
| H.264/AAC mux | 0.535732 | 0.023% | −0.135489 s |
| Other measured runtime | 6.830095 | 0.291% | −1.862794 s |
| **Model runtime** | **2,344.734228** | **100%** | **−73.974009 s; 1.031549×** |

This run used `MINIMAX_H3_TREE_ATTENTION=quality`: step mask `0x8` and
50-layer mask `0x3ff0000000000`. The first three Turbo evaluations and layers
0–39 of the last evaluation used the original exact BF16 kernel. Only the last
ten attention calls used the frame-safe route. Their adjacent same-run exact
groups took 92.597490 seconds; the routed groups took 70.313368 seconds, a
22.284122-second or 24.066% reduction. The larger 66.154372-second denoise
difference against the earlier full run includes run-to-run variation and is
not attributed entirely to the route.

| Quality gate | Exact | Frame-safe | Rejected temporal tree |
|---|---:|---:|---:|
| Scene changes at threshold 0.25 | 3: 1.541667, 2.583333, 3.916667 s | **same three times** | 10 |
| Adjacent-frame luma difference, mean | 4.495795 | **4.283097** | 9.814484 |
| Frame luma range, mean | 206.258065 | 215.596774 | 212.250000 |
| Blur mean | 5.686310 | 5.768716 | 5.749247 |
| Versus exact video | reference | PSNR 32.384649 dB; SSIM 0.932134 | PSNR 13.456034 dB; SSIM 0.659932 |
| Reviewed result | four aligned shots; no double exposure | **same cuts and composition; no reviewed transition ghosting** | softness, double exposure and extra cuts |

The [uniform contact sheet](artifacts/anime-room-864x480-turbo4/frame-safe-quality-contact-sheet.png)
places exact on the left and frame-safe on the right. The
[transition sheet](artifacts/anime-room-864x480-turbo4/frame-safe-quality-transitions.png)
places exact above frame-safe for five frames around each scene boundary.
This single-prompt ghosting gate passes; exact numeric parity, audio listening
and a multi-prompt quality suite remain open. Exact attention therefore remains
the default.

Intermediate N-to-N values are projections, not hidden full runs. Each uses
the valid baseline's measured 1,907.577705 seconds outside Video VAE plus 105
times the corresponding real-weight 256×256×22 VAE task. The first and last
rows are complete measurements. The earlier 7,249.519-second one-sequence run
is excluded because 86/124 output frames were flat; it is a failed correctness
run, not a faster baseline.

## Target

| Property | Pinned value |
|---|---|
| Machine | MacBook Pro `Mac15,6` |
| SoC | Apple M3 Pro |
| CPU | 11 cores, 5 performance + 6 efficiency |
| GPU | 14-core Metal 3 GPU |
| Unified memory | 36 GB LPDDR5 |
| Runtime contract | C API and graph control, Metal kernels, minimum Objective-C bridge required by Metal |
| Excluded runtime dependencies | Python, PyTorch, MLX, GGML, llama.cpp, ONNX Runtime, ComfyUI |

The target must leave operating-system headroom. The resident-image gate is
32 GiB. The optimized 480p run captured a 4,440,725,568-byte runtime peak
physical footprint, a 1,062,879,232-byte maximum resident set and zero swaps.

## Real-weight runtime boundary

| Stage | Implemented execution |
|---|---|
| Prompt | exact Qwen tokenizer compiled into the binary |
| Conditioner | 50 Qwen3-VL language layers, affine Q8 group 64, one layer resident at a time |
| H3 input/refiner | dense BF16/F32 projections and two refiner blocks |
| H3 transformer | 50 affine-Q4 group-64 blocks; BF16 activations/residuals and FP32 accumulation |
| Sampling | exact 30-interval Euler path, or Turbo v4-600 with four evaluations; video shift 12 and audio shift 3 |
| Video | real-weight staged Video VAE and H.264 frame encode |
| Audio | real-weight BigVGAN-style decoder to 32 kHz stereo WAV |
| Output | `ffmpeg` mux to H.264 + AAC MP4 |

The C process owns tokenizer, tensor loading, graph execution and sampling. Metal is accessed through the minimum Objective-C bridge. Python, PyTorch, MLX, GGML, llama.cpp, ONNX Runtime and ComfyUI are not linked or launched by the inference path. The deployment binary is compiled with `MINIMAX_H3_LOCAL_ONLY` and does not link libcurl; HTTP support exists only in offline checkpoint tools.

## Why the current execution artifact is derived

The official FP16 tensors are not a resident execution image for this 36 GB
target. The Qwen conditioner contains 66.715 GB of tensors and the H3
transformer contains 66.280 GB before either VAE is counted. Either heavy
stage alone exceeds the target's safe resident-memory budget.

The current bootstrap therefore uses the pinned third-party
`Sawfwair/MiniMax-H3-FL2VA-MLX-4bit` artifact. Its 28.223 GB affine-Q8
conditioner and approximately 11.333 GB group-64 affine-Q4 H3 cache can be
loaded in separate phases. This made it possible to validate the native
tokenizer-to-media path before implementing a complete official-FP16 importer
and quantizer.

This is an execution and performance bootstrap, not an official-weight parity
claim. The intended provenance path is to treat the official checkpoint as the
source of truth, compile target Q8/Q4 images with this repository's own offline
compiler, and gate them with layer-boundary error plus fixed-seed media-quality
tests. Until that path passes, the runtime is described as quantized,
mixed-precision and third-party-derived—not as the full-precision model.

## Corrected 480p N-to-N benchmark

Exact input, output hashes, benchmark scope and verification are in
[`REVIEW.html`](REVIEW.html) and [`results.json`](results.json). The preserved
[`benchmark.json`](artifacts/anime-room-864x480-turbo4/benchmark.json) and
[`verification.json`](artifacts/anime-room-864x480-turbo4/verification.json)
are the pre-optimization baseline.

| Field | Measured value |
|---|---:|
| Input | original 942-byte Chinese four-shot prompt; 240 tokens |
| Geometry | 864×480×124 at 24 fps; 5.166667 s video |
| Sampler | Turbo v4-600 EMA; four evaluations; seed 42 |
| Packed rows | 15,639 |
| Text image access / encode | 4.693886 / 4.567422 s |
| H3 denoise | 1,898.120497 s |
| Video VAE precompute / decode | 0.007911 / **487.273811 s** |
| Audio VAE decode / mux | 13.760077 / 0.671221 s |
| Complete runtime | **2,418.708237 s** |
| Pre-optimization runtime | 9,294.869743 s; current path is **3.842907× faster** |
| Runtime peak physical footprint | **4,440,725,568 bytes (4.136 GiB)** |
| Maximum resident set | 1,062,879,232 bytes (0.990 GiB) |
| Swaps | **0** |
| Output structure | 124/124 H.264 frames; 32 kHz stereo AAC; non-silent |
| Visual result | four prompt-aligned shots; 0 flat frames; no sampled spatial seam or temporal break |

H3 denoise is now 78.477% of runtime and Video VAE is 20.146%. Within the 200
H3 layer/evaluation calls, exact dense attention consumes 1,143.533 seconds,
projections 248.384 seconds and MLP 495.066 seconds. H3 attention and dense
projection/MLP work are now the next optimization boundaries.

### H3 attention round: precompiled data and Metal work

| Item | Compile/precompute boundary | Runtime Metal work | Result |
|---|---|---|---|
| Direct exact BF16 output | pipeline is built from the offline metallib; no graph-dependent table | MMA64 attention writes BF16 from FP32 fragments | 5,024.974042 → 5,012.634167 ms/call; 448,401,408 B scratch removed; bit identical |
| Four-layer command groups | fixed 50-layer schedule determines 13 groups | one encoder and synchronization per group instead of three per layer | 128² denoise 17.407398 → 17.147884 s; complete video/audio hashes identical |
| Row permutation | fixed text/audio/video geometry maps logical rows to tree-major physical rows once | three parallel BF16→BF16 reorder kernels per routed block | reused by the selected block/evaluation calls |
| Tree and routes | 342 nodes, 307 query blocks, 140,284 route entries and log token counts are compiled once | no route construction or graph traversal in the denoise loop | maximum route length 548 entries instead of 15,639 dense keys |
| K/V summaries | topology and reduction ranges are fixed | leaf, frame, temporal and root summaries are rebuilt for each dynamic layer | approximate attention remains responsive to the current hidden state |
| Tree attention | query block and route offsets are fixed | `simdgroup_matrix` QK tiles, online softmax and V accumulation on the M3 GPU | H3 denoise 1,898.120497 → 974.544700 s, with failed visual-quality gate |
| Conditioning route correction | conditioning route index is compiled after the video-leaf routes | text/audio queries attend every video row exactly | removes the prior accidental use of video route 0 for all conditioning queries |
| BF16/FP32 restoration | exact and approximate buffers share one BF16 execution image | BF16 Q/K/V with FP32 score, softmax and V accumulation | removes an unintended BF16→FP16 conversion and FP16 probability/PV accumulation from the experiment |
| Frame-safe routes | every latent frame retains separate leaf summaries; two spatial neighbors per frame are exact | no frame, temporal-group or root summary is dispatched | removes cross-frame value averaging, the direct double-exposure mechanism |
| Exact anchors | step/layer masks are emitted with the fixed four-step graph | original exact kernel handles 190/200 attention calls | measured step scan selected step 4; ten-layer scan selected layers 40–49 |

The compiler does not precompute values that depend on a denoise hidden state.
It precomputes the topology, storage layout and execution schedule, then emits
Metal kernels for the remaining reductions and attention. The current route is
geometry-only. The measured quality regression shows that the next version
must compile teacher-calibrated routes or multiple learned centroids rather
than merely increasing geometric locality. The quality profile is the first
bounded correction: it fixes conditioning semantics and arithmetic, removes
all cross-frame summaries, then uses exact step/layer anchors. It solves the
measured ghosting failure but exposes only a small safe fraction of the graph.

The next route compiler is constrained by current training-free sparse-video
work, not copied from one framework implementation:

| 2026 result | Constraint used here |
|---|---|
| [PISA](https://arxiv.org/abs/2602.01077) | rejected regions need approximation or correction, not keep/drop pruning; add first-order block compensation after the zero-order path is stable |
| [CalibAtt](https://arxiv.org/abs/2603.05503) | compile sparsity per layer, head and diffusion timestep from an offline teacher; the measured Turbo step/layer masks are the first local instance |
| [SVOO](https://arxiv.org/abs/2603.18636) | layer sparsity is heterogeneous; combine offline layer profiles with online query/key content rather than one geometry route |
| [DFSAttn](https://arxiv.org/abs/2605.23445) | preserve spatial locality during token ordering and cache masks only with adaptive refresh |
| [Sol-Attn](https://arxiv.org/abs/2607.24027) | select and correct inside online softmax so proxy-score materialization and fixed top-k budgets do not erase the theoretical gain |

MiniMax-H3 Turbo has four evaluations, not the dozens used by many diffusion
experiments. The local sensitivity scan found step 4 materially safer than
steps 1–3; a paper's uniform denoise schedule is therefore not transferred to
this target without measurement.

The earlier same-size attempt incorrectly decoded all 37 video latents as one
sequence. It produced 38 non-flat frames followed by 86 flat gray frames. The
released seven-latent, stride-five overlapping schedule removes that failure,
at the cost of increasing Video VAE time from 5,342.026 to 7,387.292 seconds.

## Video VAE structure and optimization

The full 480p N-to-N run measures the optimized Video VAE at 487.273811
seconds versus the 7,387.292038-second pre-optimization baseline: 15.160454×
faster. Component results below use deterministic latents through the same
real 2.604B-parameter Video VAE and remain separate correctness gates.

| VAE property | Value |
|---|---:|
| Decoder graph | 3D convolutional input followed by a ViT3D decoder |
| Transformer layers | 36 |
| Hidden / heads / head dimension | 2,048 / 32 / 64 |
| FF pair width | 8,192; `w1` emits 16,384 values for SwiGLU |
| Per-block dense work | 67,108,864 MAC per sequence row |
| Reference spatial task | 256×256 pixels; 7 latent frames; 1,792 tokens + 4 registers + 1 suffix = 1,797 rows |
| Corrected 480p graph | 7 temporal chunks × 15 spatial tiles × 36 layers = 3,780 block calls |
| Corrected 480p dense work | 455,847,696,138,240 MAC |

### Exact compile-time and runtime reductions

| Change | Target behavior | Correctness status |
|---|---|---|
| Whole-image Metal view | one no-copy `MTLBuffer` per mapped checkpoint instead of one buffer object per tensor use | exact; final smoke frames and audio unchanged |
| Static bindings | resolve 438 Video VAE weight offsets once, before the chunk/tile loop | exact |
| ViT3D RoPE table | build 24 `(cos,sin)` pairs per row once for the fixed tile geometry; reuse for 32 heads × 36 layers × all same-shaped tiles | exact; frames unchanged |
| Command grouping | encode four transformer layers per command buffer instead of waiting after every layer | exact; frames unchanged |
| FP16 matrix kernel | 64 activation rows × 64 output channels; eight simdgroups share one 64×32 weight tile; FP32 accumulation and FP16 public boundary | same result as the 32-row MMA kernel bit for bit |
| Attention K/V tile | eight query rows share 32 K/V rows in threadgroup memory while visiting keys in the original order | all 3,680,256 FP16 outputs match the scalar-query kernel bit for bit |

The whole-image view also applies to H3. On the same 32×32 Turbo-4 smoke,
denoise fell from 17.122752 to 7.571732 seconds because the runtime stopped
creating thousands of repeated tensor-backed Metal resources. This is a
resource-binding reduction, not a model approximation.

### Optimization ledger

This ledger keeps the decision chain, not only the winning endpoint. A row
without an isolated timing says so explicitly; combined measurements are not
retroactively assigned to individual changes.

| Step | Observation | Change and reason | Measured effect | Correctness boundary |
|---:|---|---|---:|---|
| 0 | Decoding all 37 video latents as one sequence left 86/124 flat frames | Adopt the released seven-latent task, five-latent stride and five-frame overlap; speed is irrelevant until the temporal graph is valid | 7,249.519 s invalid run → 9,294.870 s valid baseline; Video VAE 5,342.026 → 7,387.292 s | 124/124 frames, 0 flat frames, four shots |
| 1 | In the valid baseline, Video VAE used 7,387.292 s, 79.477% of 9,294.870 s | Optimize Video VAE before H3; this is the largest Amdahl term | Profiling decision; no runtime change | Same prompt, seed, Turbo-4 schedule and exact 256-pixel tiling fixed for later comparison |
| 2 | Tensor-backed Metal resources were recreated inside repeated graph execution | Map each complete checkpoint once and reuse tensor offsets; object creation and mapping do not belong in a fixed graph's inner loop | 32×32 Turbo-4 denoise 17.122752 → 7.571732 s; full-size VAE contribution not isolated | Exact resource-binding change; smoke frames and audio unchanged |
| 3 | The same 438 VAE tensor bindings, position grid and synchronization pattern recur for every tile | Resolve bindings once, precompute 24 complex RoPE coefficients per row, and encode four layers per command buffer; fixed geometry makes all three reusable | No trustworthy isolated timing; the combined prebound scalar reference is 68.619048 s per 256×256×22 task | Static bindings exact; RoPE/frame regression exact; grouping changes submission only |
| 4 | Scalar dense projections dominate the prebound graph; one real `w1` is 932.993 ms while scalar-query attention is 63.254 ms | Replace scalar projection loops with 32-row FP16 simdgroup-matrix tiles and FP32 accumulation | Component 68.619048 → 6.824399 s, 10.055× | PSNR 66.821 dB, SSIM 0.999875 versus scalar; FP16 public boundary retained |
| 5 | Each scalar query reloads the same keys and values | Let eight queries share 32 K/V rows in threadgroup memory while preserving key order | Component 6.824399 → 6.281185 s, 1.086×; attention 63.254 → 45.424 ms | 0/3,680,256 FP16 attention outputs differ; 22/22 frames bit identical to step 4 |
| 6 | The 32-row GEMM still reloads weight tiles across neighboring activation rows | Share each 32-row weight tile across the simdgroups consuming it | Component 6.281185 → 6.068952 s, 1.035× | 22/22 frames bit identical to step 5 |
| 7 | More activation rows can amortize each staged 64×32 weight tile | Process 64 activation rows and 64 output channels per dispatch | Component 6.068952 → 5.030190 s, 1.207×; real `w1` scalar 932.993 → 37.685 ms | Bit identical to the 32-row MMA boundary; final component remains PSNR 66.821 dB, SSIM 0.999875 versus scalar |
| 8 | A one-tile projection predicted 528.170 s for 105 exact full-size tasks | Run the complete graph instead of claiming the projection | Video VAE 487.273811 s, 7.743% below projection and 15.160× over the valid baseline; total 2,418.708237 s, 3.843× overall | Full prompt-to-media path passed; PSNR 43.720 dB and SSIM 0.984754 versus the pre-optimization encoded video; audio bit identical |

The final run did not reuse an earlier latent or media output. It executed 50
conditioner layers, 200 H3 layer/evaluation calls, 3,780 Video VAE block calls,
seven Audio VAE upsampling stages and a new mux. Local packed-image cache hits
skipped network transfer and offline import only. The model runtime was
2,418.708237 seconds; the shell wall was 2,484.38 seconds because it also
included SHA-256 checks of the 28.22 GB conditioner and 780 MB Turbo adapter.
Post-run media verification is excluded from both numbers.

Rejected and deferred branches remain part of the record:

| Candidate | Evidence | Decision and reason |
|---|---:|---|
| 16-query attention tile | 62.300 ms versus 45.424 ms for eight queries | rejected; lower occupancy costs more than the additional K/V reuse saves |
| `MTLBinaryArchive` in the default runtime | 36.619 ms setup versus 32.106 ms without | rejected as a default; it increased measured startup |
| 272×272 spatial tile | 14.153 → 10.269 s on 256×480×22; PSNR 46.320 dB, SSIM 0.993609 | retained as approximate research; full-run quality gate is open |
| Indirect command buffers | CPU encoding is negligible after four-layer grouping | deferred; persistent constant buffers must replace `setBytes` before ICB can remove useful work |

### Measured component increments

Command:

```sh
make minimax-h3-m3-e2e
build/minimax-h3-m3-e2e --video-vae-smoke OUTPUT_DIR 256 256 22
```

| 256×256×22 real-weight VAE path | Seconds | Increment | Output gate |
|---|---:|---:|---|
| Scalar GEMM + scalar-query attention on the current prebound graph | 68.619048 | baseline | finite 22-frame output |
| 32-row simdgroup-matrix GEMM | 6.824399 | 10.055× | versus scalar: PSNR 66.821 dB, SSIM 0.999875 |
| + eight-query tiled attention | 6.281185 | 1.086× | 22/22 PPM frames bit identical to prior row |
| + 32-row shared-weight GEMM | 6.068952 | 1.035× | 22/22 PPM frames bit identical |
| + 64-row shared-weight GEMM | **5.030190** | **1.207×** | 22/22 PPM frames bit identical |
| **Current versus scalar compute reference** | — | **13.641×** | PSNR 66.821 dB, SSIM 0.999875 |

The actual 1,797-row kernel boundaries explain the remaining time:

| Kernel | Reference | Selected | Speedup | Differential |
|---|---:|---:|---:|---:|
| Video attention | 63.254 ms scalar-query | 45.424 ms, 8 queries/group | 1.393× | 0 / 3,680,256 FP16 values differ |
| `w1`, 1,797×2,048→1,797×16,384 | 932.993 ms scalar | 37.685 ms, batch-64 MMA | 24.757× | 0 values differ from batch-32 MMA |
| 16-query attention candidate | 63.254 ms | 62.300 ms | 1.015× | rejected: occupancy loss removes the reuse gain |

### Tile geometry is a separate approximation

The released decoder uses 256-pixel tiles. For a 256×480×22 component test,
compiling the vertical tile to 272 pixels changes three tiles into two because
`272 + 272 - 64 = 480`.

| Layout | Tasks | Seconds | Comparison to 256-tile oracle |
|---|---:|---:|---|
| 256-pixel vertical tile | 3 | 14.152762 | oracle |
| 272-pixel vertical tile | 2 | 10.268616 | 1.378×; PSNR 46.320 dB; SSIM 0.993609 |

This candidate is not enabled by default. At 864×480, 272×272 tiles would
reduce each temporal chunk from 15 tasks to 8. Multiplying measured one-tile
times projected 528.170 seconds for the exact 256 layout; the complete run
measured 487.274 seconds, 7.743% below that projection. The 326.679-second
estimate for the experimental 272 layout remains a projection; its complete
run quality and thermal behavior are open.

### Metal API decisions

| API or mechanism | Decision on this target |
|---|---|
| Offline `.air` → `.metallib` | enabled; all MSL is compiled at build time |
| [`MTLBinaryArchive`](https://developer.apple.com/documentation/metal/mtlbinaryarchive) | reproducible compiler target retained, but default loading rejected: 36.619 ms setup with the archive versus 32.106 ms without it |
| [`MTLIndirectCommandBuffer`](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand) | valid for the fixed 36-layer graph, but not yet useful: four-layer grouping makes CPU encoding negligible; every `setBytes` constant must first move into a persistent buffer |
| Argument buffers | useful only after the graph becomes ICB-driven; one whole-image buffer already removes the repeated resource-object cost |
| Metal 4 tensors / ML encoder | not a deployment option on the pinned macOS 15.7.3 machine; the installed SDK declares these APIs for macOS 26; [Metal 4 introduces tensors and an ML encoder](https://developer.apple.com/videos/play/wwdc2025/205/) |
| MPS | benchmark oracle only; production remains custom MSL with a minimal Objective-C bridge |
| VideoToolbox | next media-output target: encode finalized temporal chunks without writing 124 PPM files |

The next exact compiler experiment is an offline 64×32 tiled weight image so
the selected GEMM reads each staged weight tile sequentially. Approximate
experiments are per-channel/groupwise W8 VAE weights and 272×272 spatial
geometry; both require decoded-media gates. Folding latent mean/std,
`post_quant_conv` and `x_embedder` is also possible, but removes the current
FP16 boundary and therefore needs a differential rather than an algebra-only
claim.

Audio VAE is lower priority. It is a 151.327M-parameter F32 BigVGAN-style
decoder with seven upsampling stages; the optimized 480p run spent 13.760
seconds there, 0.569% of total time. Its direct convolution, alias/Snake
activation and residual operations can be tiled and fused in Metal after the
denoiser ceases to dominate.

The external peer publishes three sampling profiles: Turbo 4 Fast, Turbo 6
Balanced and non-Turbo Quality 20. The comparable front-page sample uses four
denoise evaluations; six and twenty steps are quality modes, not the baseline
for the 192-second sample. See
[`MacOS-H3-Speedrun`](https://github.com/EvolvingLMMs-Lab/MacOS-H3-Speedrun#选择配置).

## Initial 128×128 N-to-N gate

The historical media and records are under
[`artifacts/hummingbird-128x128-euler30/`](artifacts/hummingbird-128x128-euler30/)
and summarized in [`results.json`](results.json). Generation timing and
post-run verification are separate.

| Field | Measured value |
|---|---:|
| Prompt | `a jeweled hummingbird hovering beside a red orchid, cinematic natural light` |
| Prompt tokens / packed rows | 15 / 201 |
| Geometry | 128×128×22 at 24 fps |
| Seed / denoise intervals | 314159 / 30 |
| Tokenizer / text image access / text encode | 0.008 / 7.132 / 2.258 s |
| Transformer + AdaLN first-use acquisition | 1,597.825 s |
| Denoise | 287.148 s |
| Video VAE acquisition / decode | 684.234 / 20.095 s |
| Audio VAE acquisition / decode | 79.463 / 2.479 s |
| Complete first-use wall | 2,700.482 s |
| Measured-stage subtotal excluding H3 weight acquisition | 319.611 s |
| Process CPU user / system | 213.406 / 299.469 s |
| Peak process physical footprint | 173,843,520 bytes |
| Verified media | 22/22 H.264 frames; 32 kHz stereo AAC; non-silent |
| Quality boundary | coarse prompt correspondence; subject is not sharply identifiable at 128×128; reference parity open |

The original `-shortest` A/V mux truncated the already verified 22-frame intermediate video to 20 frames at the AAC boundary. It was removed; remuxing the generated video and audio took 0.06 s and is excluded from generation timing. The final artifact SHA-256 is `576737d39aa5d2f5b24dd2e189cd7ee8dcc3544c9f78ca61ce28458a39c04225`.

## Run a prompt

```sh
tools/minimax_h3_generate.sh "a hummingbird hovering beside a red orchid"
```

With no argument, the command asks for a prompt. Its verified default is
128×128, 22 frames at 24 fps. Example override:

```sh
MINIMAX_H3_WIDTH=256 MINIMAX_H3_HEIGHT=256 \
MINIMAX_H3_FRAMES=56 MINIMAX_H3_SEED=314159 \
tools/minimax_h3_generate.sh "a hummingbird hovering beside a red orchid"
```

Generation is offline. `MINIMAX_H3_TEXT_ENCODER` must name the complete local
28,222,740,739-byte `text_encoder.safetensors`, and all four packed H3/VAE
caches must already exist under `tmp/minimax-h3-m3-cache/`. The command exits
before inference when any local file is absent. HTTP ranges are allowed only
inside an explicit offline compiler/import step; the runtime never reads model
weights over the network.

The execution checkpoint is not the official full-precision image. It is the
pinned third-party artifact derived from MiniMax-H3 Base: the 50-layer text
conditioner uses affine Q8, the 50-block H3 transformer uses group-64 affine
Q4 with BF16 runtime state, the Video VAE weights are F16, and the Audio VAE
weights are F32. The model graph is complete for this path; the weight
precision is quantized/mixed rather than full precision.

## Downstream component gate

This gate intentionally bypasses prompt encoding with a fixed one-row conditioner state. It proves the real-weight denoiser, both decoders and mux, but it is not N-to-N.

| Field | Measured value |
|---|---:|
| Geometry | 32×32×22 at 24 fps |
| H3 packed rows | 82 |
| Transformer weight download | 1,607.877 s |
| 30 denoise intervals | 232.055 s |
| Video VAE weight download / decode | 684.209 / 2.246 s |
| Audio VAE weight download / decode | 79.394 / 2.542 s |
| Mux | 0.709 s |
| Complete wall time | 2,612.518 s |
| Process CPU user / system | 248.064 / 333.892 s |
| Peak physical footprint | 147,950,592 bytes |
| Result | playable MP4; structural component gate passed |

The low footprint is a staged, file-backed measurement. It does not mean that the checkpoint files total 148 MB.

## External speed target

The comparison is pinned to [MacOS-H3-Speedrun commit `c8e0f73`](https://github.com/EvolvingLMMs-Lab/MacOS-H3-Speedrun/tree/c8e0f732fe82b00704738feff769cff621a4f7b2), with its [SolAttn dependency at `886f4b9`](https://github.com/yshenaw/ComfyUI-SolAttn-MPS/tree/886f4b9b8598b2d17445eef659a8ef28cf04d0d4).

| Challenge field | Value |
|---|---|
| Workflow | T2VA, synchronized stereo audio |
| Canvas | 864×480 |
| Frames | 124 at 24 fps, 5.167 s |
| Text rows | 86 in the published attention workload |
| Packed rows | 15,485 |
| Sampling | MiniMax-H3 Turbo v4-600 EMA, four steps |
| Seed | 42 in the published workflow JSON |
| Prompt | perfume-advertisement prompt pinned in the workflow JSON |
| Exact conditioning sink | eight 64-row blocks in the published attention microbenchmark |

The four evaluations are pinned separately from the official 20-point
Diffusers grid. ComfyUI `simple` selects base sigmas from its 1,000-point flow
grid, then H3 applies separate video and audio shifts:

| Evaluation | Base sigma | Video sigma, shift 12 | Video timestep | Audio sigma, shift 3 | Audio timestep |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| 1 | 0.75 | 0.972973 | 0.027027 | 0.900000 | 0.100000 |
| 2 | 0.50 | 0.923077 | 0.076923 | 0.750000 | 0.250000 |
| 3 | 0.25 | 0.800000 | 0.200000 | 0.500000 | 0.500000 |
| terminal | 0.00 | 0.000000 | — | 0.000000 | — |

The generic C tests pin the float32 bit patterns, not only these rounded values.

The current peer records two different workloads. They must not be merged into
one synthetic comparison:

| Peer record | Geometry | Reported time | Comparison limit |
|---|---:|---:|---|
| One attention call, M3 Ultra BF16 | 864×480×124; 15,485 rows | 308.36 ms native MPS; 59.53 ms SolAttn | kernel only; excludes projections, sampler and VAEs |
| Front-page synchronized sample | 832×480×124 at 24 fps | 192 s complete generation | the page does not attach a per-file hardware, peak-memory or stage record |

The M3 Ultra attention number is not a full-generation time and was not
measured on this M3 Pro. The 192-second sample uses a different width and
prompt. Both remain useful targets, but neither is an equivalent-run result.

## Independent execution design

MacOS-H3-Speedrun is a measurement peer. Its PyTorch/ComfyUI node graph is not the runtime design.

| Layer | C/Metal target decision | Difference from the external path |
|---|---|---|
| Prompt | compile the benchmark prompt directly to H3 context; later add staged W4 Qwen layers 0-49 | no 52.7 GB one-stage or 28.16 GB two-stage BF16 conditioner at benchmark runtime |
| AdaLN | emit exact per-block outputs for the four pinned timesteps and three modalities | no curve interpolation or target-time AdaLN projection |
| Transformer weights | mixed affine W4/W8, Metal-native order, layerwise error gates | not a 40 GB BF16 pruned DiT |
| Graph | one compiled 50-block schedule with fixed offsets and buffer lifetimes | no dynamic Python graph, hooks, loader or cache manager |
| Attention | hierarchical spatial/temporal summary tree, compiled conditioning summaries and online FP32 softmax/LSE | not SolAttn's flat 64-row all-pairs block routing |
| Cross-step reuse | retain route/tree state and refresh changed nodes | routing is treated as denoise state, not rebuilt from scratch at every call |
| Token reuse | cache ungated attention/MLP branch outputs; apply the current AOT gate on reuse | avoids carrying the previous evaluation's AdaLN gate into the current one |
| QKV | summarize normalized hidden rows first, project exact selected rows plus 366 tree summaries, and stream selected Q tiles | H3 K/V are bias-free, so `W·mean(x) = mean(W·x)`; dense K/V projection is unnecessary for summarized branches |
| MLP | fuse RMSNorm, AdaLN, W4 expansion, SwiGLU and W4 reduction in target-specific tiles | no framework operator boundaries |
| VAE | staged W8/FP16 C/Metal decoder with convolution/activation fusion and fixed tiling | no generic PyTorch VAE graph |
| Media output | overlap decoded-tile transfer and VideoToolbox encode where dependencies allow | output work is scheduled with the decoder rather than after the whole graph |

The hierarchical attention path is approximate. Dense online attention remains the correctness oracle. The first denoise evaluation stays dense until validation proves a different schedule safe.

The conservative route keeps the 86 text and 414 audio rows exact. The measured
H3-native route instead represents them with 11 eight-row text seed summaries
and 13 32-row audio summaries. The text grouping is only a performance seed;
the compiler must replace it using dense-teacher trajectory search before the
route can pass quality. The 14,985 video rows form 296 spatial leaves: each of
37 latent frames has a 15×27 patch grid split into at most 8×8 leaves. Frame
summaries feed eight temporal nodes grouped by the Video VAE's five-latent
cadence, then one root. A query expands its own leaf and represents every other
branch once at the leaf, frame or temporal level.

The runtime stores rows in tree-major physical order while retaining the
logical packed-row map. This makes one at-most-64-row leaf contiguous, allowing
eight Metal simdgroups to share route K/V tiles in one threadgroup.

## Memory plan

| Resident phase | Planned contents | Budget |
|---|---|---:|
| Compiled-prompt benchmark | prompt context, tokenizer-free metadata | <0.1 GiB |
| Transformer | mixed W4/W8 weights, exact AdaLN tables, Metal buffers | 12-16 GiB weights plus bounded state |
| Free-prompt conditioner | W4 Qwen layers 0-49, loaded before and released before transformer | 12-15 GiB |
| Video VAE | W8/FP16 packed decoder plus two tiles | <7 GiB |
| Audio VAE | packed decoder and stereo latent state | <1 GiB |
| Process peak | one heavy phase plus shared latents/output buffers | <32 GiB, zero swap |

The 36 GB machine cannot load the released 134.13 GiB workflow directly. Staging is part of the artifact contract, not an optional low-memory mode.

## Optimization order

| Order | Change | Correctness gate | Performance gate |
|---:|---|---|---|
| 1 | exact dense C/Metal block | tensor parity against the pinned reference | one block fits and runs without score-matrix allocation |
| 2 | exact AdaLN and static layout compilation | bitwise table/index checks | remove all target-time AdaLN projection dispatches |
| 3 | mixed W4/W8 projections and MLP | layerwise relative error and trajectory test | transformer image fits resident budget |
| 4 | hierarchical tree attention | dense-attention differential suite | <59.53 ms at 15,485 rows |
| 5 | cross-step tree/route reuse | route recall and fixed-seed media comparison | sampler <141.54 s |
| 6 | compiled prompt and fused VAE decoders | context and decoded-boundary comparisons | complete generation <194.18 s |
| 7 | free-prompt staged conditioner | H3 context cosine and final media tests | peak <32 GiB, zero swap |

An optimization is not enabled in the release path until both gates pass.

## Current evidence

| Check | Result |
|---|---|
| 864×480×124 geometry | 14,985 video + 414 audio + 86 text = 15,485 rows |
| Generic dual scheduler | shift 12 and shift 3 float32 grids pass pinned bit patterns |
| Generic row/AdaLN indices | pass on a caller-owned-buffer test |
| Turbo-4 schedule | four video/audio timestep pairs pass pinned float32 bit patterns |
| Turbo-4 AOT layout | 12 reachable block-modulation slots and 7 deduplicated final-norm slots; 38,895,616-byte aligned image plan passes |
| Video tree topology | 296 spatial leaves + 37 frame nodes + 8 cadence nodes + 1 root; exhaustive row-coverage test passes |
| Synthetic attention primitive | 49.893 ms GPU average including LSE output, 5 measured iterations; 0.947e-6 max error against the same approximate C primitive |
| Primitive memory | 1,031,505,280-byte process footprint; 902,068,032 Metal-owned bytes |
| Full pre-gate Q4 branch cache plan | 4,682,664,000 bytes for 50 layers × 2 branches × 15,485 rows; layout/offset test passes |
| W4 projection primitive | one 5,376→14,336 projection: 618.205 ms for 15,485 rows or 194.602 ms for 4,646 rows; not a block time |
| Sparse projection seed | first reusable evaluation, central layer band: Q 3,054 rows, K/V 3,420 rows, MLP 2,583 rows; 18.370% of dense projection MACs |
| Aggressive H3 trajectory quality | measured at 480p; 10 instead of 3 scene cuts, adjacent-frame luma 9.814 versus 4.496 exact; rejected |
| Frame-safe H3 trajectory quality | measured at 480p; same three scene cuts, adjacent-frame luma 4.283, PSNR 32.385 dB / SSIM 0.932134 versus exact; single-prompt pass, prompt-suite gate open |
| Full Metal transformer | real-weight 50-block × 30-interval path completed at 32×32×22 after changing the residual stream from FP16 to BF16 |
| Video and audio decoders | real-weight Video VAE and Audio VAE completed; H.264 + WAV mux produced a playable MP4 |
| Free-prompt conditioner | real 15-token prompt completed all 50 streamed Q8 layers and the full N-to-N media path |
| Free-prompt output verification | 22/22 video frames, stereo audio non-silent, coarse prompt response visible; reference-quality parity open |
| Corrected 480p N-to-N run | 864×480×124 Turbo-4 completed in 2,418.708 s; 4.136 GiB runtime peak physical footprint; zero swaps; 3.843× over the pre-optimization baseline |
| Corrected 480p media verification | 124/124 decoded frames, 0 flat frames, three detected cuts forming four prompt-aligned shots; no sampled tile or temporal-chunk break |
| Frame-safe 480p N-to-N run | 2,344.734 s; 4.584 GiB peak; same-run selected groups 92.597 → 70.313 s; exact remains default |

## Attention experiment

Run:

```sh
make build/minimax-h3-m3-attention-bench build/minimax-h3-m3-attention.metallib
build/minimax-h3-m3-attention-bench build/minimax-h3-m3-attention.metallib 5 2
```

| Increment | Route / layout change | Attention GPU ms | Change |
|---|---|---:|---:|
| Scalar simdgroup | one query/head, conservative route | 2,568.763 | baseline |
| 8-row MMA | small matrix tile, same route | 2,965.231 | 0.87×; rejected |
| 64-row tree-major MMA | eight simdgroups share K/V | 405.675 | 6.33× |
| H3-native audio/video tree | text exact; audio summaries; one video leaf | 70.902 | 36.23× |
| Compiled-conditioning seed | text and audio summaries | **45.020** | **57.06×** |
| Selection signal | same route plus all-row/head LSE output | **49.893** | **51.49×** |

The final row is 1.193× faster than the peer's published 59.53 ms time despite
running on M3 Pro rather than M3 Ultra. This is not yet an equivalent-quality
claim: it excludes projections and its route has not passed a real-weight
dense trajectory comparison. Raw fields are in [`results.json`](results.json).

## Projection experiment

Run:

```sh
make build/minimax-h3-m3-gemm-bench build/minimax-h3-m3-attention.metallib
build/minimax-h3-m3-gemm-bench build/minimax-h3-m3-attention.metallib 15485 5 2
```

One dense H3 block contains five times the MAC count of the measured
5,376→14,336 matrix: Q and output together are one equivalent matrix, K/V one,
and the fused SwiGLU input plus output are three. The benchmark is therefore a
kernel measurement, not a layer-time claim.

| Increment | Rows | GPU ms | Effective FP16 TFLOP/s | Max error, first 8 outputs |
|---|---:|---:|---:|---:|
| 32-output-row tile | 15,485 | 662.156 | 3.605 | old degenerate arithmetic fixture; timing only |
| 64-output-row tile, FP32 spill each 64 K | 15,485 | **618.205** | **3.861** | 0.002581 |
| Same kernel, selected-row sample | 4,646 | **194.602** | 3.680 | 0.002581 |

The arithmetic fixture varies Q4 blocks and affine metadata. The error is
against a scalar C FP32 sum using the same FP16-dequantized weights. It is not a
model-quality measurement.

The real-weight differential uses layer 0 `mlp.fc1`, eleven BF16 activation
rows from the local transformer image and the exact BF16 runtime kernel:

| Boundary | GPU time | Reference | Maximum absolute error, first 8 outputs |
|---|---:|---|---:|
| affine-Q4 group-64, BF16 input/output | 5.322 ms | scalar C with FP32 accumulation and BF16 output rounding | 0.002823 |

This establishes one projection boundary. It does not establish complete
block or denoising-trajectory parity.

The first branch-cache evaluation must be dense because no prior branch output
exists. Evaluation one may drop sharply; only evaluations one through three
must relax monotonically toward lower noise. The earlier all-four-step
monotonic rule made reuse impossible and has been removed from the C validator.

| Turbo evaluation, central 10-layer band | Q rows | K/V projection rows | MLP rows | Projection MACs / dense |
|---:|---:|---:|---:|---:|
| 0 | 15,485 | 15,485 | 15,485 | 100.000% |
| 1 | 3,054 | 3,420 | 2,583 | 18.370% |
| 2 | 4,914 | 5,280 | 4,144 | 29.223% |
| 3 | 7,993 | 8,359 | 7,203 | 49.029% |

These ratios are an unvalidated search seed. They are disabled until a dense
teacher trajectory establishes the media-quality loss. Exact counts and raw
measurements are in [`results.json`](results.json).
