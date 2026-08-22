# llm-in-c

LLM inference in C, compiled per model and per chip.

An offline compiler fixes tensor formats, graph rewrites, memory placement and schedules for one exact checkpoint on one exact machine; the target receives immutable model images, a compact C runtime and only the kernels that model–hardware pair needs.

Runtime APIs and graph control are written in C. A target backend may add the minimum platform layer required by its hardware—for example, Objective-C and Metal shaders on Apple Silicon or NEON intrinsics on Arm.

| Compile time | Deployment time |
|---|---|
| Model revision, tokenizer and source hashes | Verified packed model images |
| Architecture-specific graph rewrites | C runtime with fixed state lifetimes |
| Quantization and tensor layout | CPU or on-SoC accelerator kernels |
| Target topology and memory constraints | No Python or training framework |

The project is not a wrapper around llama.cpp, MLX or ONNX Runtime. Those projects remain useful reference implementations and benchmark peers. The code here owns checkpoint import, image formats, runtime state, tokenization, sampling and target kernels.

## Implemented model × target pairs

Every performance claim belongs to one exact model revision and one exact target. Numbers do not transfer to a different chip or model.

| Model | CPU / SoC | Execution path | Measured result | Evidence |
|---|---|---|---|---|
| Qwen3.8-27B | Apple M3 Pro, 36 GB unified memory | C runtime + Metal kernels, affine Q4, FP16 KV, adaptive MTP | **11.95 end-to-end tok/s** five-workload aggregate, up to **17.59** on long code; 7.96 tok/s without MTP, against 8.29 for mlx-lm on the same weights and machine | [model](models/qwen3.8-27b/README.md) · [target](models/qwen3.8-27b/targets/apple-m3-pro/README.md) · [raw results](models/qwen3.8-27b/targets/apple-m3-pro/results.json) · [review](models/qwen3.8-27b/targets/apple-m3-pro/REVIEW.html) |
| Qwen3.8-27B | NVIDIA Jetson Orin Nano Super, 8 GB unified, CUDA 13.2 | llama.cpp CUDA, dynamic IQ1_S, direct-I/O full-GPU residency, bounded recurrent checkpoint state | **4.17 decode tok/s** benchmark; completed Chatbox turns took **114–156 s** and approached 1 GiB process swap; fits, but not recommended for interactive chat | [model](models/qwen3.8-27b/README.md) · [target](models/qwen3.8-27b/targets/jetson-orin/README.md) · [raw results](models/qwen3.8-27b/targets/jetson-orin/results.json) |
| MiniMax-H3 | Apple M3 Pro, 36 GB unified memory | C/Metal tokenizer, streamed Q8 conditioner, affine-Q4/BF16 H3, single-pass exact attention, optimized Video VAE, Audio VAE | Exact 864×480×124 Turbo-4: **2,194.44 s** (was 2,418.71 s); all three scene cuts stable, per-frame review clean; zero swap | [model](models/minimax-h3/README.md) · [target](models/minimax-h3/targets/apple-m3-pro/README.md) · [raw results](models/minimax-h3/targets/apple-m3-pro/results.json) · [review](models/minimax-h3/targets/apple-m3-pro/H3-ATTENTION-QUALITY-REVIEW.html) |
| Qwen3.5-0.8B | Amlogic A113X, 4× Cortex-A53, 2 GB | C11 + NEON, model-specialized DeltaNet state | **3.64 prompt tok/s**, **2.60 decode tok/s**, 488 MiB generation RSS, zero swap | [model](models/qwen3.5-0.8b/README.md) · [target](models/qwen3.5-0.8b/targets/a113x/README.md) · [raw results](models/qwen3.5-0.8b/targets/a113x/results.json) |
| MiniMind-O | Amlogic A113D/A113X, 4× Cortex-A53, 2 GB | native C11 online SenseVoice → incremental Thinker/Talker → stateful Mimi, W8A8 + NEON + shared persistent workers | Input prefill overlaps speech; first PCM goes directly to ALSA; exact Mimi kernels reach **0.62–0.69 s/0.64 s audio**, while concurrent cadence remains **99–128 ms per 80 ms frame** | [model](models/minimind-o/README.md) · [target](models/minimind-o/targets/a113x/README.md) · [raw results](models/minimind-o/targets/a113x/results.json) · [A113X release](https://github.com/baryhuang/llm-in-c/releases/tag/minimindo-native-a113x-v1.3.0) |
| MOSS-TTS-Nano-100M | Amlogic A113X, 4× Cortex-A53, 2 GB | native C++17/GGML, Q8_0, exact incremental KV cache, persistent server, Cortex-A53/NEON build | **9.83 s** warm for a 0.64 s cloned reply, down from 125.73 s (**12.79×**); byte-identical WAV, 841 MiB peak RSS, zero process swap | [model](models/moss-tts-nano/README.md) · [target](models/moss-tts-nano/targets/a113x/README.md) · [raw results](models/moss-tts-nano/targets/a113x/results.json) |
| Whisper large-v3 | NVIDIA Jetson Orin Nano Super, 6 cores, 7 GB unified, CUDA 13.2 | whisper.cpp CUDA runtime + per-chip native kernel patch: fused LayerNorm, fused bias-GELU, GEMM bias epilogue, f32-output GEMM | **3.05 RTFx** fp16 (upstream 2.87, +6.3%), 4.22 RTFx q4_0 reference arm; WER 0.301%, transcripts byte-identical to upstream; 4.5 GB RSS, zero swap | [model](models/whisper-large-v3/README.md) · [target](models/whisper-large-v3/targets/jetson-orin/README.md) · [raw results](models/whisper-large-v3/targets/jetson-orin/results.json) |
| Whisper large-v3-turbo | NVIDIA Jetson Orin Nano Super, 6 cores, 7 GB unified, CUDA 13.2 | same patched runtime (shared build with large-v3) | **7.79 RTFx** fp16 (upstream 7.09, +9.8%), 10.26 batch RTFx with the CPU/GPU pipeline; 2.70 ms/token decode at q4_0 in 1.2 GB; same-file cross-device: **3.0× faster than A311Y3 NPU** (1.50 vs 4.50 s on std30.wav); WER 0.301%, transcripts byte-identical | [model](models/whisper-large-v3-turbo/README.md) · [target](models/whisper-large-v3-turbo/targets/jetson-orin/README.md) · [raw results](models/whisper-large-v3-turbo/targets/jetson-orin/results.json) |
| Whisper large-v3-turbo | Rockchip RK3588, proprietary 3-core RKNPU2 + 4× Cortex-A76 | Python-free C++17 LLMC runtime, DMA-BUF/in-place KV, selective cache sync, all-core NPU scheduling | `std30.wav` FP16 in **17.933 s** excluding load; encoder **8.725 s**, decoder **111.56 ms/token**, 2.552× over the RKNPU2 host-copy baseline; exact 77-token match; no 32-file run. Earlier Rocket/GGML measurements are retained as a separate appendix. | [model](models/whisper-large-v3-turbo/README.md) · [target](models/whisper-large-v3-turbo/targets/rk3588/README.md) · [raw results](models/whisper-large-v3-turbo/targets/rk3588/results.json) |
| Whisper large-v3-turbo | Amlogic A113X, 4× Cortex-A53, 2 GB | C11 + grouped Q4 NEON, compact audio window, double-precision NEON stem | 11 s JFK in **236.93 s**, 448.2 MiB RSS, zero swap; stem 1.57× and end to end 1.052× faster, generated IDs/logits unchanged | [model](models/whisper-large-v3-turbo/README.md) · [target](models/whisper-large-v3-turbo/targets/a113x/README.md) · [raw results](models/whisper-large-v3-turbo/targets/a113x/results.json) |
| Whisper small.en | Amlogic A113X, 4× Cortex-A53, 2 GB | C11 + NEON, mixed Q4/Q8 encoder and cached decoder | 11 s audio in **45.0 s**, 251 MiB RSS, zero swap; 0/22 word errors on the pinned JFK sample | [model](models/whisper-small.en/README.md) · [target](models/whisper-small.en/targets/a113x/README.md) · [raw results](models/whisper-small.en/targets/a113x/results.json) |
| Gemma 4 E2B | Unpinned two-vCPU x86-64 development machine | Legacy restricted C artifact | 0.598 token/s, 926 MiB RSS, zero swap | [model](models/gemma-4-e2b/README.md) · [raw results](models/gemma-4-e2b/results.json) |

The MiniMax-H3 pipeline uses static tensor bindings, precomputed RoPE, grouped
command buffers, simdgroup-matrix GEMM and exact attention. The same 480p
N-to-N workload fell from 9,294.870 to 2,194.437 seconds (4.236×); Video VAE
decode fell from 7,387.292 to 479.511 seconds. The latest attention round
replaced the two-pass exact kernel with a single-pass online-softmax kernel —
same mathematics, fp32 accumulation, at most one bf16 unit of rounding
difference on 0.19 percent of outputs — for a ~1.33× kernel speedup and 224
seconds off the exact path. A faster variant with bf16 probability fragments
and two approximate attention paths (a hierarchical tree and a late-layer
candidate) are all rejected on frame-level review: each duplicates or smears
line work in high-motion close-ups in ways whole-video luma averages fail to
catch. Attention quality gates now require per-frame review of the
highest-motion shot, and prompts are checked against the model's shot
budget — one to two shots per six seconds — before artifacts are
attributed to the runtime. The
[target record](models/minimax-h3/targets/apple-m3-pro/README.md#video-vae-structure-and-optimization)
separates measured end-to-end results, component gates and projections.
Its [optimization ledger](models/minimax-h3/targets/apple-m3-pro/README.md#optimization-ledger)
records each observation, reason, isolated or combined timing, correctness
boundary and rejected branch. The optimized run recomputed all 200 H3 layer
calls and all 3,780 Video VAE block calls; local cache hits skipped weight
download/import, not inference.

Gemma is the early restricted artifact and accepts only its compiled test
inputs. Qwen3.5, Qwen3.8, Whisper and MiniMax-H3 execute their full model
paths; task behavior is selected by input or prompt rather than compiled
labels.

## Qwen3.8-27B on Apple M3 Pro

Qwen3.8 is the most complete target in the repository. It includes free-text chat, batched prefill, incremental multi-turn state, FP16 KV cache, greedy and sampled decoding, adaptive multi-token prediction, thinking-mode streaming and an OpenAI-compatible local server.

| Five-workload resident run | Plain greedy | Adaptive MTP | Increment |
|---|---:|---:|---:|
| End-to-end throughput | 7.96 tok/s | **11.95 tok/s** | **1.50×** |
| Decode throughput | 8.06 tok/s | **12.32 tok/s** | **1.53×** |
| Long-code case end-to-end | 7.98 tok/s | **17.59 tok/s** | **2.20×** |
| Total request wall | 415.0 s | **276.0 s** | 139.0 s less |

Speculative output is token-identical to plain greedy on four of the five cases; one prose case flips a numerical near-tie onto an alternate fluent greedy continuation, and an exact-verify mode restores full token identity at 9.81 aggregate tok/s.

One separate 36-token-prompt, 28-token-reply comparison used resident models and four measured rounds:

| Runtime | End-to-end throughput | Request wall |
|---|---:|---:|
| This runtime, plain | 6.29 tok/s | 4.450 s |
| This runtime, adaptive MTP | **9.01 tok/s** | **3.109 s** |
| llama.cpp build 10360, Unsloth Q4_K_M | 5.76 tok/s | 4.864 s |

The llama.cpp checkpoint uses a different Q4 format, so this is a runtime-level comparison, not token-level quantization parity.

The stronger local baseline on this machine is Apple's own stack. The
five-workload set was run through mlx-lm 0.31.3 and oMLX 0.5.7 against the
same `mlx-community/Qwen3.8-27B-4bit` checkpoint this runtime compiles from:

| Stack | Five-workload aggregate | Long-code case |
|---|---:|---:|
| mlx-lm 0.31.3 / mlx 0.32.0 | 8.29 tok/s | 8.36 tok/s |
| oMLX 0.5.7 | 8.28 tok/s | 8.30 tok/s |
| This runtime, plain | 7.96 tok/s | 7.98 tok/s |
| This runtime, adaptive MTP | **11.95 tok/s** | **17.59 tok/s** |

Without speculation this runtime is 0.96× the best Apple stack. 15.139 GB of
mapped weights over 117–118 GB/s measured is a first-order ceiling near 7.8
tokens/s for any decoder that reads every weight once per token, and all three
non-speculative arms land within 6% of it. The speculative arm clears the
bound by amortizing one weight pass over several accepted tokens. Exact sources, hashes, prompts, output checks and per-case timings are in the [target record](models/qwen3.8-27b/targets/apple-m3-pro/results.json).

## How the repository is organized

Classification is model first, hardware target second:

```text
checkpoint + tokenizer + model rules + target profile
                         |
                         v
                  offline compiler
                         |
                         v
          packed images + target runtime + kernels
```

| Path | Responsibility |
|---|---|
| [`models/<model>/`](models/) | Source pins, architecture facts, model-level decisions and validation records |
| `models/<model>/targets/generic/` | Portable model runtime before CPU-specific specialization, where available |
| `models/<model>/targets/<target>/` | Target kernels, layouts, schedules, runtime code and measured results |
| [`compiler/`](compiler/README.md) | Offline checkpoint inspection, packing, fixture generation and target image compilers |
| [`tools/`](tools/README.md) | Commands users run directly: chat, serving, monitoring, comparison and hardware probing |
| [`tests/`](tests/) | Import, hashing, packing, sampler, state-machine and numerical parity tests |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Artifact contract and validation rules |

Checkpoints, generated model images, binaries and credentials are not committed.

## Why memory topology matters

Autoregressive decode repeatedly visits model weights. Once kernels are efficient, token generation approaches a bandwidth problem:

```text
decode tokens/s <= usable memory bandwidth / bytes visited per token
```

| Target | Resident model/image | Measured memory fact | Resulting engineering focus |
|---|---:|---:|---|
| Apple M3 Pro / Qwen3.8-27B | 15.139 GB mapped text image | 117–118 GB/s measured across complete representative layers | reduce bytes and dispatches per token; keep state in unified memory; speculate only when acceptance pays |
| A113X / Qwen3.5-0.8B | 470 MiB image | 3.591 GiB/s four-thread read probe | NEON GEMV, contiguous recurrent state, static head partition and zero swap |
| A113X / Whisper small.en | mixed Q4/Q8 image | 2 GB system RAM | bounded working memory, compact audio windows and cached decoder state |

The compiler therefore treats DIMM/channel bandwidth, unified-memory behavior, cache topology and storage speed as target inputs—not incidental machine details.

## Build and run

Run the committed tests:

```sh
make test
```

Build the Qwen3.8 Apple M3 Pro runtime and image tools:

```sh
make qwen38-m3-chat qwen38-tools qwen38-mtp-pack
```

After compiling the pinned checkpoint as described in the [target guide](models/qwen3.8-27b/targets/apple-m3-pro/README.md), start a resident terminal chat:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_chat.sh --terminal
```

Or start the OpenAI-compatible server:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_serve.py
# http://127.0.0.1:8199/v1
```

Model execution stays in the resident C/Metal process. The Python server is an optional standard-library HTTP adapter and is not part of the inference path.

## Good places to contribute

| Area | Concrete work |
|---|---|
| MiniMax-H3 / Apple Silicon | quality-gate hierarchical H3 attention and sparse projections against the corrected 480p dense trajectory; then offline-pack 64×32 VAE weight tiles |
| Qwen3.8 / Apple Silicon | fused batched prefill, DeltaNet scheduling, attention kernels, sampling and streaming overlap |
| Low-cost Arm CPUs | NEON kernels, recurrent-state layout, cache-aware thread partitioning and memory probes |
| New model support | add a model directory, independent numerical oracle, packed format and generic C runtime |
| New hardware support | add one target directory with an exact machine profile, specialized kernels and on-device measurements |
| Verification | expand layer-boundary parity, held-out quality suites, RSS/page-fault accounting and reproducible benchmark reports |

Performance patches need a correctness gate and raw before/after measurements. Approximate changes must record their quality effect; smoke examples are never presented as product-quality evaluation.

## License

No license has been selected yet.
