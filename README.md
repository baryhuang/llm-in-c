# llm-in-c

Model-specific inference runtimes, offline weight packers, and hardware-specific optimizations for text, speech, and video generation.

The repository contains custom C runtimes, Objective-C/Metal backends for Apple Silicon, and C++ integrations with upstream runtimes. Implementations and measurements are organized by model and target hardware. Checkpoints and packed model weights are obtained or generated separately.

## Repository layout

| Path | Contents |
|---|---|
| [`models/`](models/) | Model documentation, source pins, runtime code, target build scripts, patches, and results |
| [`compiler/`](compiler/) | Offline checkpoint readers, weight/image packers, reference implementations, and fixture generators |
| [`tools/`](tools/) | Chat and HTTP launchers, video generation, voice services, hardware probes, and comparison scripts |
| [`tests/`](tests/) | Python unit tests, C regression tests, and small numerical fixtures |
| [`artifacts/`](artifacts/) · [`output/pdf/`](output/pdf/) | Saved reports and exported reviews |
| [`Makefile`](Makefile) | Builds for local runtimes, packers, benchmarks, and tests; outputs go to `build/` |

Within `models/<model>/`, `targets/generic/` holds shared CPU implementations where present; `targets/<hardware>/` holds specialized code or upstream integration files and target records. Some models also separate command front ends, benchmarks, and numerical checks into `commands/`, `benchmarks/`, and `validation/`.

## Implemented models and targets

The links below contain build procedures, dependencies, recorded outputs, and measurement details. Scope varies by target, from restricted test profiles to resident services.

| Model | Target and implementation | Demonstrated behavior |
|---|---|---|
| Qwen3.8-27B | [Apple M3 Pro](models/qwen3.8-27b/targets/apple-m3-pro/README.md): custom C/Metal | Text generation, resident chat, batched prefill, sampling, adaptive multi-token prediction (MTP), and local HTTP serving |
| Qwen3.8-27B | [Jetson Orin Nano Super](models/qwen3.8-27b/targets/jetson-orin/README.md): patched llama.cpp/CUDA | IQ1_S text generation and serving; recorded interactive turns took 114–156 seconds with substantial swap |
| Qwen3.5-0.8B | [Amlogic A113X](models/qwen3.5-0.8b/targets/a113x/README.md): C11/NEON, shared generic C implementation | Runtime tokenization, caller-supplied answer scoring, and greedy text generation with Q4/Q8 weights |
| MiniMax-H3 | [Apple M3 Pro](models/minimax-h3/targets/apple-m3-pro/README.md): custom C/Metal | Text- and image-conditioned video with audio, including tokenizer, text conditioner, H3 transformer, video/audio VAEs, and MP4 output |
| MiniMind-O | [A113X](models/minimind-o/targets/a113x/README.md) · [RK3588](models/minimind-o/targets/rk3588/README.md): native C/NEON | Resident speech-to-speech prototype with SenseVoice, Thinker/Talker, streaming Mimi decoding, and ALSA capture/playback; A113X output cadence remains slower than real time |
| MOSS-TTS-Nano-100M | [A113X](models/moss-tts-nano/targets/a113x/README.md): patched audio.cpp/GGML | Q8_0 voice cloning and resident synthesis with an incremental KV cache |
| Whisper small.en | [A113X](models/whisper-small.en/targets/a113x/README.md): custom C11/NEON, shared generic C implementation | WAV-to-English transcription with a log-Mel front end and cached decoder; the recorded 11-second sample takes about 45 seconds |
| Whisper large-v3 | [Jetson Orin](models/whisper-large-v3/targets/jetson-orin/README.md): patched whisper.cpp/CUDA | Transcription with fused CUDA kernels, checked against upstream on a pinned 32-file LibriSpeech subset |
| Whisper large-v3-turbo | [Jetson Orin](models/whisper-large-v3-turbo/targets/jetson-orin/README.md): patched whisper.cpp/CUDA; [A113X](models/whisper-large-v3-turbo/targets/a113x/README.md): C11/NEON; [RK3588](models/whisper-large-v3-turbo/targets/rk3588/README.md): C++17/RKNPU2 | Transcription measured on each target; the RK3588 FP16 result covers one audio file |
| Gemma 4 E2B | [Generic CPU](models/gemma-4-e2b/README.md): C11/Q4 | Complete 35-layer text graph for a fixed two-label test profile with compiled inputs |

Target `results.json` files, benchmark records, and HTML reviews preserve the measured workloads and correctness checks. Their results apply to the recorded model, quantization, hardware, and input set.

## Build and run

Run commands from the repository root. Builds and model preparation are target-specific; the guides above document their toolchains and required weights.

### Tests

The default test suite uses a C11 compiler, Make, Python 3, and NumPy:

```sh
make test
```

It runs Python unit tests and C checks for layer fixtures, image parsing, hashing, sampling, threading, and MiniMax-H3 helpers. Small fixtures are committed. Regenerating the MiniMind-O fixture requires PyTorch. Tests using full model weights or Metal execution have separate build targets and require the corresponding model images.

### Qwen3.8 chat on Apple Silicon

Requires a C11/Objective-C compiler and Apple Metal command-line tools. The recorded target is an M3 Pro with 36 GB unified memory.

```sh
make qwen38-m3-generate qwen38-m3-chat qwen38-tools qwen38-mtp-pack
```

Prepare the pinned checkpoint and runtime images using the [compilation guide](models/qwen3.8-27b/targets/apple-m3-pro/README.md#compile-the-model-images), then start terminal chat:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_chat.sh --terminal
```

For the local OpenAI-compatible API, run:

```sh
python3 tools/qwen38_serve.py --model-dir /path/to/qwen38-runtime
# API base URL: http://127.0.0.1:8199/v1
```

Python handles HTTP requests; inference runs in the resident C/Metal process.

## Other tools

| Entry point | Purpose |
|---|---|
| [`tools/minimax_h3_generate.sh`](tools/minimax_h3_generate.sh) | Generate video and audio from a prompt using prepared local MiniMax-H3 weights |
| [MiniMax-H3 ComfyUI nodes](tools/comfyui/minimax_h3_native/README.md) | Text and image input, queueing, and output display for the native runtime |
| [ThreeHub voice tools](tools/threehub-voice/README.md) | Resident Whisper → Qwen → MOSS voice assistant, plus launchers for native MiniMind-O speech services |
| [`tools/qwen38_monitor.py`](tools/qwen38_monitor.py) | Record Apple Silicon CPU, GPU, and memory usage |
| [`tools/compare/`](tools/compare/) | Run fixed Qwen3.8 comparison workloads through mlx-lm, oMLX, and llama.cpp |
| [`tools/target_probe.c`](tools/target_probe.c) | Linux CPU topology, instruction-set, and memory-bandwidth probe, built with `make linux-tools` |

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Bury Huang.
