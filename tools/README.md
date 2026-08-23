# User tools

`tools/` contains commands a user can run directly. Model compilers, runtime source, kernel benchmarks and numerical validation programs do not live here.

## Qwen3.8

| Command | Direct use |
|---|---|
| `qwen38_chat.sh` | Run one prompt, start resident terminal chat, or launch the local Chatbox setup |
| `qwen38_serve.py` | Expose the resident C/Metal runtime through an OpenAI-compatible HTTP API |
| `qwen38_monitor.py` | Record CPU, GPU, process footprint and system memory on Apple Silicon |
| `codex-qwen` | Run the OpenAI Codex CLI against the local runtime instead of a hosted model |
| `opencode-qwen` | Run opencode against the local runtime instead of a hosted model |

`codex-qwen` starts the server if it is not already serving a large enough context, then execs `codex -p qwen`; it expects `~/.codex/qwen.config.toml`, and the target README documents both. Link it onto `PATH` with `ln -s "$PWD/tools/codex-qwen" ~/.local/bin/codex-qwen`. `opencode-qwen` works the same way against `~/.config/opencode/qwen.json`.

`support/qwen38_chatbox_config.py` is an implementation detail called by `qwen38_chat.sh`; it is separated because users do not invoke it directly.

## MiniMax-H3

| Command | Direct use |
|---|---|
| `minimax_h3_generate.sh` | Read or accept one prompt and produce synchronized video + stereo audio with the Apple Silicon C/Metal runtime |

The verified default is 128×128, 22 frames at 24 fps. Override it with
`MINIMAX_H3_WIDTH`, `MINIMAX_H3_HEIGHT`, `MINIMAX_H3_FRAMES` and
`MINIMAX_H3_SEED`. Valid frame counts follow H3's `17n+5` alignment. Set
`MINIMAX_H3_TEXT_ENCODER` to a local `text_encoder.safetensors`. Generation is
strictly offline: the command exits if the 28.22 GB conditioner or any local
H3/VAE cache is missing. Network access belongs to a separate explicit weight
installation step, not inference.

The default sampler executes 30 Euler intervals. To select the verified
four-evaluation profile, set `MINIMAX_H3_TURBO_ADAPTER` to an absolute
`file://` path for the pinned v4-600 EMA adapter. The command validates its
size and SHA-256 before starting inference.

## Hardware and comparison

| Path | Direct use |
|---|---|
| `target_probe.c` | Built by `make linux-tools`; records CPU topology, ISA and memory bandwidth for a target |
| `compare/run_all_sets.sh` | Runs the fixed Qwen3.8 workload set through mlx-lm, oMLX and llama.cpp |
| `compare/*.py` | Support programs called by `run_all_sets.sh` |

## Code that is intentionally elsewhere

| Code type | Location |
|---|---|
| Qwen3.8 offline image compiler | [`compiler/qwen3.8-27b/apple-m3-pro/`](../compiler/qwen3.8-27b/apple-m3-pro/) |
| Qwen3.8 runtime implementation | [`models/qwen3.8-27b/targets/apple-m3-pro/`](../models/qwen3.8-27b/targets/apple-m3-pro/) |
| Qwen3.8 compiled command front ends | [`commands/`](../models/qwen3.8-27b/targets/apple-m3-pro/commands/) |
| Qwen3.8 kernel benchmarks | [`benchmarks/`](../models/qwen3.8-27b/targets/apple-m3-pro/benchmarks/) |
| Qwen3.8 numerical/debug validation | [`validation/`](../models/qwen3.8-27b/targets/apple-m3-pro/validation/) |
| Whisper small.en command source | [`models/whisper-small.en/commands/`](../models/whisper-small.en/commands/) |
| Whisper small.en benchmarks and checks | [`benchmarks/`](../models/whisper-small.en/benchmarks/) · [`validation/`](../models/whisper-small.en/validation/) |
| Automated regression tests | [`tests/`](../tests/) |
