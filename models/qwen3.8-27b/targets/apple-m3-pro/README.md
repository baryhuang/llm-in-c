# Qwen3.8-27B on Apple M3 Pro

This target runs Qwen3.8-27B text generation end to end through a model-specific C runtime and Metal kernels. It supports batched prompt prefill, incremental conversation state, FP16 KV cache, greedy and sampled decoding, adaptive MTP speculative decoding and streaming output.

## Target and artifact

| Property | Pinned value |
|---|---|
| Machine | MacBook Pro `Mac15,6` |
| SoC | Apple M3 Pro, 11-core CPU, 14-core Metal 3 GPU |
| Memory | 36 GB unified memory |
| Operating system | macOS 15.7.3 |
| Weight format | affine Q4, group size 64 |
| Compiled text image | 65 mapped files, 15,138,643,968 bytes, plus tokenizer |
| MTP images | 209,436,672-byte draft layer and 29,556,736-byte projection/norm image |
| Runtime state | 158.9 MB recurrent/convolution state; FP16 KV uses 64 KiB per context token |

The target maps immutable layer images rather than loading a general model graph. Forty-eight DeltaNet layers use persistent recurrent/convolution state; sixteen full-attention layers use grouped-query KV cache. Metal kernels are specialized for the fixed matrix shapes, affine Q4 layout and prompt-length buckets used by this graph.

## Source layout

| Path | Contents |
|---|---|
| Target directory root | C/Objective-C runtime, image-format headers and Metal kernels |
| [`commands/`](commands/) | Chat and one-shot generation front ends compiled into `build/` |
| [`benchmarks/`](benchmarks/) | MLP, DeltaNet, complete-layer and attention microbenchmarks |
| [`validation/`](validation/) | Token-ID decode, tokenizer and oMLX numerical-export programs |
| [`compiler/qwen3.8-27b/apple-m3-pro/`](../../../../compiler/qwen3.8-27b/apple-m3-pro/) | Offline checkpoint inspection and runtime-image packers |

## Compiled image format

| File | Contents |
|---|---|
| `global.q38global` | embedding, final normalization and output projection |
| `layer-NN.q38delta` | one DeltaNet transformer layer |
| `layer-NN.q38att` | one full-attention transformer layer |
| `tokenizer.q38tok` | packed vocabulary, merges and special-token tables |
| `mtp-layer.q38att` | MTP draft attention layer |
| `mtp.q38mtp` | MTP input projection and normalization tensors |

Every packer checks the expected source SHA-256 before writing an image. Every image carries its format magic, version, source hash and tensor metadata; the runtime rejects the wrong model or format instead of attempting compatibility fallback.

## Build the runtime

Apple Metal command-line tools and a C11/Objective-C compiler are required.

```sh
make qwen38-m3-chat qwen38-tools qwen38-mtp-pack
```

## Compile the model images

Download the exact revisions listed on the [model page](../../README.md), then compile the three weight shards and tokenizer:

```sh
compiler/qwen3.8-27b/apple-m3-pro/qwen38_compile_runtime_images.sh \
  model-00001-of-00003.safetensors \
  model-00002-of-00003.safetensors \
  model-00003-of-00003.safetensors \
  tokenizer.json /path/to/qwen38-runtime
```

Pack the standalone MTP checkpoint into the same directory:

```sh
build/qwen38-m3-attention-pack \
  /path/to/qwen38-mtp/model.safetensors \
  /path/to/qwen38-runtime/mtp-layer.q38att \
  76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6 64

build/qwen38-mtp-pack \
  /path/to/qwen38-mtp/model.safetensors \
  /path/to/qwen38-runtime/mtp.q38mtp \
  76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6
```

## Run

Resident terminal chat:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_chat.sh --terminal
```

One prompt:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime \
  tools/qwen38_chat.sh 'Explain why virtual memory uses pages.'
```

OpenAI-compatible server:

```sh
QWEN38_MODEL_DIR=/path/to/qwen38-runtime tools/qwen38_serve.py
# base URL: http://127.0.0.1:8199/v1
```

The server accepts standard sampling fields. `reasoning_effort` values `low`, `medium` and `xhigh` enable thinking mode; `none` selects the no-thinking template. Streaming responses separate reasoning into `reasoning_content` and the final answer into `content`. The Python process implements only HTTP and template adaptation; inference stays inside the resident C/Metal process.

## Codex CLI

`tools/codex-qwen` runs the OpenAI Codex CLI against this runtime instead of a
hosted model. It starts the server if needed and then execs
`codex -p qwen`, which layers `~/.codex/qwen.config.toml` over the user
config. Every argument is forwarded, so `codex-qwen`, `codex-qwen exec '...'`
and `codex-qwen resume --last` all work.

```sh
codex-qwen                                    # interactive
codex-qwen exec 'fix the bug in calc.py'      # one shot
```

Codex 0.149 dropped `wire_api = "chat"` and speaks only the Responses API, so
`qwen38_serve.py` also serves `POST /v1/responses`. The shim renders a
Responses request into the same Qwen3.8 chat template - including the
template's own `<tool_call><function=...><parameter=...>` block - and turns
the reply back into the SSE events Codex consumes: `response.created`,
`response.output_text.delta`, `response.reasoning_text.delta`,
`response.output_item.added`/`.done` and `response.completed`. Codex ignores
`response.function_call_arguments.delta`, so each tool call is delivered whole
in one `output_item.done` carrying a `function_call` item. Both APIs share one
resident engine, so Chatbox and Codex can point at the same server.

Rendering is deliberately append-only: instructions and leading developer
messages become the system turn, assistant and `function_call` items replay as
assistant turns, and `function_call_output` items become `<tool_response>`
blocks. Each Codex turn is therefore an exact token extension of the previous
one, and the engine's conversation continuation prefills only the new suffix.

| Turn | Prompt tokens | Reused | Time to first token |
|---|---|---|---|
| Cold session start | 10,416 | 0 | 234.9 s |
| Tool result returned | 11,249 | 10,915 | 11.6 s |
| Second tool result | 11,380 | 11,249 | 5.1 s |
| Final answer | 11,482 | 11,428 | 2.9 s |

Measured on the pinned machine with `QWEN38_CONTEXT=32768`, decoding at
12-23 tok/s. The cold turn is prefill-bound at roughly 44 tok/s, so it is paid
once per session; every later turn costs only its own new tokens. A four-turn
edit task - read the file, patch it, verify, answer - completed in about five
minutes, four of which were that first prefill.

The profile exists to keep that first prefill survivable. A stock Codex turn
sends about 60,000 tokens of tool schema alone (GitHub 76 KB, Gmail 33 KB and
Sites 30 KB of JSON), so `~/.codex/qwen.config.toml` disables apps, plugins,
MCP servers, web search and subagents, leaving 12 tools and 10.4 k tokens. It
also sets `model_context_window` and `model_auto_compact_token_limit` so Codex
compacts before the runtime refuses the prompt, and a 15-minute
`stream_idle_timeout_ms` because a cold prefill outlasts the default.

Thinking is off by default: at this token rate a reasoning block per tool call
is not affordable. Set `QWEN38_RESPONSES_EFFORT=1` on the server to honor the
request's `reasoning.effort` instead.

## Verification

Verification is separate from timed benchmarking.

| Gate | Recorded result |
|---|---|
| Source integrity | Three weight shards, tokenizer and MTP checkpoint SHA-256 checked at pack time |
| Image identity | 68/68 runtime files use Qwen3.8 names and `.q38*` formats; sampled headers report `Q38M3GLB`, `Q38TOK1`, `Q38M3Q4`, `Q38M3ATT` and `Q38M3MTP` |
| Async decode API | 16/16 state-machine checks pass |
| Batched prefill parity | 36/36 checks pass across 12 lengths from 16 to 131 tokens and three execution modes; argmax identical and no NaN |
| Basic generation | `2+2` → `4`; C `max2` implementation correct; Chinese capital-of-France prompt → `巴黎` |
| MTP image smoke | `one` through `ten` generated correctly; 19 visible tokens at 13.0 decode tok/s, with 13 drafts accepted over 7 steps |
| Speculative decoding | Default fast verify token-identical to plain greedy on 3/5 throughput cases; prose and essay each flip one numerical near-tie onto an alternate fluent greedy continuation; `QWEN38_VERIFY_FAST=0` selects the exact verify with 5/5 token identity |
| Thinking stream | Reasoning and answer split verified; `reasoning_effort: none` reproduces no-thinking behavior |
| ARC-Easy five-case adaptation | 3/5 strict at a 32-token budget, 4/5 at 96; misses contain the correct content but do not complete the required `Answer: X` format |

The ARC-Easy record is a small generative smoke adaptation, not an official ARC-Easy score. Raw outputs are in [`../../benchmarks/arc-easy-5/results-macos-m3-pro.json`](../../benchmarks/arc-easy-5/results-macos-m3-pro.json).

## Measured throughput

The following five workloads ran through one resident chat process per arm with greedy seed 42. End-to-end throughput is completion tokens divided by the full request wall, including prompt prefill, first token and decode. The speculative arm uses adaptive MTP with replay-free partial accepts, a wide-tile eight-wide half-MMA verify, a depth-7 chain ceiling, four-gram context-lookup drafting and a restricted draft-head vocabulary.

| Case | Output tokens | Plain end-to-end | Adaptive MTP end-to-end | Speedup | Request wall, plain / MTP |
|---|---:|---:|---:|---:|---:|
| C `max2` function | 28 | 6.36 tok/s | **15.01 tok/s** | 2.36× | 4.4 / 1.9 s |
| Hash-table explanation | 531 / 524 | 8.19 tok/s | **10.84 tok/s** | 1.32× | 64.8 / 48.3 s |
| Python `LRUCache` class | 1,155 | 7.98 tok/s | **17.59 tok/s** | 2.20× | 144.7 / 65.7 s |
| Virtual-memory essay | 1,463 | 8.01 tok/s | **9.94 tok/s** | 1.24× | 182.7 / 147.2 s |
| Notes summary, 159-token prompt | 128 | 6.99 tok/s | **9.87 tok/s** | 1.41× | 18.3 / 13.0 s |
| **Aggregate** | **3,305 / 3,298** | **7.96 tok/s** | **11.95 tok/s** | **1.50×** | **415.0 / 276.0 s** |

Aggregate decode throughput is 8.06 plain and 12.32 speculative. Code-heavy cases run at 18.0–40.8 decode tok/s because their draft chains accept deeply; prose cases sit near 10.0–11.0, acceptance-limited. Four cases are token-identical between arms; the paired token count marks the prose case where the fast verify's accumulation order flips one numerical near-tie onto an alternate fluent greedy continuation. `QWEN38_VERIFY_FAST=0` selects the exact verify kernels, which are token-identical on all five cases at 9.81 aggregate end-to-end tok/s.

## llama.cpp comparison

A separate comparison used the same rendered 36-token prompt and 28-token greedy completion with both models resident. Each stack received one warmup followed by four measured requests. llama.cpp build 10360 used `unsloth/Qwen3.8-27B-GGUF` Q4_K_M, which is a different four-bit encoding from this runtime's affine Q4 checkpoint.

| Metric | This runtime, plain | This runtime, adaptive MTP | llama.cpp + Q4_K_M |
|---|---:|---:|---:|
| End-to-end throughput | 6.29 tok/s | **9.01 tok/s** | 5.76 tok/s |
| Request wall | 4.450 s | **3.109 s** | 4.864 s |
| First token | 1.023 s | 1.179 s | not recorded in the raw comparison |
| Decode throughput | about 8.4 tok/s | included in speculative schedule | 7.11 tok/s |

The comparison establishes end-to-end behavior on this machine; it does not establish equal quantization quality. Full source pins, per-round timings and definitions are in [`results.json`](results.json), with a human-readable view in [`REVIEW.html`](REVIEW.html).

## Apple stack comparison

The llama.cpp row above is not the strongest local stack on this machine. The
same five workloads were run through mlx-lm 0.31.3 / mlx 0.32.0 and oMLX 0.5.7
on 2026-08-18, both resident, greedy, one warmup first, against
`mlx-community/Qwen3.8-27B-4bit` — the same checkpoint this runtime compiles
its images from, with no requantization on either side. Reply token counts
differ between stacks because the completions differ; this compares
throughput, not tokens.

| Workload | mlx-lm | oMLX | This runtime, plain | This runtime, adaptive MTP | MTP vs best MLX |
|---|---:|---:|---:|---:|---:|
| Code, short | 6.29 | 6.17 | 6.36 | **15.01** | 2.39× |
| Prose | 8.08 | 8.12 | 8.19 | **10.84** | 1.33× |
| Code, long | 8.36 | 8.30 | 7.98 | **17.59** | 2.10× |
| Essay | 8.46 | 8.45 | 8.01 | **9.94** | 1.17× |
| Summary | 7.26 | 7.33 | 6.99 | **9.87** | 1.35× |
| **Aggregate** | **8.29** | **8.28** | 7.96 | **11.95** | **1.44×** |

All values are end-to-end tokens/s: reply tokens over the full request wall,
including prefill. Aggregate walls are 398.1 s for mlx-lm, 398.0 s for oMLX,
415.0 s plain and 276.0 s speculative.

Without speculation this runtime is **0.96×** the best Apple stack, not ahead
of it. That result is expected rather than disappointing: 15.139 GB of mapped
weights over the 117–118 GB/s measured on this machine is a first-order
ceiling near 7.8 tokens/s for any decoder that reads every weight once per
token, and mlx-lm (8.29), oMLX (8.28) and the plain arm (7.96) all sit within
6% of it. Hand-written kernels cannot move a bound set by weight traffic.
Adaptive MTP clears it by amortizing one weight pass over several accepted
tokens, which is where the 1.44× aggregate and 2.10× long-code margins come
from.

Raw per-case output is in [`../../../../tools/compare/set-mlxlm-38.json`](../../../../tools/compare/set-mlxlm-38.json)
and [`set-omlx-38.json`](../../../../tools/compare/set-omlx-38.json); the
merged record is `apple_stack_comparison` in [`results.json`](results.json).
