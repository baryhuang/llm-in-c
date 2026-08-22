# MiniMind-O on Amlogic A113D/A113X

This target is the first native-C MiniMind-O speech-to-speech prototype for
the ThirdReality TRHub-V3. The board reports `amlogic,a113d` and
`amlogic,meson-axg`; its application CPU is four Arm Cortex-A53 cores. The
runtime does not embed or launch Python.

## Resident pipeline

```text
USB microphone, 16 kHz PCM
        |
        v
adaptive RMS VAD + pre-roll + bounded capture queue
        |
        v
online SenseVoice chunks --> committed 768-wide audio embeddings
        |
        v
incremental Thinker/Talker prompt prefill --> response generation
        |                                      |
        +-- while the user is still talking    +--> 8 Mimi codebooks/frame
                                      |
                                      v
                          stateful Mimi decoder worker
                                      |
                                      v
                          first 80 ms PCM frame
                                      |
                                      v
                        raw 24 kHz PCM --> ALSA immediately
                                      |
                                      +--> response WAV archive
```

All model stages are resident. Capture runs in a dedicated pthread so ALSA is
drained during inference and playback. The 64 x 512-sample bounded SPSC queue
discards new echo while it is full and is flushed after a turn; stale speaker
audio is therefore not interpreted as a new command.

## Input-token streaming

Input streaming starts at `speech_start`, not at `speech_end`. The VAD thread
publishes pre-roll and each confirmed speech chunk to a 512-entry SPSC PCM
ring. A dedicated input-inference thread owns SenseVoice, Thinker and Talker
for that phase. It first prefills `<|im_start|>user\n`, then immediately
injects each committed SenseVoice embedding at an `<|audio_pad|>` position in
both language-model KV caches. End-of-speech only flushes the encoder's final
right-context tail and prefills the assistant suffix. Ownership transfers to
the generation thread only after the input thread is joined; model state is
never accessed concurrently.

The frozen SenseVoice graph was trained with full-sequence attention and an
11-tap bidirectional FSMN, so an offline prefix embedding is not final until
future audio exists. The native online graph follows the chunk/cache structure
of SenseVoice's streaming SANM path:

- 8 LFR center frames are committed per normal chunk (480 ms);
- 4 LFR frames are retained as right lookahead (240 ms);
- every one of the 70 layers retains 32 committed K/V frames (1.92 s);
- right-context frames are recomputed in the next chunk and are never entered
  into the cache until committed;
- the final chunk commits all remaining frames at input EOS.

This is truncated-attention inference for a frozen bidirectional checkpoint,
not claimed to be bit-equivalent to the old whole-utterance graph. The runtime
validates that incremental prefix, repeated audio-pad positions, and suffix
exactly equal the tokenizer output for the complete prompt before generation.
Potential trailing silence is held outside the model; it is enqueued only if
speech resumes and discarded at VAD EOS, because a committed model token
cannot be retracted.

Production logs expose the actual overlap. `input_audio_commit` is printed
when embeddings become stable, `input_prefill` after their Thinker/Talker KV
positions advance, `speech_end` when VAD closes the utterance, and
`input_caught_up` when the tail is complete. A real overlap requires at least
one `input_prefill` line before `speech_end`; `input_streaming=always` and the
chunk/cache geometry are also printed in the READY line.

## Stateful Mimi streaming

`minimindo_mimi_stream_decode` accepts exactly one newly generated codec frame
per call. Its
state contains:

- Transformer key/value cache at the global Mimi position;
- two pending samples per channel from the semantic 2x transposed upsampler;
- `kernel - 1` inputs for every causal convolution;
- one previous input sample per channel for every causal transposed
  convolution.

All four Mimi transposed convolutions have `kernel == 2 * stride`. For phase
`r`, output step `t` is the dot product of the current input against kernel
phase `r`, plus the previous input against phase `stride + r`. The loader
repacks those two discontiguous Q8 phase rows into one contiguous row. The hot
path then uses the same NEON Q8 dot kernel as GEMV and retains only the previous
input vector. One-frame streaming preserves the causal state without
recomputing a prefix; numerical comparison against older scalar accumulation
orders is documented under Correctness gates.

The runtime has no batch/non-streaming mode. It always creates the stateful
Mimi consumer before Talker generation. Talker publishes each complete
eight-codebook frame immediately; Mimi consumes it on CPU3 while the Talker
caller runs on CPU0. Both dispatchers share persistent workers on CPUs1--2
through a small FIFO. After producer EOS, Mimi moves to CPU0 and uses all three
workers. The overlap is mandatory pipeline semantics rather than an optional
flag.

Every frame is observable in the production journal. `talker_produce` records
queue insertion, `mimi_decode` records PCM publication, and `alsa_write`
records delivery to the ALSA pipe. The final JSON separately reports
`decode_overlapped_with_generation`, `decode_overlap_frames`,
`decoder_to_alsa_streaming`, and `end_to_end_streaming`; it no longer calls all
of these distinct behaviors `playback_streaming`.

Production is always token-to-code-to-PCM incremental. The decoder release-
publishes each 1,920-sample/80 ms PCM frame and speaker delivery starts on
frame one. There is no producer-EOS gate, frame watermark, complete-response
gate, or alternate continuity mode. The earlier two-, eight-, 32-frame and
complete-response experiments are retained only as rejected diagnostics.
Underruns must be fixed by sustained model/codec throughput rather than by
hiding first-audio latency. The response WAV is an archive and is not on the
playback path.

The runtime therefore reports `decode_overlapped_with_generation=true`,
`decoder_to_alsa_streaming=true`, `playback_fully_buffered=false`, and
`end_to_end_streaming=true` when the corresponding events occur before
producer EOS. Mimi state always advances one frame per call.

Inference and model execution are native C11 and launch no Python. `aplay`
remains the small external ALSA transport process. `SIGPIPE` is ignored and
ALSA/write failure is reported instead of terminating the service.

## Thread ownership and lifetime

The production lifecycle has explicit single-writer ownership:

- the capture pthread exclusively enqueues ALSA chunks; VAD exclusively
  dequeues them;
- the input-inference pthread exclusively owns SenseVoice and Thinker/Talker
  state while PCM is arriving; after EOS/join, the main thread exclusively
  owns the same Thinker/Talker state and is the only Mimi-code producer;
- the Mimi pthread exclusively owns the stateful decoder and writes each unique
  PCM range before release-publishing `decoded_frames`;
- the playback pthread reads only published PCM ranges and never touches Mimi
  state;
- every data queue is SPSC and its payload/index handoff is release/acquire;
- mutexes exist only at an empty-queue dequeue sleep / enqueue notification
  boundary. No mutex covers model state, KV/history, memcpy, compute or I/O;
- the main thread signals producer completion, joins playback, joins Mimi, and
  only then destroys queue wait objects and frees PCM.

OpenMP and `libgomp` are absent. Three pthread workers are created once and
pinned to CPUs1--3. Concurrent Talker and Mimi dispatchers enqueue row/tile
jobs into one bounded FIFO; a single atomic spin lock covers only enqueue and
dequeue. Compute runs after the lock is released. Per-dispatch completion is an
atomic counter, and reference-counted sessions prevent a lost wakeup during the
producer/decoder handoff. Workers futex-sleep only when no active session owns
them.

## Cortex-A53 kernels

The packed images use row-wise Q8 weights with an f32 scale. The hot kernels
use Armv8.0 NEON, not dot-product or i8mm instructions unavailable on the
Cortex-A53:

- Q8 x f32 GEMV widens int8 values through int16/int32 and performs four f32
  FMAs per vector;
- causal convolution builds each time window once in contiguous im2col order,
  then reuses it across output channels as a long NEON Q8 x f32 dot product;
- transposed convolution packs `(current, previous)` phase weights and uses a
  contiguous NEON Q8 dot instead of scalar/scatter output updates;
- the persistent worker pool partitions matrix rows, convolution output/time
  tiles and activation ranges into short FIFO jobs shared by Talker and Mimi;
- streaming Mimi causal/deconvolution windows are activation-quantized to i8,
  then evaluated with NEON W8A8 integer dots and per-window/per-row scales;
- Mimi retains persistent activation, convolution, quantization and attention
  scratch arenas, so its per-frame hot path performs no allocation;
- Mimi RoPE is precomputed, QK/AV attention is NEON-vectorized across
  layer/head tasks, and a paired-position dot loads each Q8 row once for the
  two transformer positions generated by one audio frame;
- production keeps the checkpoint's full 250-position sliding window. The
  64-position experiment changed long-form PCM while saving only 0.70 seconds
  over a 128-frame/10.24-second sequence;
- the online SenseVoice dense layers likewise quantize each input position to
  symmetric i8 and evaluate four positions per weight load with Armv8.0 NEON
  W8A8 dots; the projector remains Q8 x f32.

Thinker and Talker prefill now run layer-by-layer across the full prompt. This
reuses each Q8 weight row across all prompt positions, computes the Thinker LM
head only at the final prompt token, skips all Talker output heads during
prefill, and computes the fixed pad-codec projection once instead of once per
prompt token. RoPE sine/cosine values are precomputed at model load. For a
37-token prompt, prefill fell from 2.328 seconds to 1.632--1.732 seconds with
identical text and WAV output.

SenseVoice uses a four-position Q8 dot kernel: one weight row feeds four audio
positions before it is evicted. The 20-frame target A/B was 2.31--2.44 seconds
versus 2.52--2.62 seconds for the single-position kernel, with byte-identical
embeddings. Talker top-50 sampling uses a 50-entry min-heap followed by a
50-item sort instead of sorting all 2,112 logits. It removes about 66 ms of
serial work per 24-step fixture and preserves the exact WAV hash.

The input-streaming increment replaces SenseVoice's Q8 x f32 dense products
with per-position W8A8 products. On the pinned 18,880-sample fixture, the
streaming encoder CLI fell from 3.04 s to 1.11 s (2.74x); the measured dense
encoder section fell from 2.392 s to 1.134 s (2.11x). Both runs used all four
cores with 337--365% aggregate CPU and about 228 MiB peak RSS. Quantizing the
shared normalized input once for Q/K/V preserved the W8A8 output byte for byte
and accounts for the final 1.15 to 1.11 s reduction. The W8A8
embeddings have cosine 0.998620222 against the Q8 x f32 online graph, RMSE
0.029285131 and maximum absolute error 0.138373837. This is a numerical gate,
not a language-quality claim; live multilingual response quality remains a
required physical acceptance test.

The im2col change trades about 11 MiB of temporary memory on the eight-frame
fixture for contiguous vector access and reuse. It reduced Mimi decode from
7.80 to 1.89 seconds. The phase-packed transposed-convolution kernel then
reduced one-frame streaming decode from 1.93 to 1.26 seconds. The subsequent
stream-attention, paired-dot and persistent-arena work reduced the same
eight-frame test to 0.72 seconds at 318% aggregate CPU while restoring the
full 250-position attention window. Pairing two safe int8 products in int16
and evaluating four transposed-convolution phases per shared activation load
then reduced the isolated eight-frame result to 0.62--0.69 seconds without
changing the PCM SHA256.

## Correctness gates

The pinned eight-frame fixture contains all eight codebooks and produces
15,360 samples at 24 kHz. The first table is the historical phase-packed
convolution gate:

| Gate | Result |
|---|---:|
| Phase-packed whole decode vs prior im2col WAV | 16 differing bytes out of 30,720 PCM bytes |
| Phase-packed one-frame stream vs phase-packed whole decode | 11 differing bytes out of 30,720 PCM bytes |
| Whole-decode RMS | 0.0925718284 |
| Streaming RMS | 0.0925718293 |
| Whole-decode peak | 0.710567594 |

The byte differences are f32 accumulation-order rounding at roughly one PCM
least-significant bit. Frame count, sample count and audible waveform are
preserved.

The final SIMD attention rewrite changes f32 accumulation order more broadly:
against the pre-attention scalar fixture its difference signal was -49.578 dB
RMS while the reference signal was approximately -20.6 dB RMS (about 29 dB
signal-to-difference ratio). This is not claimed to be byte parity. Arena
reuse and the paired Q8 dot were byte-identical relative to that optimized
attention build. The final full-window artifact has these reproducible gates:

| Final gate | Result |
|---|---:|
| 8 frames / 0.64 s PCM | 0.62--0.69 s after exact NEON dot reuse |
| 8-frame WAV SHA256 | `94b17f8a63fc6c7d3a70bc2266649fd87d0b98641ae2bb0261dd2451da01208f` |
| 128 frames / 10.24 s PCM, 250-position window | 10.69 s, 365% CPU, 67,712 KiB RSS |
| Same 128 frames, 64-position window | 9.99 s, 364% CPU, 67,584 KiB RSS |
| 250- and 64-window WAV hashes equal | no |

Because the smaller context saved only 6.5% and changed long-form output,
production uses the checkpoint's full 250-position window. Physical
multilingual listening remains a separate model-quality gate.

## A113D measurements

The raw record is in [results.json](results.json). Important measured values:

| Test | Wall | CPU | Peak RSS | Relative |
|---|---:|---:|---:|---:|
| Historical full 16-step speech, 1 OpenMP thread | 40.58 s | 98% | 391,448 KiB | 1.00x |
| Historical full 16-step speech, 4 OpenMP threads | 13.64 s | 307% | 391,080 KiB | 2.98x |
| Mimi 8-frame whole decode, old convolution layout | 7.80 s | 347% | 59,056 KiB | 1.00x |
| Mimi 8-frame whole decode, NEON/im2col | 1.89 s | 347% | 70,300 KiB | 4.13x |
| Mimi 8-frame, one frame per streaming call | 1.93 s | 347% | 48,700 KiB | 4.04x |
| Mimi 8-frame, phase-packed one-frame streaming | 1.26 s | 318% | 58,844 KiB | 6.19x |
| Mimi 8-frame, final full-window streaming | 0.62--0.69 s | about 318% | about 59 MiB | 11.3--12.6x |
| Mimi 128-frame, final full-window streaming | 10.69 s | 365% | 67,712 KiB | 0.96x real time |
| Online SenseVoice, Q8 x f32, 18,880 samples | 3.04 s | 337% | 228,568 KiB | 1.00x |
| Online SenseVoice, W8A8, 18,880 samples | 1.11 s | 365% | 228,580 KiB | 2.74x |

The historical four-thread A/B uses the same input, seed, 16 generation steps, eight
audio frames and identical answer text. Four cores are genuinely used; the
remaining gap from 400% comes from serial sampling/attention sections, short
parallel regions and shared L2/memory bandwidth.

The current no-OpenMP, queue-only-lock build was measured on a deterministic
Chinese prompt (`seed=20260821`, 50 total steps, 40 audio frames):

| Metric | Result |
|---|---:|
| Full model/decode wall | 8.23 s |
| Aggregate CPU | 358% |
| Peak RSS | 212,456 KiB |
| Prefill | 0.910 s / 293% |
| Generation plus overlapping Mimi | 4.021 s / 373% |
| Four-core Mimi drain | 3.120 s / 367% |
| Frames decoded before Talker EOS | 11 |
| Voluntary context switches | 12 |
| Involuntary scheduler preemptions | 6,911 |
| WAV SHA256 | `b61b0662379fa0bb6b3ec304b72bc53000f8e8457649b81bc611f73c48f7289c` |

The WAV hash is identical to the prior mutex-backed queue build. The 12
voluntary switches cover session wake/sleep and process lifecycle rather than
per-matrix dispatch. Generation reaches 373% process CPU because the
three-thread Thinker/Talker group and the one-thread Mimi consumer occupy all
four cores concurrently.

The earlier 1.18-second input / 24-step / 16-frame fixture gives a useful
historical distribution of the full response path:

| Stage | Wall | Process CPU / wall | Approximate work |
|---|---:|---:|---:|
| SenseVoice + audio projector | 2.275 s | 286% | 4.4B MAC at 20 encoder frames |
| Thinker/Talker batch prefill | 1.769 s | 308% | 2.18B + 1.24B MAC |
| Thinker/Talker generation | 2.147 s | 244% | about 2.2B MAC plus sampling |
| Mimi drain | 2.366 s | 288% | about 166M MAC per output frame |
| Model/decode total | 8.686 s | — | no swap; 427,492 KiB peak RSS |

All four CPUs are active, but no stage sustains 400%. The short transformer
regions have serial attention/sampling gaps, and SenseVoice/Mimi stream large
Q8 weights through the A53 cluster's shared L2. This is why merely increasing
the worker count cannot produce 1--3 second latency. With the exact
current graph, the pre-audio floor is already approximately 2.3 s SenseVoice +
1.7 s prefill + 2.1 s generation + 1.8 s to the safe Mimi buffer.

The optimized Mimi profiler for eight frames reports:

| Stage | Time |
|---|---:|
| Codebook projections | 6.783 ms |
| Upsampler + Mimi Transformer | 268.359 ms |
| Initial causal convolution | 27.689 ms |
| Decoder stage 1 | 147.704 ms |
| Decoder stage 2 | 310.648 ms |
| Decoder stage 3 | 549.765 ms |
| Decoder stage 4 | 521.444 ms |
| Final convolution | 44.715 ms |
| Total | 1,877.109 ms |

The profile above predates the phase-packed transposed-convolution, persistent
arena, paired-dot and SIMD attention kernels. Final four-core Mimi is near
real time over a 10.24-second stream (10.69 seconds wall), but the one-core
overlap lane is still about 3x slower than playback while Thinker/Talker own
the other cores. The complete system therefore still does not meet a 1--3
second conversational latency target.

Earlier 48-step resident turns took 18.916--21.405 seconds and are retained in
the raw result file as historical measurements. A 24-step fast-path experiment
reduced latency, but was not production-correct: the same limit bounds both
text generation and the staggered eight-codebook audio stream, so every turn
stopped at 16 Mimi frames (1.28 seconds) even when the decoded text was longer.
The listener therefore heard only the first few words. Production now ends
text at the first complete sentence, with a 32-step fallback limit, then gives
the staggered audio codebooks a separate 192-step/15.36-second drain budget.
It exits immediately when the next complete interleaved eight-codebook frame
is invalid/EOS; it no longer waits for eight independent EOS observations.
In a fixed-seed Chinese test text reached EOS at step 10, while audio
correctly continued until total step 56 and emitted 44 frames/3.52 seconds. A
forced-limit test capped text at 16 steps; it inserted text EOS at step 17 and
continued to total step 75, producing 61 frames/4.88 seconds with
`text_limit_hit=true` and `audio_drain_complete=true`. The prior 24-step speaker
run remains useful as a rejected buffering measurement: model generation
ended at 6,271 ms and playback began with 12 buffered frames at 8,044 ms. The
later full-window buffering experiment produced 88 frames/7.04 seconds and
decoded 34 frames before Talker EOS, but delayed speaker delivery until 14,849
ms. Neither policy is production. The complete historical trace is in
[`benchmarks/v1.3.0-continuous-playback.log`](benchmarks/v1.3.0-continuous-playback.log).

An earlier OpenMP `3 generation threads + 1 overlapping Mimi thread` A/B was
rejected because repeated team entry and passive wakeups raised generation to
8.161 seconds and total model time to 15.258 seconds. The next fixed 3+1
pthread partition was also rejected because one-core Mimi needed 220--310 ms
per 80 ms frame. Production now runs Talker on CPU0 and Mimi on CPU3 while both
share CPU1/2 workers through the minimal-lock FIFO. Thinker audio drain advances
one bridge per Talker step; a 16-step batch was rejected because it created
0.7--0.9 second code-arrival gaps.

With NEON W8A8 generation and the shared pool, the fixed English prompt
produced 28 PCM frames, decoded 27 before producer EOS, and completed the model
path in 4.655 seconds. Once audio generation started, Talker/Mimi cadence was
mostly 100--125 ms per 80 ms frame. This is materially closer to real time but
still lacks worst-case underrun headroom; production preserves first-frame
playback and exposes the remaining deficit in per-frame logs.

The model is therefore usable as a correctness prototype, but it is not the
1--3 second immediate-dialog path. The product design should keep two paths:
an aggressively bounded tiny safety/intent model for immediate local response,
and this higher-accuracy transcript/reasoning path for notification decisions.
Further exact-graph work should first reduce SenseVoice weight traffic and
autoregressive Thinker/Talker cost; a separate tiny model is required to
change the latency class.

A live 7.52-second utterance demonstrated why the paths must be bounded:
SenseVoice took 63.709 seconds, 142-token prefill took 12.477 seconds, and first
audio arrived at 83.126 seconds. The fast resident service now closes a turn at
approximately three seconds even without detected silence and records
`end=limit`; silence-terminated turns record `end=silence`. Longer recordings
must be handed to the accurate asynchronous path instead of occupying the
interactive process.

## Service and monitoring

The deployed unit is `threehub-minimindo-native.service`. Follow activity with:

```sh
ssh root@100.123.75.40 \
  'journalctl -fu threehub-minimindo-native.service'
```

The unit sets the P10S microphone capture gain to 50% / +8 dB before launch.
At the device's 0 dB setting the endpoint delivered all-zero PCM; +8 dB
produced an idle RMS around 50--70 and speech peaks above 2,000 while leaving
speaker volume unchanged. It also sets `PCM` playback to 5% and explicitly
unmutes it. Although the P10S ALSA descriptor reports raw playback value zero
as 0 dB and switched on, the physical output was silent at that value. A
native 48 kHz stereo endpoint test proved that USB frames were advancing, and
audible playback resumed as soon as the raw control moved to 5%. Keep this
non-zero initialization in the service so USB resets and reboots cannot leave
the endpoint silently accepting samples.

Successful native turns include per-frame `STREAM` events and stage fields
`audio_encode_ms`,
`prefill_ms`, `generate_ms`, `mimi_drain_ms`, `first_audio_ms`, and
`streaming_lead_ms`. The four `*_cpu_pct` fields expose effective per-stage CPU
parallelism; values above 100% prove multi-core execution. The production event
order is:

```text
speech_start -> input_audio_commit/input_prefill (capture continues)
speech_end -> input_caught_up -> talker_produce frame=1
           -> mimi_decode frame=1 -> alsa_write frame=1
           -> Talker EOS/model_end -> playback_end -> inference_end
```

`streaming`/`decode_overlapped_with_generation` can be false for a response so
short that Talker finishes before the first Mimi frame completes. The
production invariants are one-frame stateful decode, no batch-mode switch, and
`0 < first_audio_ms < producer_end_ms` whenever a normal response produces PCM
before Talker EOS.

The model directory remains `/dev/shm/minimindo-o-native-v1` because the 6.9
GiB root filesystem had only about 138 MiB free while the six runtime artifacts
require about 377 MiB. `/dev/shm` is erased on reboot. The persistent
`/usr/local/bin/run-minimindo-native-a113x.sh` launcher solves that cold-boot
failure by verifying every artifact against a pinned SHA256 and downloading
only missing or corrupt files. The executable comes from
[`minimindo-native-a113x-v1.3.0`](https://github.com/baryhuang/llm-in-c/releases/tag/minimindo-native-a113x-v1.3.0);
the unchanged packed model images remain pinned to the v1.0.0 release.

Downloads use a per-process `.part` file in `/dev/shm`; the launcher verifies
it before an atomic rename. A truncated or incorrect asset is never executed.
Once all six files pass, the launcher `exec`s the native-C binary with the live
production arguments. The unit retries after 30 seconds if the network is
unavailable. A warm service restart performs local SHA checks and downloads
nothing; a board reboot repopulates the volatile directory automatically. The
full empty-cache systemd test downloaded 377 MiB and reached `READY` in about
78 seconds. A subsequent warm restart verified the same files and reached
`READY` in about 6 seconds, including a 539 ms model warm-up.

To verify/download the release without starting inference:

```sh
MINIMINDO_DOWNLOAD_ONLY=1 \
  /usr/local/bin/run-minimindo-native-a113x.sh
```
