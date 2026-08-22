# MiniMind-O native streaming: engineering learnings

Date: 2026-08-22
Target: ThirdReality TRHub-V3, Amlogic A113D/A113X, four Cortex-A53 cores,
2 GiB RAM
Runtime constraint: native C11 on the box; no Python process and no OpenMP

## Final solution and measured results

The final P1 prototype is one always-resident native-C service with one
mandatory end-to-end streaming path. Input tokens enter the model while the
user is speaking, every complete Talker code frame enters Mimi immediately,
and the first decoded 1,920-sample/80 ms PCM frame is written to ALSA as soon
as it exists. Waiting for producer EOS, a PCM watermark, or a complete response
is a product violation, not a continuity option:

```text
ALSA capture -> VAD -> PCM SPSC queue -> online SenseVoice/W8A8
                                          -> Thinker/Talker incremental prefill
                                          -> Talker code SPSC queue
                                          -> Mimi one-frame causal decode
                                          -> first PCM frame -> ALSA immediately
                                          -> subsequent PCM frames as produced
```

The implementation makes the following production choices:

- VAD starts the encoder/prefill thread at `speech_start`, not at
  `speech_end`. Stable audio embeddings enter Thinker and Talker while capture
  is still running.
- SenseVoice commits 8 LFR center frames with 4 frames of right context. Each
  of its 70 layers retains a 32-frame K/V cache; the uncertain VAD tail is not
  committed until it becomes speech.
- Cortex-A53 NEON W8A8 kernels and a persistent pthread pool use all four CPU
  cores without OpenMP team creation. Talker on CPU0 and Mimi on CPU3 share
  workers on CPUs1--2 through a tiny compute FIFO; its spin lock covers only
  enqueue/dequeue, never inference work.
- Thinker/Talker state is single-owner. The input worker is joined at EOS and
  ownership is handed to the generation thread; model state has no mutex.
- The data paths are SPSC queues. Release/acquire atomics publish payloads; a
  mutex is used only around an empty-queue condition-variable sleep and its
  enqueue notification.
- Talker frames are decoded as they are produced. ALSA starts on decoded frame
  one. Capture continues to drain while inference and playback run.
- The ThreeHub RGB status LED is out-of-band telemetry, not part of the model
  graph. Critical threads only publish the latest state through a nonblocking
  atomic mailbox; a dedicated native-C worker invokes the board's resident
  supervisor. Listening is very-slow green, inference slow yellow, and
  playback slow pure blue. Blue begins only after ALSA accepts the first PCM
  frame, so the light describes externally observable playback rather than
  merely available model tokens.
- After text EOS, Thinker advances one bridge at a time. The rejected 16-step
  bridge batch improved aggregate throughput but created a visible 0.7--0.9 s
  hole in Talker code arrival and therefore was not streaming.
- Mimi uses the checkpoint's complete 250-position attention window. A
  64-position experiment was only 7% faster over 128 frames but changed the
  long-form waveform, so it was rejected as an audio-quality optimization.
- The executable and model releases are SHA256-pinned. The systemd runner
  verifies them, downloads missing artifacts through a `.part` file, and then
  starts the already-warm resident service.

The measured result on the A113X box is:

| Gate | Final measured result |
|---|---|
| Online SenseVoice, 1.18 s fixture | 1.11 s wall, 365% CPU, 228,580 KiB peak RSS |
| Encoder improvement | 3.04 s to 1.11 s, 2.74x wall-clock speedup |
| W8A8 numerical gate | cosine 0.998620222, RMSE 0.029285131 over 15,360 values |
| Input prefill correctness | positions `4 -> 12 -> 24 -> 37`; whole-prompt tokenizer parity |
| Persistent compute pool | 2,000 job passes and 100 ownership handoffs passed |
| Four-core text/audio workload | 8.23 s wall at 358% CPU; 40 Mimi frames |
| Streaming Mimi, 8 frames/0.64 s PCM | 0.62--0.69 s wall after exact NEON dot reuse; PCM SHA256 unchanged |
| Streaming Mimi, 128 frames/10.24 s PCM | 10.69 s wall, 365% CPU; 64-position variant was 9.99 s |
| Concurrent fixed-prompt trace | 27 of 28 frames decoded before producer EOS; no 16-step generation gaps |
| Concurrent output cadence | mostly 100--125 ms per 80 ms frame; improved but not yet a guaranteed underrun-free RTF |
| Mandatory playback policy | first decoded frame, 80 ms; no EOS/watermark/full-response gate |
| Empty-cache boot recovery | 78 s download/restore; warm restart verification 6 s; warm-up 539 ms |

These results prove the queueing, ownership, incremental-input, and first-frame
audio path. They do **not** yet prove uninterrupted real-time playback: the
current combined Thinker/Talker/Mimi cadence is usually 20--45 ms slower than
each 80 ms audio frame. That remaining throughput deficit must be removed in
the kernels/model path; it must not be hidden by delaying first audio. The
small immediate safety-dialog model therefore remains a separate required
product path, while MiniMind-O remains the accurate/native S2S prototype.

This record captures the mistakes, measurements, and design rules learned
while turning the first MiniMind-O speech-to-speech prototype into an
always-resident, bidirectionally streaming pipeline. It is intentionally more
general than a changelog: the goal is to prevent the same architectural errors
in later AI-hub models.

## 1. Output streaming is only half of streaming

The first implementation streamed Talker codebooks into Mimi and streamed
decoded PCM into ALSA, but it still waited for VAD `speech_end` before starting
SenseVoice and Thinker/Talker prefill. It was correct to call the output path
streaming, but incorrect to call the complete interaction end-to-end
streaming.

The decisive rule is:

> As soon as speech starts, stable audio representations must enter the model
> state while capture continues.

The production input path now starts at `speech_start`:

```text
ALSA capture
    -> VAD/pre-roll
    -> online SenseVoice chunk
    -> committed 768-wide audio embedding
    -> <|audio_pad|> replacement
    -> Thinker bridge prefill
    -> Talker prefill
```

At `speech_end`, the runtime processes only the encoder tail and the fixed
assistant suffix. It does not cold-encode or cold-prefill the whole utterance.

The observable acceptance criterion is stronger than a final JSON boolean. On
a normal multi-second utterance, at least one `input_audio_commit` and
`input_prefill` event must occur before `speech_end`. A diagram or an output
worker alone cannot establish this.

## 2. A frozen bidirectional encoder cannot commit an arbitrary prefix

SenseVoice in this checkpoint uses full-sequence self-attention and an
11-position bidirectional FSMN. Re-running the offline graph on a longer audio
prefix can change embeddings previously produced for the shorter prefix.
Naively injecting those earlier embeddings into a causal language-model KV
cache would make irreversible decisions from non-final values.

The practical native-C solution is chunk-aware truncated attention:

- 8 LFR frames form the committed center, approximately 480 ms;
- 4 LFR frames are right lookahead, approximately 240 ms;
- each of the 70 encoder layers keeps 32 committed K/V frames, approximately
  1.92 seconds;
- right-lookahead frames are recomputed in the next chunk;
- only the center enters the cache and language models;
- EOS flushes all remaining frames.

This follows the streaming shape of SenseVoice SANM, but it is an online
approximation of a frozen bidirectional checkpoint. It is not bit-equivalent
to whole-utterance inference and must not be described as such.

The frontend has its own future dependency: one LFR position stacks seven mel
frames around a six-frame stride. A non-final LFR frame is exposed only after
the required right mel context exists. Encoder streaming must respect both
frontend stability and transformer/FSMN lookahead.

## 3. Committed tokens cannot be retracted, so VAD owns the uncertain tail

The original VAD appended low-energy trailing chunks and trimmed them only
after deciding that speech had ended. That works for a batch encoder but not
for a model whose KV cache has already consumed audio tokens.

The streaming design keeps possible trailing silence in a small pending ring:

- if speech resumes, the pending chunks are enqueued in order;
- if VAD closes the turn, they are discarded without entering the model;
- pre-roll remains bounded and is submitted when speech is first confirmed.

This is a general rule for incremental systems: ambiguity must stay outside an
irreversible state machine until it is resolved.

## 4. Model-state ownership is better than model-state locking

Thinker and Talker do not need a read/write lock. During capture-side prefill,
the input-inference thread is their only owner. At EOS, that thread finishes
and is joined. Generation then becomes the only owner. No two threads access
the same model state concurrently.

The ownership sequence is:

```text
input thread: reset -> prefix -> audio chunks -> suffix -> stop
                                                        |
                                                   join/handoff
                                                        |
main thread:                                      generation -> EOS
```

This eliminates locks around KV cache updates, tensor buffers, matrix kernels,
and model position. A lock would hide an invalid ownership model rather than
make it correct.

## 5. Locks belong only at queue enqueue/dequeue sleep boundaries

There are four data pipelines with SPSC ownership:

| Queue | Producer | Consumer |
|---|---|---|
| capture PCM | ALSA capture pthread | VAD/main thread |
| confirmed input PCM | VAD/main thread | input encoder/prefill pthread |
| Mimi code frames | Talker/main thread | Mimi decoder pthread |
| decoded PCM frames | Mimi decoder pthread | ALSA playback pthread |

Payload and sequence publication use release/acquire atomics. Mutexes exist
only to close the condition-variable race when a consumer sleeps on an empty
queue and a producer announces new data. No mutex covers:

- model inference or KV/history state;
- frontend, encoder, projector, Thinker, Talker, or Mimi computation;
- memcpy or queue payload ownership;
- file or ALSA I/O;
- worker-pool matrix execution.

The persistent compute pool uses one bounded FIFO so the Talker and Mimi
dispatchers can share CPU1/2. A tiny atomic spin lock protects only FIFO
enqueue/dequeue; a task executes after the lock is released. Atomic completion
counters replace per-job condition variables, and workers futex-sleep only
between inference sessions. This replaced OpenMP team creation and its large
context-switch/control overhead. The acceptance test passed 2,000 single-
dispatcher passes, 2,000 concurrent passes per dispatcher, and 100 session
handoffs. It also exposed a lost-wakeup bug at the Talker/Mimi handoff; giving
each dispatcher a reference-counted session fixed it before deployment.

The lesson is not that every mutex is slow. The lesson is that a mutex should
never compensate for unclear data ownership, and its protected interval must
not include work that can be performed after dequeue or before enqueue.

## 6. Overlap and acceleration solve different latency terms

Input streaming hides work under the user's speaking time; it does not make
that work cheaper. The first online Q8 x f32 SenseVoice implementation still
needed 3.04 seconds for a pinned 1.18-second fixture, despite using about 337%
aggregate CPU. It was structurally streaming but slower than real time.

On Cortex-A53, Armv8.0 NEON W8A8 dense products were the successful local
optimization:

- activations are symmetrically quantized per input position;
- int8 weights and activations use widening `vmull_s8`/`vpadalq_s16` dots;
- four input positions reuse one weight-row load;
- normalized Q/K/V input is quantized once and shared across all three
  projections;
- the audio projector remains Q8 x f32 because it was not a measured
  bottleneck.

Measured on the box:

| Online encoder | Wall | Aggregate CPU | Peak RSS |
|---|---:|---:|---:|
| Q8 weights x f32 activation | 3.04 s | 337% | 228,568 KiB |
| W8A8, final shared-QKV build | 1.11 s | 365% | 228,580 KiB |

The final speedup is 2.74x. Sharing Q/K/V activation quantization improved the
last W8A8 version from 1.15 to 1.11 seconds and preserved its output byte for
byte.

The general latency equation is therefore:

```text
post-speech wait = max(0, input work - speaking-time overlap)
                 + final right-context flush
                 + fixed suffix prefill
                 + first response token/code/PCM buffer
```

Measure and optimize each term separately. A smaller final number without
stage events cannot prove which work was actually overlapped.

## 7. Real streaming is a non-negotiable end-to-end contract

The failed two-, eight-, 32-, and complete-response buffering experiments were
useful diagnostics, but none is an acceptable product solution. They traded
away time-to-first-audio and made an internal streaming decoder externally
non-streaming. The corrected contract is:

1. Capture publishes PCM while the user is speaking.
2. Stable SenseVoice embeddings immediately advance Thinker/Talker state.
3. Talker publishes every complete eight-codebook frame immediately.
4. Mimi accepts exactly that one new frame and release-publishes its 80 ms PCM.
5. ALSA receives frame one immediately, before producer EOS, and receives every
   following frame as it arrives.

The earlier 3+1 CPU partition was also wrong: a one-core Mimi frame took
220--310 ms and guaranteed underruns. Production now keeps Talker on CPU0 and
Mimi on CPU3, while both dispatchers share persistent CPU1/2 workers through a
small FIFO. The only compute-pool lock is an atomic spin lock around FIFO
enqueue/dequeue. No matrix, attention, convolution, KV update, or PCM copy is
performed while holding it. A reference-counted session keeps workers awake
through the producer/decoder handoff; an A113X test caught and fixed the lost-
wakeup race before deployment.

The 16-position Thinker drain batch was a second latency bug. It paused Talker
code production for roughly 0.8 seconds even though its average throughput was
better. Advancing one bridge per output step removes that hole and allowed
27--28 of 28 frames to decode before producer EOS in the fixed-prompt tests.
Combined production/decode cadence is now mostly 100--125 ms per 80 ms frame,
so codec/model throughput still needs work. The shortfall is reported directly
and is never converted into a hidden PCM start buffer.

Mimi quality also constrains performance shortcuts. The optimized full
250-position attention window decoded 128 frames/10.24 seconds of PCM in
10.69 seconds. Reducing the window to 64 positions saved only 0.70 seconds
(6.5%) and changed the long waveform, so production retains the checkpoint
window. The useful wins came from precomputed RoPE, NEON QK/AV kernels, paired
position W8A8 dots, persistent quantization/activation arenas, and removing
per-frame allocation—not from throwing away model context.

## 8. Numerical similarity is necessary but not a quality claim

The W8A8 online embedding gate compared 15,360 float values against the Q8 x
f32 online graph:

- cosine similarity: `0.998620222`;
- RMSE: `0.029285131`;
- maximum absolute error: `0.138373837`.

These numbers show that the kernel did not catastrophically corrupt encoder
output. They do not prove equal multilingual recognition or response quality.
Autoregressive sampling can amplify small logit differences, especially in a
small model. Physical Chinese/English tests and a fixed speech evaluation set
remain separate acceptance gates.

Likewise, a file-fed test queues audio faster than real time. Its
`speech_start_to_ready_ms` validates state transitions and total work, not the
amount of work hidden during a live utterance. Live overlap must be established
from timestamps around actual `speech_start`, `input_prefill`, and
`speech_end` events.

## 9. Logging must describe handoffs, not just stage completion

The useful production events are:

```text
EVENT  speech_start
EVENT  input_stream_start
STREAM input_audio_commit
STREAM input_prefill
EVENT  speech_end
EVENT  input_stream_eos
EVENT  input_caught_up
STREAM talker_produce
STREAM mimi_decode
STREAM alsa_write
EVENT  model_end
EVENT  playback_end
EVENT  inference_end
```

Each queue boundary needs a timestamp and monotonic count. Thinker and Talker
positions are printed after input prefill, allowing direct verification that
the caches advanced. For the pinned file test they advanced `4 -> 12 -> 24 ->
37`, and the incremental token sequence matched whole-prompt tokenization.

Final metrics distinguish input streaming, decoder/generation overlap, ALSA
streaming, first-audio latency, producer completion, and codec drain. One
generic `streaming=true` field is insufficient for debugging.

## 10. Operational correctness is part of latency correctness

Early versions suffered broken pipes, model warm-up failures, lost capture
during inference, silent playback, and truncated audio. Those are pipeline
failures, not secondary deployment issues.

The production rules are:

- all model images stay mapped in one resident service;
- ALSA capture is continuously drained, including during inference/playback;
- the service never falls back to a batch mode;
- model and executable artifacts are SHA256-pinned;
- the runner downloads missing/corrupt files to a `.part` path and atomically
  renames only after validation;
- systemd restarts the service and initializes microphone/speaker controls;
- release executables are built and tested before upload;
- target binaries link only `libc`/`libm` and do not depend on `libgomp`.

This prevents a fast benchmark binary from becoming a slow or nonfunctional
product after reboot.

## 11. What remains true after this increment

- A sub-lookahead utterance may have no input commit before EOS; that is a
  consequence of the 240 ms right-context quality policy, not a batch fallback.
- The frozen bidirectional encoder still trades some offline context for
  bounded streaming context.
- MiniMind-O response quality and Mimi real-time factor are independent of
  input-token streaming and must be evaluated independently.
- The fast safety-dialog path and the accurate notification/reasoning path
  should remain separate product paths; optimizing one model cannot satisfy
  both latency/accuracy objectives automatically.

## Code and evidence

- Online encoder and W8A8 kernels:
  [`targets/generic/minimindo_audio_encoder.c`](targets/generic/minimindo_audio_encoder.c)
- PCM queue, incremental prefill, model ownership handoff, and stream logs:
  [`targets/generic/minimindo_speech.c`](targets/generic/minimindo_speech.c)
- A113X design and measurements:
  [`targets/a113x/README.md`](targets/a113x/README.md)
- Machine-readable results:
  [`targets/a113x/results.json`](targets/a113x/results.json)
- Historical rejected full-buffer speaker trace:
  [`targets/a113x/benchmarks/v1.3.0-continuous-playback.log`](targets/a113x/benchmarks/v1.3.0-continuous-playback.log)
- Current first-frame policy, kernel A/B, and shared-pool cadence trace:
  [`targets/a113x/benchmarks/v1.4.0-real-streaming-a113x.log`](targets/a113x/benchmarks/v1.4.0-real-streaming-a113x.log)
- Auto-download deployment runner:
  [`../../tools/threehub-voice/run-minimindo-native-a113x.sh`](../../tools/threehub-voice/run-minimindo-native-a113x.sh)
