# MiniMind-O

Status: native-C speech-to-speech prototype running as a resident service on
the Amlogic A113D/A113X target. The runtime target is `minimind-3o`, with the
vision path removed. Python/PyTorch is allowed only on the build host to
produce pinned fixtures and packed model images; the target executable and hot
path do not embed or launch Python.

Implemented components include the Q8 Thinker and Talker, tokenizer,
SenseVoice audio encoder/projector, stateful Mimi decoder, continuous ALSA
capture/VAD, Cortex-A53 NEON kernels and a persistent four-core worker
pool. Input audio is chunk-encoded and prefills Thinker/Talker while the user
is still speaking. Output codebooks enter the stateful Mimi decoder while
Talker is still generating, and the first decoded 80 ms PCM frame is written
to ALSA immediately. The runtime never waits for producer EOS, a PCM watermark,
or a complete response. Talker and Mimi share persistent CPU workers through a
minimal-lock FIFO; the remaining 99--128 ms observed cadence for each 80 ms
frame is exposed as a throughput deficit, not hidden with buffering. There is
no production batch-decoder or playback-mode switch. Target
architecture, correctness gates and measurements are in the
[A113X target record](targets/a113x/README.md). The architectural mistakes,
measurement lessons, and reusable streaming rules from the prototype are in
[LEARNINGS.md](LEARNINGS.md).

Upstream source: <https://github.com/jingyaogong/minimind-o>.
