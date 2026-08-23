# ThreeHub resident voice assistant

This is the always-on A113X pipeline deployed on the ThirdReality TRHub-V3.
One `arecord` process holds ALSA continuously and feeds the native C
segmenter. Whisper Base multilingual, Qwen3.5-0.8B, and MOSS-TTS-Nano each
remain alive with their model mapped between turns. There is no Python process
in capture, VAD, inference, synthesis, monitoring, or supervision.

```text
ALSA (held continuously) -> C RMS segmenter -> resident Whisper + Silero VAD
                                              -> resident Qwen
                                              -> resident MOSS -> ALSA speaker
```

Speaker playback creates `playback.active`. The capture process continues to
drain ALSA while that file exists but discards those frames, preventing the
cloned response from becoming a new user turn. Up to eight complete utterances
can wait while a slow inference stage is active; additional turns are counted
as dropped instead of filling `/dev/shm`.

## Live monitor

The capture process atomically updates peak and RMS levels ten times per
second. The monitor reads that same state and never opens the microphone:

```sh
ssh root@100.123.75.40 /root/threehub-voice/threehub-voice-monitor.sh
```

The production defaults start after 100 ms at 1.2% frame RMS and stop after
1.2 seconds below 0.7% RMS. `VOICE_START_PERCENT`, `VOICE_STOP_PERCENT`,
`VOICE_START_MS`, `VOICE_STOP_MS`, and `VOICE_MAX_QUEUED_UTTERANCES` tune the
installation without rebuilding.

## Build and release

All target executables are built before upload. Pin whisper.cpp v1.8.0 at
commit `41fc9dea6a4fe056424be86f61164413903fcff4`, then run in a Linux/AArch64
Debian 12 environment:

```sh
tools/threehub-voice/build-a113x.sh build/threehub-release /path/to/whisper.cpp
```

The build uses GCC/CMake only. The assistant launcher downloads every public
runtime/model artifact from GitHub Releases and verifies its SHA-256. The
private voice reference stays at `/root/threehub-voice/voice_ref.wav` and is
never uploaded.

Install the launcher, monitor, and unit, then enable supervision:

```sh
install -m 0755 threehub-voice-assistant.sh threehub-voice-monitor.sh \
  /root/threehub-voice/
install -m 0644 threehub-voice-assistant.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now threehub-voice-assistant.service
```

`systemd` restarts the pipeline on failure. Startup does one real request
through each model before setting the pipeline state to `listening`; requests
after that use the already-running servers. Raw validation measurements are in
[`results.json`](results.json).

## MiniMind-O native S2S runner

`run-minimindo-native-a113x.sh` is the persistent launcher for the separate
native MiniMind-O service. It pins the executable to GitHub release
`minimindo-native-a113x-v1.6.0` and the unchanged packed model images to
`minimindo-native-a113x-v1.0.0`. Missing or corrupt files are downloaded to a
per-process `.part` file, verified by SHA-256, chmodded, and atomically renamed
before execution. The default live policy is first-sentence completion with a
32-text-step fallback; Mimi then drains staggered audio codebooks to EOS.

The executable is native C11/AArch64 with no OpenMP or Python dependency. Its
Mimi decoder always accepts one codec frame per call, uses the full
250-position checkpoint attention window, and overlaps code decoding with
Talker generation. Speaker delivery starts on the first decoded 80 ms PCM
frame. It never waits for producer EOS, a frame watermark, or the complete
response; continuity work must improve sustained model/codec throughput rather
than hide latency in a start buffer.

The same native process exposes the live pipeline on the ThreeHub front RGB
LED through the board's resident `supervisor`: listening is a very slow green
blink, inference is a slow yellow blink, and playback is a slow pure/deep-blue
blink. An isolated LED worker performs the supervisor calls. Capture,
generation, Mimi decode, and ALSA playback only publish the latest state to a
nonblocking atomic mailbox, so LED control cannot block or lock the streaming
path. Playback changes to blue only after the first 80 ms PCM frame has been
successfully written to ALSA.

The P10S volume wheel is also handled inside the always-resident native
process. A blocking evdev thread reads volume-up, volume-down, and mute events
and updates the ALSA `PCM` mixer immediately, including while speech PCM is
being streamed. It uses a persistent mixer handle, a 2% step, no subprocess,
and no model/audio-path lock. The service's 5% `ExecStartPre` remains only the
safe non-zero boot default.

To populate and verify a volatile install without starting the service:

```sh
MINIMINDO_DOWNLOAD_ONLY=1 \
  MINIMINDO_INSTALL_DIR=/dev/shm/minimindo-o-native-v1 \
  ./run-minimindo-native-a113x.sh
```

## MiniMind-O on RK3588

`run-minimindo-native-rk3588.sh` is the same launcher for the RK3588 hub. It
downloads only the packed model images — the executable is built on the board
by `models/minimind-o/targets/rk3588/build-native-speech.sh` and installed to
`/usr/local/bin`, because no RK3588 release binary is published.

The launcher resolves the P10S card index by name from `/proc/asound/cards`
instead of assuming card 0, and exports the three settings that differ from the
ThirdReality hub: `MINIMINDO_CPU_BASE=4` so the four inference lanes land on the
Cortex-A76 cluster rather than the A55 cluster, `MINIMINDO_VOLUME_MIXER=hw:<card>`
for the volume wheel, and `MINIMINDO_HUB_LED=0` because this board has no
`supervisor` binary owning the RGB lines. Everything else — streaming policy,
Mimi decoder, VAD, playback path — is the shared runtime. The port and its
measurements are recorded in
[`models/minimind-o/targets/rk3588/README.md`](../../models/minimind-o/targets/rk3588/README.md).

```sh
install -m 0644 minimindo-native-rk3588.service /etc/systemd/system/
systemctl enable --now minimindo-native-rk3588.service
```
