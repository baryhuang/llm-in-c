# MiniMind-O on Rockchip RK3588

Second hardware target for the native-C MiniMind-O speech-to-speech runtime,
after [A113X](../a113x/README.md). The board used here is a CareMojo hub:
Armbian 26.8.2 (trixie), kernel 6.1.115-vendor-rk35xx, 8 GB RAM, GCC 14.2,
with the same MV-SILICON P10S USB speakerphone as the ThirdReality hub.

The pipeline, streaming semantics and correctness gates are unchanged; this
record covers only what the port had to adjust and what the part measures.

## What the port changes

RK3588 is not a bigger A113X. It is four Cortex-A76 plus four Cortex-A55 on
Armv8.2-A, and its USB audio card does not enumerate first. Four things follow.

**CPU lanes.** The runtime owns four lanes: lane 0 is the caller, lanes 1--3
are the persistent workers, and the Mimi decoder takes lane 3 while Talker
generates. On A113X those lanes are CPUs 0--3 because every core is identical.
On RK3588 the lanes must land on the A76 cluster, so
`minimindo_parallel_pin_current` maps lane *n* to CPU `MINIMINDO_CPU_BASE + n`
and this target sets `MINIMINDO_CPU_BASE=4`. A base that would push a lane past
the online CPUs is ignored rather than allowed to fail every pinning call. The
default of 0 keeps A113X bit-identical.

**Audio card.** The P10S is card 0 on the A113X hub and card 4 here, behind the
on-SoC DP, HDMI and ES8388 devices. The launcher resolves the index by name
from `/proc/asound/cards`, then derives the capture device, the playback device
and the mixer. `MINIMINDO_VOLUME_MIXER` replaces the previously hard-coded
`hw:0` in the volume monitor; `MINIMINDO_AUDIO_CARD` overrides the search.

**Status LED.** The RGB pattern states are executed by the ThirdReality
`supervisor` binary, which owns the GPIO lines and the blink timer. RK3588 has
no such supervisor, so the LED worker would spawn a doomed process per state
change. `hub_led_start` now requires an executable supervisor, and
`MINIMINDO_HUB_LED=0` disables it explicitly. The voice path already tolerated
a missing LED worker, so the turn is otherwise identical.

**Build flags.** `-march=armv8.2-a+simd+dotprod+fp16 -mtune=cortex-a76.cortex-a55`
replaces the A53 pair. The ISA baseline is what both clusters implement; the
tuning model is the big.LITTLE pair. The Q8 kernels themselves are unchanged
`vmull_s8`/`vpadalq_s16` chains — the Armv8.2 dot product is available on this
part and is not yet used.

## Measurements

Same utterance on both lane sets: prompt `你好`, seed 3, `--max-tokens 48`,
39 audio frames (3.12 s of speech at 80 ms per frame), service stopped.

| lanes | prefill_ms | generate_ms | model_ms | ms per 80 ms frame |
| --- | --- | --- | --- | --- |
| CPU 4--7, Cortex-A76 | 173 | 665 | 932 | 17.1 |
| CPU 0--3, Cortex-A55 | 1071 | 13258 | 14366 | 340.0 |

The 15x gap is larger than the 1.8 GHz/2.4 GHz clock ratio: the in-order A55
stalls on the dependent NEON multiply-accumulate chains that the out-of-order
A76 hides. Pinning to the little cluster is not a slower configuration, it is a
non-working one. For reference, the A113X v1.4.0 log generates 28 frames in
3546 ms, or 126.6 ms per frame, so the A76 lanes are 7.4x faster per frame.

Stateful Mimi decode is 14.3--14.8 ms per frame, `rtf` 0.18. Resident warm-up
is 133 ms against 684 ms on A113X, and resident set size is 200 MB.

One live turn, captured through the P10S microphone:

```text
EVENT speech_start        turn=1 preroll_ms=512
EVENT input_stream_start  turn=1 prefix_tokens=4
EVENT speech_end          turn=1 samples=16384 duration_ms=1024
EVENT input_caught_up     turn=1 speech_end_to_ready_ms=226
EVENT first_audio         turn=1 elapsed_ms=1446 buffered_frames=1
```

with `"input_streaming":true`, `"decoder_to_alsa_streaming":true` and
`"end_to_end_streaming":true` in the turn JSON: input embeddings prefilled
while the speaker was still talking, and PCM reached ALSA on frame one.

The `ondemand` governor idles the A76 cluster at 408 MHz and ramps under load;
the numbers above include that ramp. `performance` on `policy4`/`policy6` would
remove it at the cost of idle power, and is left to the operator.

## Build and install

Build on the board — the script refuses anything but Linux/AArch64 and needs
`libasound2-dev`:

```sh
sh models/minimind-o/targets/rk3588/build-native-speech.sh
install -m 0755 build/minimindo-speech-rk3588 /usr/local/bin/
install -m 0755 tools/threehub-voice/run-minimindo-native-rk3588.sh /usr/local/bin/
install -m 0644 tools/threehub-voice/minimindo-native-rk3588.service /etc/systemd/system/
systemctl enable --now minimindo-native-rk3588.service
```

The model images are architecture independent and are pulled, SHA256-verified,
from the A113X model release into `/dev/shm/minimindo-o-native-v1` (378 MB).
`/dev/shm` is tmpfs, so the launcher re-downloads them after a reboot; the
executable lives on the root filesystem because it is built, not downloaded.
`MINIMINDO_DOWNLOAD_ONLY=1` verifies the assets without starting the runtime.

Environment knobs: `MINIMINDO_CPU_BASE` (4), `MINIMINDO_AUDIO_CARD` (resolved),
`MINIMINDO_VOLUME_MIXER` (`hw:<card>`), `MINIMINDO_HUB_LED` (0),
`MINIMINDO_PLAYBACK_PERCENT` (5%), `MINIMINDO_CAPTURE_PERCENT` (50%),
`MINIMINDO_INSTALL_DIR`, `MINIMINDO_SPEECH_BINARY`.
