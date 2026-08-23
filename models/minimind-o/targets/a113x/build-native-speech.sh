#!/bin/sh
set -eu

if test "$#" -gt 1; then
    printf 'usage: %s [OUTPUT]\n' "$0" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
repository=$(CDPATH= cd -- "$script_dir/../../../.." && pwd)
generic=$repository/models/minimind-o/targets/generic
output=${1:-$repository/build/minimindo-speech-a113x}

case "$(uname -s):$(uname -m)" in
    Linux:aarch64|Linux:arm64) ;;
    *)
        printf 'error: build inside a Linux/AArch64 environment\n' >&2
        exit 1
        ;;
esac

for command_name in gcc sha256sum strip; do
    command -v "$command_name" >/dev/null 2>&1 || {
        printf 'error: required build command is missing: %s\n' \
            "$command_name" >&2
        exit 1
    }
done

test -f /usr/include/alsa/asoundlib.h || {
    printf 'error: ALSA development headers are required\n' >&2
    exit 1
}

mkdir -p "$(dirname "$output")"
gcc -Ofast -std=c11 -Wall -Wextra -Wpedantic -Werror \
    -DMINIMINDO_GENERATION_W8A8=1 \
    -DMINIMINDO_ALSA_VOLUME_MONITOR=1 \
    -march=armv8-a+simd -mtune=cortex-a53 -I"$generic" \
    "$generic/minimindo_speech.c" \
    "$generic/minimindo_thinker.c" \
    "$generic/minimindo_talker.c" \
    "$generic/minimindo_tokenizer.c" \
    "$generic/minimindo_mimi.c" \
    "$generic/minimindo_audio_encoder.c" \
    "$generic/minimindo_volume.c" \
    "$generic/minimindo_parallel.c" \
    -o "$output" -pthread -lm -lasound
strip "$output"
sha256sum "$output"
