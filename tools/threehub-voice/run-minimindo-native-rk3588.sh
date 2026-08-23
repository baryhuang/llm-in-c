#!/bin/sh
set -eu

# MiniMind-O speech-to-speech launcher for RK3588 hubs (Armbian trixie).
#
# The model assets are architecture independent and are pulled from the same
# release as the A113X hub. The executable is not: it is built on the board by
# models/minimind-o/targets/rk3588/build-native-speech.sh and installed next to
# the weights, so this script verifies rather than downloads it.

release_repository=${MINIMINDO_RELEASE_REPOSITORY:-baryhuang/llm-in-c}
model_release_tag=${MINIMINDO_MODEL_RELEASE_TAG:-minimindo-native-a113x-v1.0.0}
install_dir=${MINIMINDO_INSTALL_DIR:-/dev/shm/minimindo-o-native-v1}
release_root=${MINIMINDO_RELEASE_ROOT_URL:-https://github.com/$release_repository/releases/download}
speech_binary=${MINIMINDO_SPEECH_BINARY:-$install_dir/minimindo-speech-rk3588}
curl_command=${CURL:-curl}
sha256_command=${SHA256SUM:-sha256sum}
capture_percent=${MINIMINDO_CAPTURE_PERCENT:-50%}
playback_percent=${MINIMINDO_PLAYBACK_PERCENT:-5%}
part=

asset_sha256()
{
    case "$1" in
        minimindo-thinker-q8-v1.mmo)
            printf '%s\n' 7a0c78199510275aa25af55fcf6f1f5bd66ca05fdb99db3c18abd28c258a66ab ;;
        minimindo-talker-q8-v1.mmo)
            printf '%s\n' f50ec2c3d42d50dd35ba95b74f8438d808620c27e756ce4bc5bf3d3619ab5706 ;;
        minimindo-tokenizer-v1.mmotok)
            printf '%s\n' f4543e7d18a9c14f53b1baf9a38d97d398dc03c42dc4bb70a9c806deb9d37ae3 ;;
        minimindo-mimi-q8-v1.mmo)
            printf '%s\n' 98abdabe68b7f05e51fbdf4549c137b3b0ca6b0eb22e13b90b665197a284e294 ;;
        minimindo-sensevoice-q8-v1.mmo)
            printf '%s\n' 72c228b9b713daa1c6f559d253050612b58012db747fcdd4cdc4b7a375c44c97 ;;
        *) return 1 ;;
    esac
}

file_matches()
{
    test -f "$2" || return 1
    actual=$($sha256_command "$2" | awk '{print $1}')
    test "$actual" = "$1"
}

download_asset()
{
    asset=$1
    expected=$(asset_sha256 "$asset")
    destination=$install_dir/$asset
    if file_matches "$expected" "$destination"; then
        printf 'artifact ready: %s\n' "$asset"
        return 0
    fi

    part=$install_dir/.$asset.part.$$
    rm -f "$part"
    printf 'downloading %s from release %s\n' "$asset" "$model_release_tag"
    $curl_command --fail --location --show-error --silent \
        --connect-timeout 20 --retry 8 --retry-delay 2 \
        --output "$part" "$release_root/$model_release_tag/$asset"
    if ! file_matches "$expected" "$part"; then
        printf 'SHA256 mismatch for %s\n' "$asset" >&2
        return 1
    fi
    chmod 0644 "$part"
    mv -f "$part" "$destination"
    part=
    printf 'artifact installed: %s\n' "$asset"
}

cleanup()
{
    if test -n "$part"; then rm -f "$part"; fi
}
trap cleanup EXIT HUP INT TERM

# The P10S is USB audio, so its card index depends on enumeration order: 0 on
# the A113X hub, 4 on RK3588 behind the on-SoC DP/HDMI/ES8388 devices. Resolve
# it by name instead of assuming either number.
resolve_card()
{
    if test -n "${MINIMINDO_AUDIO_CARD:-}"; then
        printf '%s\n' "$MINIMINDO_AUDIO_CARD"
        return 0
    fi
    card=$(sed -n 's/^ *\([0-9][0-9]*\) \[P10S *\].*/\1/p' /proc/asound/cards |
        head -1)
    if test -z "$card"; then
        printf 'error: no P10S audio card found in /proc/asound/cards\n' >&2
        return 1
    fi
    printf '%s\n' "$card"
}

umask 022
mkdir -p "$install_dir"
for asset in \
    minimindo-thinker-q8-v1.mmo \
    minimindo-talker-q8-v1.mmo \
    minimindo-tokenizer-v1.mmotok \
    minimindo-mimi-q8-v1.mmo \
    minimindo-sensevoice-q8-v1.mmo
do
    download_asset "$asset"
done

if test "${MINIMINDO_DOWNLOAD_ONLY:-0}" = 1; then
    printf 'all MiniMind-O model artifacts verified in %s\n' "$install_dir"
    exit 0
fi

test -x "$speech_binary" || {
    printf 'error: build the RK3588 runtime first: %s\n' "$speech_binary" >&2
    exit 1
}

audio_card=$(resolve_card)
amixer -c "$audio_card" set Mic "$capture_percent" cap >/dev/null 2>&1 || true
amixer -c "$audio_card" set PCM "$playback_percent" unmute >/dev/null 2>&1 || true

# Lanes 0..3 map onto CPUs 4..7, the Cortex-A76 cluster. The A55 cluster is
# left to the kernel, ALSA and the capture thread's interrupt work.
MINIMINDO_CPU_BASE=${MINIMINDO_CPU_BASE:-4}
MINIMINDO_VOLUME_MIXER=${MINIMINDO_VOLUME_MIXER:-hw:$audio_card}
MINIMINDO_HUB_LED=${MINIMINDO_HUB_LED:-0}
export MINIMINDO_CPU_BASE MINIMINDO_VOLUME_MIXER MINIMINDO_HUB_LED

if test "$#" -eq 0; then
    set -- --live \
        --audio-encoder "$install_dir/minimindo-sensevoice-q8-v1.mmo" \
        --capture-device "plughw:$audio_card,0" \
        --playback-device "plughw:$audio_card,0" \
        --max-tokens 32 \
        --seed 20260821
fi

exec "$speech_binary" \
    "$install_dir/minimindo-thinker-q8-v1.mmo" \
    "$install_dir/minimindo-talker-q8-v1.mmo" \
    "$install_dir/minimindo-tokenizer-v1.mmotok" \
    "$install_dir/minimindo-mimi-q8-v1.mmo" \
    "$@"
