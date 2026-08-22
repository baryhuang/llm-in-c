#!/bin/sh
set -eu

release_repository=${MINIMINDO_RELEASE_REPOSITORY:-baryhuang/llm-in-c}
release_tag=${MINIMINDO_RELEASE_TAG:-minimindo-native-a113x-v1.5.0}
model_release_tag=${MINIMINDO_MODEL_RELEASE_TAG:-minimindo-native-a113x-v1.0.0}
install_dir=${MINIMINDO_INSTALL_DIR:-/dev/shm/minimindo-o-native-v1}
release_root=${MINIMINDO_RELEASE_ROOT_URL:-https://github.com/$release_repository/releases/download}
curl_command=${CURL:-curl}
sha256_command=${SHA256SUM:-sha256sum}
part=

asset_sha256()
{
    case "$1" in
        minimindo-speech-a113x)
            printf '%s\n' 00de173208feb6f9dc2b7816f4f2a4beffa7c77c011c077030c6cd140221d6a0 ;;
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

asset_release_tag()
{
    case "$1" in
        minimindo-speech-a113x) printf '%s\n' "$release_tag" ;;
        *) printf '%s\n' "$model_release_tag" ;;
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
    asset_tag=$(asset_release_tag "$asset")
    destination=$install_dir/$asset
    if file_matches "$expected" "$destination"; then
        printf 'artifact ready: %s\n' "$asset"
        return 0
    fi

    part=$install_dir/.$asset.part.$$
    rm -f "$part"
    printf 'downloading %s from release %s\n' "$asset" "$asset_tag"
    $curl_command --fail --location --show-error --silent \
        --connect-timeout 20 --retry 8 --retry-delay 2 \
        --output "$part" "$release_root/$asset_tag/$asset"
    if ! file_matches "$expected" "$part"; then
        printf 'SHA256 mismatch for %s\n' "$asset" >&2
        return 1
    fi
    if test "$asset" = minimindo-speech-a113x; then
        chmod 0755 "$part"
    else
        chmod 0644 "$part"
    fi
    mv -f "$part" "$destination"
    part=
    printf 'artifact installed: %s\n' "$asset"
}

cleanup()
{
    if test -n "$part"; then rm -f "$part"; fi
}
trap cleanup EXIT HUP INT TERM

umask 022
mkdir -p "$install_dir"
for asset in \
    minimindo-speech-a113x \
    minimindo-thinker-q8-v1.mmo \
    minimindo-talker-q8-v1.mmo \
    minimindo-tokenizer-v1.mmotok \
    minimindo-mimi-q8-v1.mmo \
    minimindo-sensevoice-q8-v1.mmo
do
    download_asset "$asset"
done

if test "${MINIMINDO_DOWNLOAD_ONLY:-0}" = 1; then
    printf 'all MiniMind-O artifacts verified in %s\n' "$install_dir"
    exit 0
fi

if test "$#" -eq 0; then
    set -- --live \
        --audio-encoder "$install_dir/minimindo-sensevoice-q8-v1.mmo" \
        --capture-device plughw:0,0 \
        --playback-device plughw:0,0 \
        --max-tokens 32 \
        --seed 20260821
fi

exec "$install_dir/minimindo-speech-a113x" \
    "$install_dir/minimindo-thinker-q8-v1.mmo" \
    "$install_dir/minimindo-talker-q8-v1.mmo" \
    "$install_dir/minimindo-tokenizer-v1.mmotok" \
    "$install_dir/minimindo-mimi-q8-v1.mmo" \
    "$@"
