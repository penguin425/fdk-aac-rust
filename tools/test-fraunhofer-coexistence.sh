#!/bin/sh
set -eu

readonly mpegh_url=https://github.com/Fraunhofer-IIS/mpeghdec.git
readonly mpegh_revision=8149df84a777ea7d0a9a326f3c36067aec39201e
readonly work_dir=${FRAUNHOFER_COEXISTENCE_DIR:-target/fraunhofer-coexistence}
readonly source_dir=$work_dir/mpeghdec
readonly build_dir=$work_dir/mpeghdec-build
readonly fdk_target_dir=$work_dir/fdk-target
readonly expected_collisions=api/mpeghdec-r4.0.0-collisions.txt

for command in cargo git cmake c++ nm objcopy awk sort comm cmp find; do
  command -v "$command" >/dev/null || {
    echo "$command is required for the Fraunhofer coexistence test" >&2
    exit 1
  }
done

mkdir -p "$work_dir"
if [ ! -d "$source_dir/.git" ]; then
  git init --quiet "$source_dir"
  git -C "$source_dir" remote add origin "$mpegh_url"
fi
git -C "$source_dir" fetch --quiet --depth=1 origin "$mpegh_revision"
test "$(git -C "$source_dir" rev-parse FETCH_HEAD)" = "$mpegh_revision"
git -c advice.detachedHead=false -C "$source_dir" checkout --quiet --detach FETCH_HEAD
if [ -n "$(git -C "$source_dir" status --porcelain)" ]; then
  echo "The pinned mpeghdec checkout is unexpectedly dirty" >&2
  exit 1
fi

CARGO_TARGET_DIR="$fdk_target_dir" cargo build --quiet --locked -p fdk-aac-rust-sys
fdk_archive=$(find "$fdk_target_dir/debug/build" -path '*/out/libfdk-aac.a' -print -quit)
test -n "$fdk_archive"

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -Dmpeghdec_BUILD_BINARIES=OFF \
  -Dmpeghdec_BUILD_UIMANAGER=OFF
cmake --build "$build_dir" --config Release --parallel 2
mpegh_archive=$build_dir/lib/libmpeghdec.a
test -f "$mpegh_archive"

nm -g --defined-only "$fdk_archive" |
  awk 'NF >= 3 {print $NF}' | sort -u > "$work_dir/fdk.symbols"
nm -g --defined-only "$mpegh_archive" |
  awk 'NF >= 3 {print $NF}' | sort -u > "$work_dir/mpeghdec.symbols"
comm -12 "$work_dir/fdk.symbols" "$work_dir/mpeghdec.symbols" > "$work_dir/collisions.txt"

if ! cmp --silent "$expected_collisions" "$work_dir/collisions.txt"; then
  echo "Fraunhofer static symbol collisions changed; review the generated diff:" >&2
  diff -u "$expected_collisions" "$work_dir/collisions.txt" >&2 || true
  exit 1
fi
if grep -q '^mpeghdecoder_' "$work_dir/collisions.txt"; then
  echo "A public mpeghdec API symbol collides and cannot be namespaced as internal" >&2
  exit 1
fi

awk '{print $1, "mpeghdec_internal_" $1}' "$work_dir/collisions.txt" > "$work_dir/redefine.syms"
namespaced_archive=$work_dir/libmpeghdec-namespaced.a
objcopy --redefine-syms="$work_dir/redefine.syms" "$mpegh_archive" "$namespaced_archive"
nm -g --defined-only "$namespaced_archive" |
  awk 'NF >= 3 {print $NF}' | sort -u > "$work_dir/mpeghdec-namespaced.symbols"
if comm -12 "$work_dir/fdk.symbols" "$work_dir/mpeghdec-namespaced.symbols" |
  grep -q .
then
  echo "Namespaced mpeghdec archive still collides with fdk-aac" >&2
  exit 1
fi

fdk_source=$(find "$fdk_target_dir/debug/build" \
  -type d -name 'fdk-aac-upstream-*' -print -quit)
test -n "$fdk_source"
cat > "$work_dir/probe.cpp" <<'CPP'
#include "aacenc_lib.h"
#include "mpeghdecoder.h"

int main() {
  HANDLE_AACENCODER encoder = nullptr;
  if (aacEncOpen(&encoder, 0, 2) != AACENC_OK) return 1;
  if (aacEncClose(&encoder) != AACENC_OK) return 2;
  HANDLE_MPEGH_DECODER_CONTEXT decoder = mpeghdecoder_init(2);
  if (decoder == nullptr) return 3;
  mpeghdecoder_destroy(decoder);
  return 0;
}
CPP
c++ -std=c++11 -DMPEGHDEC_STATIC=1 \
  -I"$fdk_source/libAACenc/include" \
  -I"$fdk_source/libSYS/include" \
  -I"$source_dir/include" \
  -c "$work_dir/probe.cpp" -o "$work_dir/probe.o"

for order in fdk-first mpeghdec-first; do
  case "$order" in
    fdk-first)
      c++ "$work_dir/probe.o" "$fdk_archive" "$namespaced_archive" \
        -lm -lpthread -o "$work_dir/probe-$order"
      ;;
    mpeghdec-first)
      c++ "$work_dir/probe.o" "$namespaced_archive" "$fdk_archive" \
        -lm -lpthread -o "$work_dir/probe-$order"
      ;;
  esac
  "$work_dir/probe-$order"
done

collision_count=$(wc -l < "$work_dir/collisions.txt" | tr -d '[:space:]')
echo "Verified $collision_count reviewed collisions and zero collisions after namespacing."
echo "Both static link orders opened and closed the AAC and MPEG-H codecs successfully."
