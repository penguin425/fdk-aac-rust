#!/bin/sh
set -eu

target_dir=${EXTERNAL_FFI_TARGET_DIR:-target/external-ffi}
work_dir="$target_dir/consumer"

rm -rf "$target_dir"
mkdir -p "$work_dir"
CARGO_TARGET_DIR="$target_dir" cargo build --locked -p fdk-aac-rust-sys
archive=$(find "$target_dir/debug/build" -path '*/out/libfdk-aac.a' -print | head -n 1)
source_root=$(find "$target_dir/debug/build" -type d -name 'fdk-aac-upstream-*' -print | head -n 1)
test -n "$archive"
test -n "$source_root"

# This source is generated under target/ so the repository remains C/C++ free.
# It exercises the canonical libfdk-aac entry points used by C applications
# such as FFmpeg, including encoder parameters and decoder buffer reset.
cat > "$work_dir/consumer.c" <<'EOF'
#include <stdio.h>
#include "aacenc_lib.h"
#include "aacdecoder_lib.h"

int main(void) {
  HANDLE_AACENCODER encoder = 0;
  if (aacEncOpen(&encoder, 0, 2) != 0 || encoder == 0) return 1;
  if (aacEncoder_SetParam(encoder, AACENC_AOT, AOT_AAC_LC) != 0) return 2;
  if (aacEncoder_SetParam(encoder, AACENC_SAMPLERATE, 48000) != 0) return 3;
  if (aacEncoder_SetParam(encoder, AACENC_CHANNELMODE, MODE_2) != 0) return 4;
  if (aacEncoder_SetParam(encoder, AACENC_BITRATE, 128000) != 0) return 5;
  (void)aacEncoder_GetParam(encoder, AACENC_BITRATE);
  if (aacEncClose(&encoder) != 0 || encoder != 0) return 6;

  HANDLE_AACDECODER decoder = aacDecoder_Open(TT_MP4_ADTS, 1);
  if (decoder == 0) return 7;
  if (aacDecoder_SetParam(decoder, AAC_TPDEC_CLEAR_BUFFER, 1) != 0) return 8;
  aacDecoder_Close(decoder);
  puts("external C ABI consumer passed");
  return 0;
}
EOF

${CC:-cc} -std=c11 -Wall -Wextra -Werror -Wno-unused-function \
  "$work_dir/consumer.c" \
  -I"$source_root/libAACenc/include" -I"$source_root/libAACdec/include" \
  -I"$source_root/libSYS/include" \
  "$archive" -lstdc++ -lm -lpthread -o "$work_dir/consumer"
"$work_dir/consumer"
