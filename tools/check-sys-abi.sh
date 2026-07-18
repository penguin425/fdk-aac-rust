#!/bin/sh
set -eu

sha256sum --check api/fdk-aac-rust-sys.sha256

cat <<'EOF'
The sys FFI declaration snapshot matches. Any intentional change to
crates/fdk-aac-sys/src/lib.rs requires an ABI review and an explicit snapshot
update with:
  sha256sum crates/fdk-aac-sys/src/lib.rs > api/fdk-aac-rust-sys.sha256
EOF
