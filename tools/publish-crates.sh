#!/bin/sh
set -eu

version=${1:-}
if ! printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "Usage: $0 MAJOR.MINOR.PATCH" >&2
  exit 1
fi

if [ -z "${CARGO_REGISTRY_TOKEN:-}" ]; then
  echo "CARGO_REGISTRY_TOKEN is required" >&2
  exit 1
fi

api_base=${CRATES_IO_API_BASE:-https://crates.io/api/v1}

crate_exists() {
  crate=$1
  curl --fail --silent --show-error --output /dev/null \
    --user-agent "fdk-aac-rust-release-workflow" \
    "$api_base/crates/$crate/$version"
}

wait_for_crate() {
  crate=$1
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    if crate_exists "$crate"; then
      echo "$crate $version is available from crates.io"
      return 0
    fi
    attempts=$((attempts + 1))
    sleep 10
  done
  echo "Timed out waiting for $crate $version in the crates.io index" >&2
  return 1
}

publish_if_missing() {
  package=$1
  crate=$2
  if crate_exists "$crate"; then
    echo "$crate $version is already published; skipping"
  else
    cargo publish -p "$package" --locked
  fi
  wait_for_crate "$crate"
}

# The safe crate has a versioned dependency on the sys crate. crates.io must
# have indexed that exact sys version before Cargo can verify the safe crate.
publish_if_missing fdk-aac-rust-sys fdk-aac-rust-sys
publish_if_missing fdk-aac-rust fdk-aac-rust

for archive in \
  "target/package/fdk-aac-rust-sys-$version.crate" \
  "target/package/fdk-aac-rust-$version.crate"
do
  if [ ! -f "$archive" ]; then
    cargo package --locked -p "$(basename "$archive" "-$version.crate")"
  fi
  cp "$archive" "$(basename "$archive")"
done
