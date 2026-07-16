#!/bin/sh
set -eu

upstream_url=https://github.com/mstorsjo/fdk-aac.git
revision_file="$(dirname "$0")/../upstream/revision"

if [ -n "${FDK_AAC_SOURCE_DIR:-}" ]; then
  echo "Unset FDK_AAC_SOURCE_DIR before updating the upstream revision" >&2
  exit 1
fi

current=$(tr -d '[:space:]' < "$revision_file")
latest=$(git ls-remote "$upstream_url" HEAD | awk '{print $1}')

if [ -z "$latest" ]; then
  echo "Could not resolve upstream HEAD" >&2
  exit 1
fi

if [ "$current" = "$latest" ]; then
  echo "Already at upstream HEAD: $current"
  exit 0
fi

echo "Testing upstream revision $latest (current: $current)"
FDK_AAC_REVISION="$latest" cargo test --workspace
FDK_AAC_REVISION="$latest" cargo test -p fdk-aac-rust --no-default-features
cargo fmt --all -- --check

printf '%s\n' "$latest" > "$revision_file"
for build_script in \
  "$(dirname "$0")/../crates/fdk-aac/build.rs" \
  "$(dirname "$0")/../crates/fdk-aac-sys/build.rs"
do
  temporary="$build_script.tmp"
  awk -v revision="$latest" '
    /^const DEFAULT_UPSTREAM_REVISION: &str = / {
      print "const DEFAULT_UPSTREAM_REVISION: &str = \"" revision "\";"
      next
    }
    { print }
  ' "$build_script" > "$temporary"
  mv "$temporary" "$build_script"
done
echo "Updated $revision_file to $latest"
