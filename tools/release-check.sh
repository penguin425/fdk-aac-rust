#!/bin/sh
set -eu

tag=${1:-}
if ! printf '%s\n' "$tag" | grep -Eq '^rust-v[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "Usage: $0 rust-vMAJOR.MINOR.PATCH" >&2
  exit 1
fi

version=${tag#rust-v}
workspace_version=$(awk -F '"' '
  /^\[workspace.package\]/ { workspace = 1; next }
  /^\[/ { workspace = 0 }
  workspace && /^version = / { print $2; exit }
' Cargo.toml)

if [ "$version" != "$workspace_version" ]; then
  echo "Tag $tag does not match workspace version $workspace_version" >&2
  exit 1
fi

dependency_version=$(sed -n '/^fdk-aac-sys = /s/.*version = "\([^"]*\)".*/\1/p' \
  crates/fdk-aac/Cargo.toml)
if [ "$dependency_version" != "$version" ]; then
  echo "fdk-aac-rust-sys dependency version $dependency_version does not match $version" >&2
  exit 1
fi

if ! grep -q "^## $version - " CHANGELOG.md; then
  echo "CHANGELOG.md has no release entry for $version" >&2
  exit 1
fi

revision=$(tr -d '[:space:]' < upstream/revision)
case "$revision" in
  *[!0-9a-f]*|'')
    echo "upstream/revision is not a lowercase hexadecimal SHA" >&2
    exit 1
    ;;
esac
if [ "${#revision}" -ne 40 ]; then
  echo "upstream/revision must contain a full 40-character SHA" >&2
  exit 1
fi

for build_script in crates/fdk-aac/build.rs crates/fdk-aac-sys/build.rs; do
  if ! grep -q "DEFAULT_UPSTREAM_REVISION: &str = \"$revision\"" "$build_script"; then
    echo "$build_script does not match upstream/revision" >&2
    exit 1
  fi
done

if [ "${REQUIRE_ANNOTATED_TAG:-0}" = 1 ]; then
  if [ "$(git cat-file -t "refs/tags/$tag")" != tag ]; then
    echo "$tag must be an annotated tag" >&2
    exit 1
  fi
fi

./tools/verify-license-files.sh

cargo fmt --all -- --check
cargo check --workspace --all-targets --locked
cargo test -p fdk-aac-rust --no-default-features --locked
cargo test --workspace --locked
cargo package -p fdk-aac-rust-sys --locked --allow-dirty
./tools/verify-license-files.sh "target/package/fdk-aac-rust-sys-$version.crate"
# Cargo replaces the path dependency with crates.io during packaging. Before a
# first release, the matching sys version does not exist there yet, so the main
# crate can only be fully packaged after the sys publication has propagated.
cargo package -p fdk-aac-rust --list --allow-dirty >/dev/null

echo "Release checks passed for $tag"
