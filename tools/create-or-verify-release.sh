#!/bin/sh
set -eu

version=${1:-}
expected_sha=${2:-}
if ! printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "Usage: $0 MAJOR.MINOR.PATCH GIT_SHA" >&2
  exit 1
fi
if ! printf '%s\n' "$expected_sha" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "GIT_SHA must be a full lowercase 40-character commit SHA" >&2
  exit 1
fi

tag="rust-v$version"
gh_release() {
  if [ -n "${GITHUB_REPOSITORY:-}" ]; then
    gh release "$@" --repo "$GITHUB_REPOSITORY"
  else
    gh release "$@"
  fi
}
expected_assets="$(printf '%s\n' \
  SHA256SUMS \
  "fdk-aac-rust-$version.crate" \
  "fdk-aac-rust-$version.tar.gz" \
  fdk-aac-rust-sys.cdx.json \
  "fdk-aac-rust-sys-$version.crate" \
  fdk-aac-rust.cdx.json | sort)"

test "$(git rev-list -n 1 "$tag")" = "$expected_sha"

if gh_release view "$tag" >/dev/null 2>&1; then
  actual_assets=$(gh_release view "$tag" --json assets --jq '.assets[].name' | sort)
  if [ "$actual_assets" != "$expected_assets" ]; then
    echo "Existing $tag release has a different asset set; refusing to modify it" >&2
    exit 1
  fi
  tmp_dir=$(mktemp -d)
  trap 'find "$tmp_dir" -type f -delete; rmdir "$tmp_dir"' EXIT HUP INT TERM
  gh_release download "$tag" --dir "$tmp_dir"
  for asset in $expected_assets; do
    if ! cmp --silent "$asset" "$tmp_dir/$asset"; then
      echo "Existing $tag asset $asset differs; refusing to overwrite it" >&2
      exit 1
    fi
  done
  echo "Existing $tag release and assets are unchanged and verified"
  exit 0
fi

gh_release create "$tag" \
  "fdk-aac-rust-$version.tar.gz" \
  "fdk-aac-rust-sys-$version.crate" \
  "fdk-aac-rust-$version.crate" \
  fdk-aac-rust-sys.cdx.json fdk-aac-rust.cdx.json \
  SHA256SUMS \
  --verify-tag \
  --title "fdk-aac Rust port $version" \
  --notes-file RELEASE_NOTES.md
