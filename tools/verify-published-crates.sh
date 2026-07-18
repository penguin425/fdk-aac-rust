#!/bin/sh
set -eu

version=${1:-}
expected_sha=${2:-}
artifact_dir=${3:-.}

if ! printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  echo "Usage: $0 MAJOR.MINOR.PATCH GIT_SHA [ARTIFACT_DIR]" >&2
  exit 1
fi
if ! printf '%s\n' "$expected_sha" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "GIT_SHA must be a full lowercase 40-character commit SHA" >&2
  exit 1
fi

tmp_dir=$(mktemp -d)
trap 'find "$tmp_dir" -type f -delete; rmdir "$tmp_dir"' EXIT HUP INT TERM

for crate in fdk-aac-rust-sys fdk-aac-rust; do
  archive="$crate-$version.crate"
  local_archive="$artifact_dir/$archive"
  published_archive="$tmp_dir/$archive"
  test -f "$local_archive"
  curl --fail --location --silent --show-error \
    --retry 10 --retry-delay 10 --retry-all-errors \
    --user-agent "fdk-aac-rust-release-verifier" \
    --output "$published_archive" \
    "https://crates.io/api/v1/crates/$crate/$version/download"
  if ! cmp --silent "$local_archive" "$published_archive"; then
    echo "Published $archive differs from the local release artifact" >&2
    exit 1
  fi
  vcs_sha=$(
    tar -xOf "$published_archive" "$crate-$version/.cargo_vcs_info.json" |
      python3 -c 'import json, sys; print(json.load(sys.stdin)["git"]["sha1"])'
  )
  if [ "$vcs_sha" != "$expected_sha" ]; then
    echo "$archive records Git commit $vcs_sha, expected $expected_sha" >&2
    exit 1
  fi
done

echo "crates.io packages match commit $expected_sha"
