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

tag="rust-v$version"
tag_sha=$(git rev-list -n 1 "$tag")
if [ "$tag_sha" != "$expected_sha" ]; then
  echo "$tag resolves to $tag_sha, expected $expected_sha" >&2
  exit 1
fi

if [ -n "${GITHUB_REPOSITORY:-}" ]; then
  release_tag=$(gh api "repos/$GITHUB_REPOSITORY/releases/tags/$tag" --jq .tag_name)
  if [ "$release_tag" != "$tag" ]; then
    echo "GitHub Release tag $release_tag does not match $tag" >&2
    exit 1
  fi
fi

"$(dirname "$0")/verify-published-crates.sh" \
  "$version" "$expected_sha" "$artifact_dir"

docs_attempts=${DOCS_RS_ATTEMPTS:-80}
docs_delay=${DOCS_RS_DELAY_SECONDS:-15}
for spec in \
  "fdk-aac-rust/fdk_aac_rust" \
  "fdk-aac-rust-sys/fdk_aac_rust_sys"
do
  crate=${spec%/*}
  module=${spec#*/}
  url="https://docs.rs/$crate/$version/$module/"
  attempt=1
  while [ "$attempt" -le "$docs_attempts" ]; do
    status=$(curl --silent --show-error --output /dev/null \
      --write-out '%{http_code}' --max-redirs 0 "$url" || true)
    if [ "$status" = 200 ]; then
      echo "$crate $version documentation is available from docs.rs"
      break
    fi
    if [ "$attempt" -eq "$docs_attempts" ]; then
      echo "Timed out waiting for successful docs.rs output at $url" >&2
      exit 1
    fi
    sleep "$docs_delay"
    attempt=$((attempt + 1))
  done
done

echo "Release, crates.io packages, and docs.rs all match commit $expected_sha"
