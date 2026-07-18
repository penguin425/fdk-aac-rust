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
    echo "Published $archive differs from the attested release artifact" >&2
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
