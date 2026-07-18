#!/bin/sh
set -eu

workflow=.github/workflows/release.yml

step_line() {
  pattern=$1
  line=$(grep -n -F -- "$pattern" "$workflow" | head -n 1 | cut -d: -f1)
  test -n "$line" || {
    echo "Release workflow is missing: $pattern" >&2
    exit 1
  }
  printf '%s\n' "$line"
}

publish=$(step_line "- name: Publish crates.io packages")
verify=$(step_line "- name: Verify canonical crates before creating Git references")
tag=$(step_line "- name: Create annotated version tag after registry publication")
release=$(step_line "- name: Create or verify immutable-ready GitHub Release")

if ! test "$publish" -lt "$verify" \
  || ! test "$verify" -lt "$tag" \
  || ! test "$tag" -lt "$release"
then
  echo "Release order must be publish -> verify -> tag -> GitHub Release" >&2
  exit 1
fi

if grep -R -n -F -- '--clobber' "$workflow" tools/create-or-verify-release.sh; then
  echo "Published Release assets must never be overwritten" >&2
  exit 1
fi
if grep -n -E '^[[:space:]]+release:' "$workflow"; then
  echo "GitHub Release events must not trigger package publication" >&2
  exit 1
fi

echo "Registry-first immutable Release order is enforced"
