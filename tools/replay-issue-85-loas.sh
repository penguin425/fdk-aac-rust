#!/usr/bin/env bash
set -euo pipefail

readonly archive_url="https://github.com/mstorsjo/fdk-aac/files/2152152/test-latm.zip"
readonly archive_sha256="c8f33e639f4afa68bcfaa8e88a986f00c3234ea1b8fd177c31af22b1becd5026"
readonly expected="frames=159 buffered=0 discarded=0"
readonly root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

for command in curl sha256sum unzip; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done

curl --fail --location --silent --show-error \
  --output "${work_dir}/test-latm.zip" "${archive_url}"
printf '%s  %s\n' "${archive_sha256}" "${work_dir}/test-latm.zip" |
  sha256sum --check --strict
unzip -q "${work_dir}/test-latm.zip" -d "${work_dir}/fixture"

cd "${root_dir}"
actual="$(cargo run --locked --quiet -p fdk-aac-rust --no-default-features \
  --example replay_loas -- "${work_dir}/fixture/test.latm" 37)"
if [[ "${actual}" != "${expected}" ]]; then
  echo "unexpected LOAS replay result: ${actual}" >&2
  exit 1
fi
echo "${actual}"
