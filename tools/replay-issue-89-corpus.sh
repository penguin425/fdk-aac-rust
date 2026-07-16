#!/usr/bin/env bash
set -euo pipefail

readonly archive_url="https://github.com/mstorsjo/fdk-aac/files/2219859/crashes.zip"
readonly archive_sha256="702a973c42c15116e69c01cba2bfe5db8e1852ddad419c08ede74526084549d1"
readonly root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

for command in curl sha256sum unzip timeout; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done

curl --fail --location --silent --show-error \
  --output "${work_dir}/crashes.zip" "${archive_url}"
printf '%s  %s\n' "${archive_sha256}" "${work_dir}/crashes.zip" |
  sha256sum --check --strict
unzip -q "${work_dir}/crashes.zip" -d "${work_dir}/corpus"

cd "${root_dir}"
cargo build --locked --quiet -p fdk-aac-rust --example replay_adts

total=0
while IFS= read -r -d '' input; do
  total=$((total + 1))
  timeout 10 target/debug/examples/replay_adts "${input}"
done < <(find "${work_dir}/corpus" -type f -name input -print0)

if ((total != 39)); then
  echo "expected 39 upstream inputs, found ${total}" >&2
  exit 1
fi
echo "replayed ${total} upstream issue #89 inputs without a crash"
