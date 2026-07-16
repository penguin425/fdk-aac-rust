#!/bin/sh
set -eu

expected_notice_sha256=95ec80da40b4af12ad4c4f3158c9cfb80f2479f3246e4260cb600827cc8c7836
modified_name='Third-Party Modified Version of the Fraunhofer FDK AAC Codec Library for Android'

actual_notice_sha256=$(sha256sum NOTICE | awk '{print $1}')
if [ "$actual_notice_sha256" != "$expected_notice_sha256" ]; then
  echo "NOTICE is not the retained Fraunhofer FDK AAC license text" >&2
  exit 1
fi

if ! grep -Fq "$modified_name" README.md; then
  echo "README.md is missing the required modified-version name" >&2
  exit 1
fi
if ! grep -Eq '^> \*\*Modification notice \([0-9]{4}-[0-9]{2}-[0-9]{2}\):\*\*' README.md; then
  echo "README.md is missing a dated, prominent modification notice" >&2
  exit 1
fi
if ! grep -Fq 'the software license grants no express or implied patent license' README.md; then
  echo "README.md is missing the patent-license warning" >&2
  exit 1
fi

for manifest in crates/fdk-aac/Cargo.toml crates/fdk-aac-sys/Cargo.toml; do
  if ! grep -Fq 'license-file = "../../NOTICE"' "$manifest"; then
    echo "$manifest does not declare the retained NOTICE file" >&2
    exit 1
  fi
done

for archive in "$@"; do
  if [ ! -f "$archive" ]; then
    echo "Package archive not found: $archive" >&2
    exit 1
  fi
  archive_listing=$(tar -tzf "$archive")
  prefix=$(printf '%s\n' "$archive_listing" | sed -n '1{s|/.*||;p;}')
  if [ -z "$prefix" ]; then
    echo "Package archive is empty: $archive" >&2
    exit 1
  fi
  for file in NOTICE README.md; do
    if ! printf '%s\n' "$archive_listing" | grep -Fxq "$prefix/$file"; then
      echo "$archive does not contain $file" >&2
      exit 1
    fi
    if ! tar -xOzf "$archive" "$prefix/$file" | cmp -s - "$file"; then
      echo "$archive contains a modified or incomplete $file" >&2
      exit 1
    fi
  done
done

echo "License distribution checks passed"
