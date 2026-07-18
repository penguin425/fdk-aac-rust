# Changelog

All notable changes to this third-party Rust port are documented here. Release
versions follow semantic versioning for the Rust crates and do not replace or
reinterpret the version numbers of the original C/C++ fdk-aac project.

## Unreleased

## 0.2.2 - 2026-07-18

- Added Pure Rust CI on Linux x86-64, Windows MSVC, macOS ARM64, Linux ARM64,
  and musl while retaining C/C++ differential tests on Linux x86-64.
- Declared Rust 1.87 as the MSRV and added MSRV tests, crates.io-based SemVer
  API checks, and an explicit sys FFI declaration snapshot gate.
- Added daily libFuzzer campaigns for ADTS, LOAS/LATM, raw access units, ASC,
  USAC/xHE-AAC, DRC, encoder configuration, and the legacy FFI boundary.
- Added CycloneDX SBOMs, GitHub build-provenance attestations, published-commit
  verification, and post-release crates.io/docs.rs availability gates.
- Added a pinned `mpeghdec` static-symbol audit and a tested Linux namespacing
  procedure that links and executes both codec APIs in either archive order.
- Added retained AAC-LC throughput, first-frame latency, peak-RSS, and binary-
  size measurements with intentionally broad catastrophic-regression guards.
- Added black-box external application tests for incremental streaming, seek
  restart, continuous decoding, and an FFmpeg-style C ABI consumer.
- Changed releases to publish and verify the canonical crates.io packages
  before creating the immutable tag, and prohibited Release asset overwrites.

## 0.2.1 - 2026-07-18

- Added automated crates.io publication in dependency order with safe reruns,
  index-propagation checks, and post-publication source archives.
- Added Cargo repository and registry metadata for both public crates.
- Attached both `.crate` source archives and their SHA-256 checksums to GitHub
  Releases, with retained-license verification for every archive.
- Added a consolidated survey and prioritized backlog of public FDK AAC
  improvement requests from upstream users, integrators, and distributions.

## 0.2.0 - 2026-07-17

- Added continuous decoding of real exhale multichannel USAC streams, including
  correct `usacChannelConfigIndex` handling and SCE/CPE/LFE element dispatch.
- Fixed USAC arithmetic-context wrapping and CPE bit parsing to match the
  pinned FDK reference implementation.
- Hardened HE-AACv2 FFI input buffering, ADTS configuration-change detection,
  decoder interruption state, and dynamic HE-AACv2 stream information.
- Added fixed-point extrema, distinct-signal 5.1, upstream crash-corpus, and
  incremental LOAS regression coverage.
- Added decoder delay, DRC, upstream-issue, Android 17 xHE-AAC source, and
  release-maintenance documentation plus a reproducible AAC-LC benchmark.

## 0.1.1 - 2026-07-17

- Added checked length and result conversions at the optional C FFI boundary.
- Hardened generated upstream checkouts against stale files and redirected
  cache paths.
- Added continuous dependency, advisory, secret, and FFI security checks.
- Pinned GitHub Actions to immutable revisions and enabled automated dependency
  update proposals.

## 0.1.0 - 2026-07-15

- Completed the declared Pure Rust migration scope for the supported public
  decoder, encoder, profile, and transport configurations.
- Added optional `fdk-aac-rust-sys` compatibility wrappers and differential tests
  against a pinned upstream fdk-aac revision.
- Removed vendored C/C++ source and header files from the tracked repository.
- Added reproducible upstream revision validation and update tooling.
- Added Cargo workspace packaging for `fdk-aac-rust` and `fdk-aac-rust-sys`.
- Retained the complete Fraunhofer FDK AAC license and prominent third-party
  modification notices.
