# Changelog

All notable changes to this third-party Rust port are documented here. Release
versions follow semantic versioning for the Rust crates and do not replace or
reinterpret the version numbers of the original C/C++ fdk-aac project.

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
