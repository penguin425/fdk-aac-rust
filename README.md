# Third-Party Modified Version of the Fraunhofer FDK AAC Codec Library for Android — Rust Port

> **Modification notice (2026-07-14):** This repository is a third-party Rust
> port and modification of the Fraunhofer FDK AAC Codec Library for Android. It
> is not an official Fraunhofer project and is not endorsed by Fraunhofer. The
> original C/C++ source has been replaced by a Rust implementation in the
> tracked source tree; an explicitly pinned upstream revision remains available
> as a build-time reference for tables, compatibility, and differential tests.

This project is derived from the FDK AAC distribution maintained at
[mstorsjo/fdk-aac](https://github.com/mstorsjo/fdk-aac), whose build metadata
identifies the upstream package as `fdk-aac` version 2.0.3. The codec originates
from the Fraunhofer FDK AAC Codec Library for Android and implements MPEG
Advanced Audio Coding (AAC) encoding and decoding.

This repository retains the upstream Git history for provenance and
attribution. As a result, GitHub's **Contributors** list includes authors of
commits inherited from the original C/C++ project; inclusion there does not
necessarily mean that an author directly contributed to this Rust port.

The purpose of this repository is to port that implementation to Rust while
continuously comparing observable behavior with a known upstream Git revision.
It does not claim affiliation with Fraunhofer, Android, mstorsjo, MPEG, ISO, or
IEC.

## Important licensing and patent notice

This is a modified version governed by the
[Software License for the Fraunhofer FDK AAC Codec Library for Android](NOTICE).
The complete license text is retained in [`NOTICE`](NOTICE) and must remain with
source redistributions. Binary redistribution has additional source-availability
and documentation obligations described there.

In particular:

- the Fraunhofer name may not be used to endorse or promote this modified
  version without prior written permission;
- copyright license fees may not be charged for use, copying, or distribution
  of the codec or modifications;
- modified versions must be identified as a **Third-Party Modified Version of
  the Fraunhofer FDK AAC Codec Library for Android** and carry prominent change
  notices;
- **the software license grants no express or implied patent license**;
- use for encoding or decoding MPEG AAC bitstreams may require appropriate
  patent licenses from the relevant patent owners or licensing administrator;
- the software is supplied without warranty, as stated in the full license.

This summary is informational and does not replace [`NOTICE`](NOTICE). Anyone
redistributing or using this project is responsible for reviewing and complying
with the complete license and any applicable patent requirements.
The source and binary redistribution checklist is documented in
[`docs/DISTRIBUTION.md`](docs/DISTRIBUTION.md); project releases intentionally
contain source archives rather than compiled binary artifacts.

[`MODULE_LICENSE_FRAUNHOFER`](MODULE_LICENSE_FRAUNHOFER) is retained from the
upstream Android source distribution as licensing metadata.

## What changed in this Rust port

Compared with the original C/C++ distribution, this repository:

- implements codec, transport, encoder, and decoder components in Rust;
- exposes the Rust implementation through the `fdk-aac-rust` crate;
- keeps an optional `fdk-aac-rust-sys` reference layer for compatibility and
  differential testing;
- does not vendor C/C++ source or header files in the tracked tree;
- fetches a full, pinned upstream commit into Cargo's generated `target/`
  directory when reference source is required;
- tracks the upstream baseline explicitly and tests newer revisions before
  promoting them.

Implementation and restructuring changes are recorded by the repository's Git
history. Detailed migration status is maintained in
[`docs/PORT_STATUS.md`](docs/PORT_STATUS.md) and
[`docs/PURE_RUST_PARITY_ROADMAP.md`](docs/PURE_RUST_PARITY_ROADMAP.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| [`crates/fdk-aac`](crates/fdk-aac) | Safe Rust codec and transport implementation |
| [`crates/fdk-aac-sys`](crates/fdk-aac-sys) | Optional reference FFI used for compatibility and differential tests |
| [`upstream/revision`](upstream/revision) | Full Git SHA of the reference upstream version |
| [`crates/fdk-aac-sys/build-support`](crates/fdk-aac-sys/build-support) | Packaged reference-only capture hooks used by differential tests |
| [`docs/UPSTREAM.md`](docs/UPSTREAM.md) | Upstream source and revision policy |
| [`docs/RELEASING.md`](docs/RELEASING.md) | Versioning and automated release procedure |
| [`docs/PORT_STATUS.md`](docs/PORT_STATUS.md) | Detailed Rust migration status |
| [`docs/DRC.md`](docs/DRC.md) | DRC metadata, loudness targets, and decoder controls |
| [`tools/update-upstream.sh`](tools/update-upstream.sh) | Validated upstream revision update tool |

No `.c`, `.cc`, `.cpp`, or C/C++ header is tracked. With the `ffi` feature,
Cargo fetches the pinned reference source and materializes a small test adapter
under `target/`. All fetched and generated reference files are removed by
`cargo clean`.

## Requirements

- a current stable Rust toolchain;
- Git and network access for the first build, or a compatible local upstream
  source tree supplied through `FDK_AAC_SOURCE_DIR`;
- a C++ compiler for the default `ffi` feature and differential tests.

## Build and test

Test the Rust implementation without compiling or linking the reference C++
library:

```sh
cargo test -p fdk-aac-rust --no-default-features
```

The Rust build currently reads some reference tables from the pinned upstream
source at compile time, so the first build still needs GitHub access or
`FDK_AAC_SOURCE_DIR`.

Run the full workspace, including differential tests against the pinned
reference:

```sh
cargo test --workspace
```

Run formatting and compile checks:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets
```

## Upstream comparison policy

Normal builds use the complete SHA stored in
[`upstream/revision`](upstream/revision). A candidate revision can be tested
without changing that baseline:

```sh
FDK_AAC_REVISION=<full-40-character-sha> cargo test --workspace
```

To resolve the current GitHub `HEAD`, run the required test gates, and update
the pin only after they succeed:

```sh
./tools/update-upstream.sh
```

See [`docs/UPSTREAM.md`](docs/UPSTREAM.md) for revision ordering, failure
handling, and comparison rules.

## Releases

Rust releases use independent semantic versions, annotated `rust-vX.Y.Z` Git
tags, and dated entries in [`CHANGELOG.md`](CHANGELOG.md). The prefix prevents
collisions with the original project's inherited `v0.1.x` and `v2.0.x` tags. A
tag that passes the complete test suite publishes `fdk-aac-rust-sys` followed
by `fdk-aac-rust` to crates.io and creates a GitHub Release containing both
`.crate` source archives and SHA-256 checksums.
See [`docs/RELEASING.md`](docs/RELEASING.md) for the preparation and recovery
procedure.

## Security

Report suspected vulnerabilities privately as described in
[`SECURITY.md`](SECURITY.md). The latest source-level review, remediated findings,
automated checks, and residual risks are recorded in
[`docs/SECURITY_AUDIT.md`](docs/SECURITY_AUDIT.md).

## Status and compatibility

The Rust migration is complete for the public decoder/encoder profiles,
transports, and configurations defined in the project's parity scope. The
project is now in ongoing compatibility-validation and maintenance mode: new
upstream revisions are tested before the pinned reference is advanced.

Completion of the declared migration scope is not a claim that every possible
AAC bitstream, malformed input, private upstream implementation detail,
platform, or future upstream configuration is universally bit-exact.
Configurations intentionally outside the supported public scope return explicit
`Unsupported` errors; those rejections are compatibility boundaries, not
unfinished placeholders. Differential tests and the parity record are the
source of truth for verified behavior.

The original upstream project and its documentation remain the reference for
the C/C++ implementation. Issues in this Rust port should not be reported to
Fraunhofer or the upstream maintainer unless they are independently reproduced
against the unmodified upstream source.
