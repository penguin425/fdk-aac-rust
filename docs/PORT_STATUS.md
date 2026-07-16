# Pure Rust FDK AAC port

This document records the completed Pure Rust codec/transport migration plus
the optional safe wrappers around the original Fraunhofer FDK AAC
implementation used for continuing differential validation.

For the completed parity scope, validation evidence, and historical migration
checklist, see
[`PURE_RUST_PARITY_ROADMAP.md`](PURE_RUST_PARITY_ROADMAP.md).
For source selection, revision advancement, and failure handling, see
[`UPSTREAM.md`](UPSTREAM.md).

- `fdk-aac-rust` builds without C/C++ using `--no-default-features`; codec algorithms,
  transports, AAC/ER/ELD/USAC decoding, AAC-LC/HE-AAC/PS encoding, and DRM HCR
  paths are implemented in Rust.
- The default `ffi` feature retains RAII wrappers and C/C++ differential tests.
  The C/C++ tree is not vendored: build scripts fetch the revision pinned in
  [`upstream/revision`](../upstream/revision) from
  `https://github.com/mstorsjo/fdk-aac.git` into Cargo's `OUT_DIR`.
- `fdk-aac-rust-sys` is optional for users of the Pure Rust API.

For offline builds or testing another checkout, set `FDK_AAC_SOURCE_DIR` to an
existing fdk-aac source tree. The automatic checkout used by `fdk-aac-sys`
also applies `crates/fdk-aac-sys/build-support/test-bridge.patch`, which contains only the capture
hooks required by the differential tests.

No `.c`, `.cc`, `.cpp`, or C/C++ header is kept in the repository. The small
differential-test adapter is stored as `crates/fdk-aac-sys/build-support/qmf-test-wrapper.bridge`
and materialized as a temporary `.cpp` file under Cargo's `OUT_DIR` only while
the upstream reference library is compiled.

### Advancing the upstream comparison revision

The committed SHA is intentionally fixed so normal builds stay reproducible.
To test a particular full commit SHA without changing the pin:

```sh
FDK_AAC_REVISION=<40-character-commit-sha> cargo test --workspace
```

To resolve the current GitHub `HEAD`, run both the differential and Pure Rust
test suites against it, and update the pin only after they pass:

```sh
./tools/update-upstream.sh
```

Commit each successful `upstream/revision` change separately. If an upstream
change conflicts with `crates/fdk-aac-sys/build-support/test-bridge.patch` or changes observable codec
behavior, the update stops before changing the pin; adapt the Rust port and
bridge, rerun the tests, then promote that revision.

The detailed procedure, including environment-variable precedence and bisecting
multiple candidate revisions, is maintained in [`UPSTREAM.md`](UPSTREAM.md).

Completed Pure Rust migration scope:

- [x] ADTS header parse/write and frame slicing
- [x] AudioSpecificConfig / GASpecificConfig parsing for common AAC-LC/HE-AAC configs
- [x] Program Config Element parsing/writing for ASC channelConfiguration 0
- [x] reusable Pure Rust bitstream reader/writer primitives
- [x] ADTS stream iterator and raw access-unit extraction
- [x] Raw decoder configuration from Pure Rust ADTS/ASC parsing
- [x] AAC raw_data_block top-level element identification and PCE/FIL/DSE parsing
- [x] AAC-LC ICS side-info parsing and SCE/LFE/CPE side-info prefixes
- [x] AAC-LC section_data parsing through codebook grids
- [x] AAC-LC scale_factor_data traversal, scalefactor Huffman table decoding, and accumulator logic
- [x] AAC-LC spectral Huffman codebook tables 1-11 and FDK-style table metadata
- [x] AAC-LC spectral Huffman tuple expansion, sign bits, ESCBOOK escape handling, and grouped-window placement skeleton
- [x] AAC-LC 1024/128 scalefactor band offset tables and spectral decoder lookup integration
- [x] AAC-LC f32 reference inverse quantization from scalefactors and spectral coefficients
- [x] AAC-LC pulse_data parsing/application and TNS syntax/f32 reference filtering
- [x] AAC-LC CPE MS stereo mask parsing and f32 spectrum reconstruction
- [x] AAC-LC CPE intensity stereo f32 right-channel reconstruction
- [x] AAC-LC PNS f32 noise generation with CPE correlation support
- [x] AAC-LC SCE/LFE single-channel f32 decode orchestration through filterbank
- [x] AAC-LC CPE common-window/two-channel stream decode orchestration through PNS/TNS spectra
- [x] AAC-LC CPE MS/intensity stereo orchestration and stereo f32 filterbank output
- [x] stateful Pure Rust AAC-LC raw/ADTS first-audio-element f32 decoder dispatch
- [x] raw_data_block DSE/FIL/PCE skip before first AAC-LC audio element dispatch
- [x] Pure Rust AAC-LC f32/i16 interleaved frame output helpers
- [x] ADTS stream decoded-frame iterator helpers for interleaved f32/i16 output
- [x] ADTS stream multi-element/multichannel iterator helpers for frame and interleaved output
- [x] AAC-LC channel_configuration 1/2 validation for stateful Pure Rust decoder
- [x] AAC-LC multi-element raw_data_block assembly for channel_configuration 1-7 in bitstream order
- [x] AAC-LC channel labels for channel_configuration 1-7 and PCE-derived layouts
- [x] PCE tag_select matching against decoded SCE/CPE/LFE element_instance_tag
- [x] AAC-LC CCE prefix parser for coupling target/gain header syntax
- [x] Pure Rust decoder surfaces parsed CCE prefix metadata in unsupported-coupling errors
- [x] AAC-LC CCE coupled channel_stream decode and gain_element list parsing scaffold
- [x] AAC-LC CCE zero-gain/no-op coupling application scaffold with explicit non-zero gain error
- [x] AAC-LC CCE frequency-domain common-gain spectrum application core
- [x] AAC-LC CCE bandwise frequency gain, time-domain sample gain, target-spectrum matching, gain_element_scale, and independently-switched gain-list handling
- [x] AAC-LC multichannel raw_data_block staging path with frequency-domain CCE spectrum application before filterbank
- [x] AAC-LC legacy raw_data_block decode delegates to staging path and multichannel path integrates time-domain CCE sample application
- [x] AAC-LC f32 reference long/start/stop/short IMDCT, sine/KBD windows, and overlap-add skeleton
- [x] planned f32 IMDCT path with reusable cosine kernel replacing per-sample trig loop
- [x] initial libFDK-vs-Pure-Rust decode parity harness for ADTS AAC-LC silence fixtures
- [x] libFDK-vs-Pure-Rust parity coverage for raw-config payload decode and two-frame ADTS stream iteration
- [x] libFDK-vs-Pure-Rust parity coverage for zero SCE mono, PCE channelConfiguration=0, zero CCE, and non-zero pulse smoke fixtures
- [x] PCM delta-report helper for non-zero libFDK-vs-Pure-Rust fixture analysis
- [x] Pure Rust transport decoder facade unifies ASC-configured raw access units and ADTS frames
- [x] unsupported ADIF/LATM/LOAS/DRM transport selections return explicit Pure Rust transport errors
- [x] first Pure Rust fixed-point DSP primitives and PCM/SGL/DBL scaling helpers
- [x] reference Q31 fixed-point IMDCT kernel/plan as first fixed-point filterbank building block
- [x] Q31 fixed-point sine/KBD window conversion, block-switching windows, and long-window overlap-add path
- [x] Q31 fixed-point eight-short window overlap-add and fixed-point AAC-LC synthesis dispatcher
- [x] bridge from inverse-quantized f32 spectra to Q31 fixed-point synthesis path
- [x] AacLcDecoder fixed-point filterbank state and channel/CCE stream synthesis helpers to Q31/i16
- [x] raw SCE/CPE fixed-point i16 decode helpers backed by Q31 filterbanks
- [x] AacLcDecoder stateful raw/ADTS fixed-point interleaved i16 APIs for direct SCE/CPE frames
- [x] ADTS stream iterator for stateful fixed-point interleaved i16 SCE/CPE decode
- [x] PCE/channelConfiguration=0 coverage for ADTS fixed-point SCE decode path
- [x] fixed-point multichannel staged raw/ADTS/stream interleaved i16 decode path with frequency CCE support
- [x] fixed-point multichannel time-domain CCE sample application path
- [x] fixed CPE path uses a fixed-style MS stereo bridge before fixed synthesis
- [x] fixed CPE path uses a fixed-style intensity stereo bridge before fixed synthesis
- [x] FixedInverseQuantizedSpectrum Q31 bridge type and fixed filterbank entrypoint
- [x] fixed inverse quantization bridge helpers producing FixedInverseQuantizedSpectrum
- [x] decoder-side fixed single-channel spectrum structs and raw SCE fixed-spectrum bridge decode path
- [x] fixed-spectrum PNS bridge for single-channel fixed decode path
- [x] fixed-spectrum TNS bridge for single-channel fixed decode path
- [x] fixed-spectrum CPE bridge structs/path, pair PNS, pair TNS, and fixed-spectrum stereo tool bridge
- [x] fixed-spectrum CCE bridge structs/path and frequency coupling application bridge
- [x] fixed-spectrum CPE bridge-to-fixed-filterbank interleaved i16 decode API
- [x] multichannel fixed decode staging now uses fixed-spectrum SCE/CPE/CCE bridge paths
- [x] integer fixed inverse-quantization fast path for all scalefactor remainders
- [x] non-bridge fixed inverse quantization API with bridge aliases retained
- [x] fixed PNS noise generation, normalization, and gain application use integer Q31 helpers
- [x] fixed TNS lattice synthesis uses integer Q31 coefficients/state instead of f32 bridge
- [x] fixed PNS/TNS non-bridge APIs with bridge aliases retained for compatibility
- [x] libFDK parity/smoke tests for ADTS/raw zero SCE/CPE/PCE/CCE and nonzero pulse fixtures
- [x] remaining fixed-point AAC-LC core DSP primitives currently represented in Rust fixed path
- [x] strict raw/ADTS AAC-LC decode APIs reject non-zero trailing payload bits
- [x] strict ADTS stream iterator variants for f32/i16/fixed and multichannel decode paths
- [x] AAC LC bitstream decode path for currently supported raw/ADTS SCE/CPE/PCE/CCE AAC-LC subset
- [x] unsupported advanced AAC syntax coverage for prediction, gain control, non-LC AOT, and multi-raw-block ADTS
- [x] nonzero spectral raw SCE smoke fixture exercises section/scalefactor/spectral/inverse/filterbank path
- [x] incremental LOAS buffering, byte-aligned synchronization recovery, and f32/fixed-i16 LATM AAC-LC stream dispatch
- [x] incremental ADIF header/raw_data_block buffering with transactional f32/fixed-i16 decode
- [x] ADTS CBR loss estimation and optional f32/fixed-i16 PCM repeat/fade concealment
- [x] fixed-spectrum ADTS concealment before IMDCT with filterbank overlap preservation
- [x] transactional ADTS look-ahead for surrounding-good-frame Q31 spectral interpolation
- [x] transactional ADTS look-ahead for f32 surrounding-good-frame spectral interpolation
- [x] Supported C/C++ FDK transport/profile/encoder parity; detailed evidence is maintained in `PURE_RUST_PARITY_ROADMAP.md`

## Build and test

```sh
cargo test --workspace
```

No CMake/autotools installation is required for the Rust build path; the sys crate
uses the `cc` crate to compile the source files listed in the top-level
`CMakeLists.txt`.

### Quality gates

Line coverage is a diagnostic, not the acceptance criterion for this port. In
particular, tests should not replace fallible bitstream handling with panics just
to execute an otherwise valid error branch. Changes should pass these gates:

```sh
# Pure Rust implementation, without compiling or linking the C/C++ library.
cargo test -p fdk-aac-rust --no-default-features

# Full workspace, including C/C++ FDK differential tests fetched from GitHub.
cargo test --workspace

# Focused C/C++-versus-Rust PCM parity tests.
cargo test -p fdk-aac-rust ffi::tests

# Deterministic malformed-input and incremental-parser panic checks.
cargo test -p fdk-aac-rust --no-default-features never_panic

cargo fmt --all -- --check
```

The differential suite covers exact PCM comparisons for zero AAC-LC fixtures
and correlation/RMS comparisons for non-zero ER AAC-LD and ELD-SBR streams.
When `cargo-mutants` is available, mutation testing is also useful for judging
whether assertions detect behavioral changes; it is intentionally an additional
signal rather than a requirement to obtain a particular coverage percentage.
