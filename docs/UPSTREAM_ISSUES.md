# Upstream open-issue audit

This document records the review of every open issue in
[`mstorsjo/fdk-aac`](https://github.com/mstorsjo/fdk-aac/issues) as of
2026-07-17. The comparison baseline is the C/C++ revision in
[`upstream/revision`](../upstream/revision), while applicability is decided
against the Pure Rust implementation and its optional FFI compatibility layer.

An upstream issue remaining open does not by itself prove that this port is
affected. Each issue is assigned one of these dispositions:

- **Fix**: implement or harden behavior in a dedicated pull request.
- **Verify**: add a conformance, regression, fuzz, or benchmark check; fix only
  if the check demonstrates a defect.
- **Document**: clarify a supported constraint or public workflow.
- **Track**: investigate a distinct future source or feature independently.
- **Not applicable**: the report is in a CLI, container integration, build
  system, obsolete C/C++ revision, or unsupported external environment.
- **Insufficient reproduction**: no actionable input or correct calling
  sequence is available; reconsider when one is supplied.

## Actionable issues

| Upstream | Disposition | Rust-port action |
|---|---|---|
| [#180](https://github.com/mstorsjo/fdk-aac/issues/180) | Track | Review the Android 17 xHE-AAC encoder as a separately licensed/source-controlled candidate; do not silently move the current baseline. |
| [#177](https://github.com/mstorsjo/fdk-aac/issues/177) | Verified ([PR #10](https://github.com/penguin425/fdk-aac-rust/pull/10)) | Distinct-signal 5.1 encode/decode coverage proves all six channels, including the reported fourth channel, remain audible for every supported channel order. |
| [#172](https://github.com/mstorsjo/fdk-aac/issues/172) | Documented ([PR #12](https://github.com/penguin425/fdk-aac-rust/pull/12)) | Explain target loudness, encoder reference level, DRC instruction selection, gain scaling, and observable stream-info fields. Existing unit tests exercise legacy and unified DRC gain paths. |
| [#171](https://github.com/mstorsjo/fdk-aac/issues/171) | Documented ([PR #8](https://github.com/penguin425/fdk-aac-rust/pull/8)) | The apparent one-frame offset is codec preroll/overlap unless a complete fixture demonstrates otherwise; document output delay and trimming instead of deleting decoded PCM blindly. |
| [#170](https://github.com/mstorsjo/fdk-aac/issues/170) | Documented ([PR #8](https://github.com/penguin425/fdk-aac-rust/pull/8)) | The API reports algorithmic output delay; the guide explains LC/HE preroll and why low-delay profiles are a format choice rather than a decoder switch. |
| [#160](https://github.com/mstorsjo/fdk-aac/issues/160) | Documented ([PR #8](https://github.com/penguin425/fdk-aac-rust/pull/8)) | Describe delay trimming for library users; this port does not ship the `fdkaac` CLI requested upstream. |
| [#154](https://github.com/mstorsjo/fdk-aac/issues/154) | Fixed ([PR #9](https://github.com/penguin425/fdk-aac-rust/pull/9)) | Every ADTS frame facade now rejects sample-rate or channel-layout changes explicitly rather than retaining stale output state. |
| [#152](https://github.com/mstorsjo/fdk-aac/issues/152) | Verified ([PR #7](https://github.com/penguin425/fdk-aac-rust/pull/7)) | Rust signed shifts have defined semantics; extrema tests cover negative and `i16::MIN` conversion plus full-scale LC/HE/HEv2 encode paths. The reported C expression is not used by the Pure Rust path. |
| [#151](https://github.com/mstorsjo/fdk-aac/issues/151) | Verified ([PR #15](https://github.com/penguin425/fdk-aac-rust/pull/15)) | Add a deterministic release-mode AAC-LC decoder throughput benchmark for this independent implementation. |
| [#148](https://github.com/mstorsjo/fdk-aac/issues/148) | Explained + verified ([PR #8](https://github.com/penguin425/fdk-aac-rust/pull/8)) | A fresh decoder cannot reconstruct the previous MDCT overlap when starting at frame five. One-shot drain coverage and random-access guidance prevent treating missing predecessor state as a flush bug. |
| [#129](https://github.com/mstorsjo/fdk-aac/issues/129) | Fixed ([PR #6](https://github.com/penguin425/fdk-aac-rust/pull/6)) | The safe HE-AACv2 FFI encoder buffers incomplete PCM and never invokes the vulnerable C path with a short input slice. |
| [#120](https://github.com/mstorsjo/fdk-aac/issues/120) | Partially fixed ([PR #11](https://github.com/penguin425/fdk-aac-rust/pull/11)) | A real exhale 1.2.2 5.1 ASC exposed misuse of outer `channelConfiguration=0`; initialization now uses `usacChannelConfigIndex=6`. Real access units also identify remaining TCX/FD conformance work, so full exhale compatibility remains tracked. |
| [#114](https://github.com/mstorsjo/fdk-aac/issues/114) | Documented ([PR #8](https://github.com/penguin425/fdk-aac-rust/pull/8)) | Document preroll, random access, `signal_interruption()`, transport resynchronization, and container trimming responsibilities. |
| [#89](https://github.com/mstorsjo/fdk-aac/issues/89) | Verified ([PR #13](https://github.com/penguin425/fdk-aac-rust/pull/13)) | A checksum-pinned, timeout-bounded harness replays all 39 historical malformed ADTS inputs against the pinned FDK reference; none crashes. |
| [#85](https://github.com/mstorsjo/fdk-aac/issues/85) | Verified ([PR #14](https://github.com/penguin425/fdk-aac-rust/pull/14)) | The valid upstream LOAS fixture decodes all 159 frames when fed in 37-byte chunks with no discarded or buffered tail. The original MCP1 data has no LOAS sync layer and requires external packet boundaries. |
| [#78](https://github.com/mstorsjo/fdk-aac/issues/78) | Already handled | ER AAC-LD is supported, while AAC-LD LTP returns the specific `LtpUnsupported` error and has mono/stereo regression coverage. |
| [#43](https://github.com/mstorsjo/fdk-aac/issues/43) | Verified ([PR #7](https://github.com/penguin425/fdk-aac-rust/pull/7)) | Full-scale LC/HE/HEv2 tests cover the reported fixed-point boundary class. The DVD source from the historical encoder assertion is unavailable for exact replay. |
| [#25](https://github.com/mstorsjo/fdk-aac/issues/25) | Fix + verify | Support and test HE-AACv2 output changing from provisional mono/core rate to stereo/output rate without stale stream information. |
| [#16](https://github.com/mstorsjo/fdk-aac/issues/16) | Documented ([PR #8](https://github.com/penguin425/fdk-aac-rust/pull/8)) | State that raw AAC access units require external packet boundaries and one complete access unit per decode call. |

## Reviewed issues that do not currently require a codec fix

| Upstream | Disposition | Reason |
|---|---|---|
| [#175](https://github.com/mstorsjo/fdk-aac/issues/175) | Not applicable | Report concerns `fdkaac` bitrate selection/interpretation and does not provide evidence of a codec defect. |
| [#174](https://github.com/mstorsjo/fdk-aac/issues/174) | Not applicable | FFmpeg `BitRate_Mode` metadata bug, fixed in FFmpeg commit `46c6ca3`. |
| [#173](https://github.com/mstorsjo/fdk-aac/issues/173) | Not applicable | Host C++ preprocessor/autotools installation failure; the Rust build uses `cc` and a pinned generated checkout. |
| [#169](https://github.com/mstorsjo/fdk-aac/issues/169) | Insufficient reproduction | Arduino wrapper reports an invalid handle without configuration, input, or a standalone reproduction. |
| [#166](https://github.com/mstorsjo/fdk-aac/issues/166) | Not applicable | C/C++ `-mcpu` optimization/build-size question. |
| [#165](https://github.com/mstorsjo/fdk-aac/issues/165) | Not applicable | This repository has its own security policy and private vulnerability reporting. |
| [#163](https://github.com/mstorsjo/fdk-aac/issues/163) | Not applicable | Explicitly off-topic discussion. |
| [#162](https://github.com/mstorsjo/fdk-aac/issues/162) | Not applicable | Disabled/incorrect upstream ARM assembly optimization; Pure Rust does not compile that assembly. |
| [#161](https://github.com/mstorsjo/fdk-aac/issues/161) | Insufficient reproduction | Device-specific playback failure with no bitstream or library-level reproduction. |
| [#157](https://github.com/mstorsjo/fdk-aac/issues/157) | Not applicable | Upstream compilation support question. |
| [#156](https://github.com/mstorsjo/fdk-aac/issues/156) | Not applicable | C++ ODR warnings under LTO; no corresponding duplicated C++ types exist in Pure Rust. |
| [#155](https://github.com/mstorsjo/fdk-aac/issues/155) | Not applicable | Upstream release-history question. |
| [#153](https://github.com/mstorsjo/fdk-aac/issues/153) | Not applicable | Upstream release-version request. Rust versions are independent and documented. |
| [#150](https://github.com/mstorsjo/fdk-aac/issues/150) | Not applicable | Android NDK integration question. |
| [#149](https://github.com/mstorsjo/fdk-aac/issues/149) | Documented constraint | Public encoder input is 16-bit PCM or normalized `f32`; 24-bit container conversion belongs to the caller. |
| [#147](https://github.com/mstorsjo/fdk-aac/issues/147) | Insufficient reproduction | CLI pipe report contains no codec call sequence or reusable input. |
| [#145](https://github.com/mstorsjo/fdk-aac/issues/145) | Insufficient reproduction | Cross-encoder AAC-LD report has no retained fixture/configuration. |
| [#144](https://github.com/mstorsjo/fdk-aac/issues/144) | Not applicable | Upstream source build failure. |
| [#143](https://github.com/mstorsjo/fdk-aac/issues/143) | Not applicable | Upstream changelog maintenance request. |
| [#141](https://github.com/mstorsjo/fdk-aac/issues/141) | Not applicable | C static-library size/configuration question. |
| [#134](https://github.com/mstorsjo/fdk-aac/issues/134) | Not applicable | Cross-platform CMake/version-resource request. |
| [#128](https://github.com/mstorsjo/fdk-aac/issues/128) | Not applicable | Upstream installation question. |
| [#126](https://github.com/mstorsjo/fdk-aac/issues/126) | Documented constraint | 320-sample AAC-ELD frames are not a supported FDK frame geometry; the Rust configuration rejects unsupported lengths. |
| [#125](https://github.com/mstorsjo/fdk-aac/issues/125) | Not applicable | Patent/license ownership question; this port preserves the upstream notices and does not grant patent rights. |
| [#124](https://github.com/mstorsjo/fdk-aac/issues/124) | Not applicable | NDK cross-compilation file-selection problem. |
| [#122](https://github.com/mstorsjo/fdk-aac/issues/122) | Insufficient reproduction | PHP FFI pointer construction is incomplete and no input is available; safe Rust wrappers do not expose this pointer shape. |
| [#108](https://github.com/mstorsjo/fdk-aac/issues/108) | Not applicable | Standalone C example compilation problem. |
| [#103](https://github.com/mstorsjo/fdk-aac/issues/103) | Not applicable | Visual Studio 2013 C++ static-library integration problem. |
| [#102](https://github.com/mstorsjo/fdk-aac/issues/102) | Documented constraint | AAC-ELD cannot be carried in ADTS; the encoder configuration enforces compatible transports. |
| [#87](https://github.com/mstorsjo/fdk-aac/issues/87) | No demonstrated defect | Reporter observed the same ELD quality with raw and LOAS; discussion identifies profile/bitrate suitability, not transport corruption. |
| [#80](https://github.com/mstorsjo/fdk-aac/issues/80) | Insufficient reproduction | The byte array lacks transport type and calling sequence and appears to include non-AAC framing. |
| [#74](https://github.com/mstorsjo/fdk-aac/issues/74) | Not applicable | Termux installation question. |
| [#49](https://github.com/mstorsjo/fdk-aac/issues/49) | Not applicable | C static-library size question. |
| [#35](https://github.com/mstorsjo/fdk-aac/issues/35) | Not applicable | Trailing gapless trim is MP4/container metadata behavior; this project intentionally does not ship an MP4 writer. |
| [#15](https://github.com/mstorsjo/fdk-aac/issues/15) | Insufficient reproduction | Historical memory-usage question without a measurable configuration or current failure. |

## Maintenance rule

When an actionable row is completed, link its Rust pull request in this table
and state whether the result was a code fix, regression coverage, documentation,
or a demonstrated non-applicability. Re-audit newly opened upstream issues
before each Rust minor release and whenever `upstream/revision` moves.
