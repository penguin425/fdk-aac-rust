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
| [#177](https://github.com/mstorsjo/fdk-aac/issues/177) | Verify | Encode channel-distinct 5.1 PCM and prove that no channel is lost or reordered for both channel-order modes. |
| [#172](https://github.com/mstorsjo/fdk-aac/issues/172) | Verify + document | Add an encode/decode DRC target-level fixture and a public metadata example. |
| [#171](https://github.com/mstorsjo/fdk-aac/issues/171) | Verify | Compare leading samples, decoder delay, and total output length against the reference decoder. |
| [#170](https://github.com/mstorsjo/fdk-aac/issues/170) | Document | Report HE-AAC delay through the API and explain that LC/HE/LD/ELD have different algorithmic-delay tradeoffs. |
| [#160](https://github.com/mstorsjo/fdk-aac/issues/160) | Document | Describe delay trimming for users of the library; this port does not ship the `fdkaac` CLI requested upstream. |
| [#154](https://github.com/mstorsjo/fdk-aac/issues/154) | Fix | Detect ADTS configuration changes per frame and either reconfigure safely or return an explicit configuration-change error. Never retain a stale output shape. |
| [#152](https://github.com/mstorsjo/fdk-aac/issues/152) | Fix + verify | Audit fixed-point signed shifts and overflow boundaries; add sanitizer-equivalent boundary tests for both the Rust and optional C paths. |
| [#151](https://github.com/mstorsjo/fdk-aac/issues/151) | Verify | Add reproducible AAC-LC decode benchmarks so performance regressions are measurable rather than inferred from upstream versions. |
| [#148](https://github.com/mstorsjo/fdk-aac/issues/148) | Fix | Make stream delay reporting and decoder draining consistent for one-frame, partial-stream, and ordinary end-of-stream cases. |
| [#129](https://github.com/mstorsjo/fdk-aac/issues/129) | Fix | Ensure short HE-AACv2 input cannot reach an unsafe C deinterleave path; preserve incremental-input semantics with owned buffering rather than exposing C memory unsafety. |
| [#120](https://github.com/mstorsjo/fdk-aac/issues/120) | Verify | Decode real dual-mono and multichannel USAC/exhale fixtures; existing synthetic dispatch tests are not sufficient evidence. |
| [#114](https://github.com/mstorsjo/fdk-aac/issues/114) | Verify + document | Test seek/discontinuity recovery and document `signal_interruption()` plus transport resynchronization requirements. |
| [#89](https://github.com/mstorsjo/fdk-aac/issues/89) | Verify | Import the historical SBR crashing inputs into the regression/fuzz corpus. The C fix does not prove safety of an independent Rust implementation. |
| [#85](https://github.com/mstorsjo/fdk-aac/issues/85) | Verify | Exercise LATM MCP1 packet and LOAS stream boundaries with real fixtures and incremental chunks. |
| [#78](https://github.com/mstorsjo/fdk-aac/issues/78) | Fix + document | Return an explicit unsupported-feature error for ER-AAC-LD LTP until prediction is implemented; do not misreport it as generic corruption. |
| [#43](https://github.com/mstorsjo/fdk-aac/issues/43) | Fix + verify | Exercise psychoacoustic and SBR fixed-point extrema and prevent wraparound, panic, or non-finite state. |
| [#25](https://github.com/mstorsjo/fdk-aac/issues/25) | Fix + verify | Support and test HE-AACv2 output changing from provisional mono/core rate to stereo/output rate without stale stream information. |
| [#16](https://github.com/mstorsjo/fdk-aac/issues/16) | Document | State that raw AAC access units require external packet boundaries and one complete access unit per decode call. |

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
