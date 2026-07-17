# Global FDK AAC improvement-request survey

This document consolidates public requests made by FDK AAC users and
integrators outside this repository. It is a demand survey, not a statement
that every report is an FDK defect or that every requested feature belongs in
this Rust port.

Snapshot date: **2026-07-18**.

## Method

The survey reviewed:

- all 54 open issues and four open pull requests in
  [`mstorsjo/fdk-aac`](https://github.com/mstorsjo/fdk-aac/issues), whose
  port-specific disposition is maintained in
  [`UPSTREAM_ISSUES.md`](UPSTREAM_ISSUES.md);
- public GitHub issues mentioning `fdk-aac`, `libfdk_aac`, and xHE-AAC in
  players, media servers, broadcast tools, build systems, and codec projects;
- primary distribution bug trackers where licensing and package availability
  affect real integrations.

GitHub search returned thousands of mentions, most of which were build logs,
dependency inventories, or unrelated application failures. The rows below
retain requests with a concrete codec/API implication, an independent demand
signal, or a recurring integration problem. Counts are therefore deliberately
not presented as popularity votes.

## Consolidated requests

| Priority | Improvement request | Public demand signals | Rust-port disposition |
|---|---|---|---|
| P0 | Broaden real-world xHE-AAC/USAC conformance, including multichannel, TCX, and malformed streams | Upstream [#120](https://github.com/mstorsjo/fdk-aac/issues/120), [mpv-android #1274](https://github.com/mpv-android/mpv-android/issues/1274), and [Audiobookshelf #4236](https://github.com/advplyr/audiobookshelf/issues/4236) report real files that common FFmpeg-based applications cannot decode correctly. | **Verify broadly.** The Rust implementation includes FD, ACELP, TCX, multichannel, SBR, and MPS paths. The remaining work is licensed real-world corpus coverage and correcting any mismatch that corpus exposes, not implementing a known missing TCX path. |
| P0 | Provide a modern xHE-AAC encoder | Upstream [#180](https://github.com/mstorsjo/fdk-aac/issues/180) points to the Android 17 encoder source; player/server requests show growing xHE adoption. | **Track separately.** The Android 17 encoder is a distinct 498-file source and licence surface, not a routine baseline update. Decide provenance and licensing before porting it. |
| P0 | Make seeking, preroll, draining, and gapless output unambiguous and testable | Upstream [#114](https://github.com/mstorsjo/fdk-aac/issues/114), [#148](https://github.com/mstorsjo/fdk-aac/issues/148), [#171](https://github.com/mstorsjo/fdk-aac/issues/171), and Mixxx [#14624](https://github.com/mixxxdj/mixxx/issues/14624) expose repeated boundary and delay confusion. | **Continue verification.** Existing interruption/draining tests should be extended with container-derived edit lists, random seeks, and fdk-aac-free boundary fixtures. Container metadata remains outside the codec crate. |
| P0 | Preserve safety under partial input and hostile bitstreams | Upstream [#89](https://github.com/mstorsjo/fdk-aac/issues/89), [#122](https://github.com/mstorsjo/fdk-aac/issues/122), and [#129](https://github.com/mstorsjo/fdk-aac/issues/129) cover crashes, pointer misuse, and short HE-AACv2 input. | **Continuous requirement.** Keep fuzzing all transports and profiles, replay the upstream crash corpus, and expose only length-checked safe APIs. |
| P1 | Demonstrate performance and reduce latency, memory, and binary size | Upstream [#15](https://github.com/mstorsjo/fdk-aac/issues/15), [#49](https://github.com/mstorsjo/fdk-aac/issues/49), [#141](https://github.com/mstorsjo/fdk-aac/issues/141), [#151](https://github.com/mstorsjo/fdk-aac/issues/151), and [#170](https://github.com/mstorsjo/fdk-aac/issues/170) repeatedly ask about these costs. | **Benchmark before optimizing.** Add profile/transport matrices for throughput, allocations, first-frame latency, algorithmic delay, and stripped size on x86-64 and ARM64. Do not conflate codec delay with CPU time. |
| P1 | Improve multichannel correctness and channel-layout interoperability | Upstream [#177](https://github.com/mstorsjo/fdk-aac/issues/177) reported a lost channel and [#175](https://github.com/mstorsjo/fdk-aac/issues/175) exposed multichannel bitrate expectations. | **Keep as a release gate.** Distinct-signal 5.1 coverage exists; add 7.1 layouts, PCE-based layouts, downmix metadata, and container channel-order interop. |
| P1 | Make quality/bitrate behavior measurable and metadata accurate | Upstream [#174](https://github.com/mstorsjo/fdk-aac/issues/174), PeerTube [#5652](https://github.com/Chocobozzz/PeerTube/issues/5652), and Flacon [#117](https://github.com/flacon/flacon/issues/117) show demand for FDK quality and correct VBR reporting. | **API and benchmark work.** Publish deterministic bitrate/quality sweeps and clearly distinguish configuration mode, achieved bitrate, and container metadata. FFmpeg/container metadata bugs are not codec fixes. |
| P1 | Stabilize the public ABI and coexist cleanly with other Fraunhofer libraries | The fdk-aac 2.0.1 ABI change broke fdkaac builds ([Debian #955248](https://bugs.debian.org/955248)); linking FDK AAC with `mpeghdec` produces duplicate symbols ([mpeghdec #16](https://github.com/Fraunhofer-IIS/mpeghdec/issues/16)). | **Rust API first; FFI compatibility second.** Add ABI snapshots for the compatibility crate, document SemVer guarantees, and test combined static linking. Pure Rust internals should not export generic FDK symbols. |
| P1 | Support predictable cross-platform builds and packaging | Upstream requests include Visual Studio/NuGet [PR #168](https://github.com/mstorsjo/fdk-aac/pull/168), Meson [PR #139](https://github.com/mstorsjo/fdk-aac/pull/139), Android/NDK reports, and version resources [#134](https://github.com/mstorsjo/fdk-aac/issues/134). | **Use Rust-native distribution.** Test the crates on Linux, macOS, Windows MSVC, Android, iOS, musl, x86-64, and ARM64. Avoid adding C/C++ build systems to the Pure Rust path. |
| P1 | Make legal/provenance boundaries machine-readable and easy to audit | Debian's PipeWire request [#1021370](https://bugs.debian.org/1021370) documents continuing package friction; Fedora's [fdk-aac-free review](https://bugzilla.redhat.com/show_bug.cgi?id=1501522) records the split-source approach. | **Document, do not reinterpret.** Preserve upstream notices, distinguish source subsets and the Android 17 xHE licence, publish SBOM/provenance, and never imply patent rights. Distribution policy remains distributor-specific. |
| P2 | Improve API guidance for DRC, raw access units, transport selection, frame geometry, and input PCM conversion | Upstream [#16](https://github.com/mstorsjo/fdk-aac/issues/16), [#102](https://github.com/mstorsjo/fdk-aac/issues/102), [#126](https://github.com/mstorsjo/fdk-aac/issues/126), [#149](https://github.com/mstorsjo/fdk-aac/issues/149), and [#172](https://github.com/mstorsjo/fdk-aac/issues/172) are mainly usage questions caused by weak discoverability. | **Improve examples and errors.** Add executable examples and typed configuration errors; keep PCM file parsing and containers out of the core codec. |
| P2 | Publish a visible maintenance and security process | Upstream [#143](https://github.com/mstorsjo/fdk-aac/issues/143), [#153](https://github.com/mstorsjo/fdk-aac/issues/153), and [#165](https://github.com/mstorsjo/fdk-aac/issues/165) request changelog, releases, and security policy. | **Already established; maintain it.** Continue versioned changelogs, GitHub Releases, pinned upstream comparisons, private vulnerability reporting, and release automation. |

## Recommended implementation order

1. Broaden real-world USAC/xHE-AAC decoder conformance with a
   licensed/checksummed corpus spanning FD, TCX, mono, stereo, 5.1, and stream
   changes.
2. Expand seek/drain/gapless boundary tests using the Mixxx failure as an
   independent integration fixture.
3. Add the performance/allocation/size matrix and ARM64 CI so optimization
   decisions are evidence-based.
4. Add 7.1/PCE/downmix conformance and combined-static-link ABI checks.
5. Decide whether Android 17 xHE encoding is a separately licensed crate or
   explicitly outside project scope before any code is imported.
6. Add SBOM/provenance artifacts and the remaining platform CI targets.

## Maintenance rule

Refresh this survey before every minor release. A new external report should
only become a codec task after its fixture, configuration, expected behavior,
and licence permit reproduction. Keep application, container, packaging, and
codec responsibilities separate, but retain recurring external reports as
demand signals even when the fix belongs elsewhere.
