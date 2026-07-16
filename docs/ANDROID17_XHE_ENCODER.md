# Android 17 xHE-AAC encoder evaluation

Upstream issue [mstorsjo/fdk-aac#180](https://github.com/mstorsjo/fdk-aac/issues/180)
points to the xHE-AAC encoder first published in AOSP. It is relevant future
source material, but it is not an ordinary update of the C/C++ reference
currently pinned by this repository.

## Audited source

- AOSP project: `platform/external/aac`
- tag: `android-17.0.0_r1`
- tag object advertised by Gitiles: `67344ecb9d44a48c94a6097efd04ed801b4a24e7`
- checked-out commit: `41f344ffc0bacea87cac5bb1756bd40761265d1e`
- audited directory: `xhe-aac/`
- scope at that revision: 498 files, 7,958,291 Git blob bytes

The directory implements an encoder and its Android Codec2 integration. It is
not a decoder fix for multichannel exhale streams and cannot resolve decoder
issue #120 by changing `upstream/revision`.

## License boundary

`xhe-aac/NOTICE` contains the separate **Software Copyright License for the
Fraunhofer FDK Extended High Efficiency AAC Encoder Software for Android**,
copyright through 2025. Its required original and modified-version names refer
specifically to the Extended High Efficiency AAC Encoder Software, rather than
the older codec-library name currently used by this project.

Importing or translating this source would therefore require, at minimum:

1. retaining the complete additional encoder license;
2. adding its distinct prominent modified-version name and dated changes;
3. extending source and binary distribution checks to cover both license
   families;
4. keeping the existing no-patent-license warning applicable to the new
   encoder;
5. recording the Android tag and commit independently from the mstorsjo decoder
   reference revision.

The source must not be copied into the existing optional FFI checkout as if it
were part of `mstorsjo/fdk-aac`. Doing so would make revision provenance and the
required notices inaccurate.

## Technical integration decision

The encoder is large enough to be a separate porting project. A future import
should use its own crate or clearly isolated module, source revision file,
license verification, conformance fixtures, and feature flag. Before exposing
a public encoder API it also needs coverage for:

- mono and stereo bit-rate/rate combinations;
- AudioSpecificConfig and AudioPreRoll generation;
- loudness measurement and MPEG-D DRC metadata;
- random-access-point configuration;
- incremental input, flushing, delay and gapless metadata;
- interoperability with an independent xHE-AAC decoder.

For the current repository the disposition is therefore **track as a separate
licensed source and feature**, not **update the pinned C/C++ decoder baseline**.
No source from `xhe-aac/` is included by this evaluation.

