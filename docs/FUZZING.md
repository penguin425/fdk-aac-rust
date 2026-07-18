# Continuous fuzzing

Eight libFuzzer harnesses cover the untrusted-input and API boundaries most
relevant to this codec:

| Target | Coverage |
| --- | --- |
| `adts` | Complete, streamed, and multi-block ADTS parsing |
| `loas_latm` | LOAS framing and LATM AudioMuxElement parsing |
| `raw_access_unit` | Raw AAC syntax and configured raw decoding |
| `asc` | AudioSpecificConfig and ProgramConfig parsing |
| `usac_xhe_aac` | USAC/xHE-AAC ASC and DRM static configuration |
| `drc` | MPEG-4, DVB, and unified DRC metadata |
| `encoder_config` | Stateful encoder configuration API validation |
| `ffi_boundary` | Safe wrappers around the legacy FDK ABI |

Pull requests compile every harness. A daily scheduled GitHub Actions run and
manual dispatch execute each target for five minutes. Crash inputs are retained
as workflow artifacts for 30 days. The seven Pure Rust targets do not enable
the `ffi` feature; only `ffi_boundary` downloads and builds the pinned upstream
reference implementation.

To reproduce a Pure Rust target locally:

```sh
cargo install cargo-fuzz --version 0.13.2 --locked
cargo +nightly fuzz run adts
```

The FFI target additionally needs its feature:

```sh
cargo +nightly fuzz run ffi_boundary --features ffi
```

Harnesses cap individual inputs before allocating or decoding. libFuzzer's
generated corpus lives below `fuzz/corpus/` and crash artifacts below
`fuzz/artifacts/`; neither generated directory is committed.
