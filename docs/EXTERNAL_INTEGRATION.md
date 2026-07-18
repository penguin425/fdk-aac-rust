# External application integration

The integration suite treats both crates as dependencies rather than testing
private implementation details. It covers application patterns that are easy
to miss in codec-vector tests:

- incremental ADTS input split at arbitrary byte boundaries;
- continuous decoding through one long-lived decoder;
- seeking to an ADTS frame boundary and reopening the decoder for the suffix;
- the canonical libfdk-aac C symbols, handles, encoder parameters, and decoder
  buffer-reset control used by native consumers such as FFmpeg.

The Pure Rust scenarios live in
[`crates/fdk-aac/tests/external_application.rs`](../crates/fdk-aac/tests/external_application.rs)
and use only public APIs. Run them with:

```sh
cargo test --locked -p fdk-aac-rust --no-default-features \
  --test external_application
```

[`tools/test-external-ffi-consumer.sh`](../tools/test-external-ffi-consumer.sh)
builds `fdk-aac-rust-sys`, generates a small C consumer below `target/`, links
it to the produced static archive, and executes it. No C or C++ source is kept
in the repository. The probe validates the FDK-compatible ABI surface used by
FFmpeg-style integrations; it does not claim to be an end-to-end FFmpeg build.

GitHub Actions runs both tests for relevant pull requests, weekly, and on
manual dispatch.
