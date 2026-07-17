# Rust API and FFI compatibility

The minimum supported Rust version (MSRV) is **1.80.0**. CI compiles and tests
the Pure Rust crate with that exact toolchain. Raising the MSRV is a documented
compatibility change and must not happen accidentally through new syntax or
standard-library APIs.

`cargo-semver-checks` compares both published crates with version 0.2.1 on
crates.io. A reported breaking public API change requires either a compatible
implementation or an intentional SemVer version decision.

The optional sys crate additionally treats
[`crates/fdk-aac-sys/src/lib.rs`](../crates/fdk-aac-sys/src/lib.rs) as its FFI
declaration snapshot. CI verifies its committed SHA-256 value. After reviewing
the C ABI layout, constants, and function declarations for an intentional
change, update the snapshot explicitly:

```sh
sha256sum crates/fdk-aac-sys/src/lib.rs > api/fdk-aac-rust-sys.sha256
./tools/check-sys-abi.sh
```

Do not refresh the snapshot merely to make CI pass. The accompanying pull
request must explain whether the change is ABI-compatible and how that was
verified.
