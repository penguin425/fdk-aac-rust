# Upstream reference policy

## Goals

The upstream reference must satisfy two requirements:

1. ordinary builds are reproducible;
2. new fdk-aac revisions can be evaluated without silently changing the
   baseline.

For that reason, builds do not follow a moving branch. The default reference is
the full SHA stored in [`upstream/revision`](../upstream/revision).

## Source selection

The build scripts select a source in this order:

1. `FDK_AAC_SOURCE_DIR`, when set, supplies an existing local source tree;
2. otherwise `FDK_AAC_REVISION`, when set, selects a temporary test revision;
3. otherwise the committed SHA in `upstream/revision` is used.

The two revision-based modes fetch
`https://github.com/mstorsjo/fdk-aac.git` into Cargo's `OUT_DIR`. Revision
values must be complete 40-character commit SHAs. Branch names and abbreviated
SHAs are rejected to keep results unambiguous.

`FDK_AAC_SOURCE_DIR` takes precedence over revision selection. Unset it when
testing `FDK_AAC_REVISION` or running the update script.

The revision-based modes reset and clean their generated checkout, then verify
that `HEAD` is exactly the pinned commit before reading or compiling it.
`FDK_AAC_SOURCE_DIR` is different: it is an explicit trusted-local-source
override and is never reset because the directory belongs to the caller. Do not
set it in CI or release jobs unless that complete source tree has been verified
independently.

## Moving to a newer revision

Run:

```sh
./tools/update-upstream.sh
```

The script performs the following transaction:

1. resolves the current GitHub `HEAD`;
2. leaves the repository unchanged when it already matches the pin;
3. runs the full workspace differential tests against the candidate SHA;
4. runs the Pure Rust test suite without default features;
5. checks formatting;
6. writes the candidate to `upstream/revision` only if every preceding step
   succeeds.

Review the upstream commits between the old and new pins before committing the
change. Keep each successful pin movement in a separate commit so failures can
be bisected and the comparison baseline remains obvious.

## Testing a specific candidate

To test a commit without promoting it:

```sh
FDK_AAC_REVISION=<full-40-character-sha> cargo test --workspace
FDK_AAC_REVISION=<full-40-character-sha> cargo test -p fdk-aac-rust --no-default-features
```

This is also the recommended way to bisect multiple upstream commits: test them
oldest to newest and promote only the newest consecutively passing revision.

## Handling failures

Do not update `upstream/revision` when any gate fails.

- If `crates/fdk-aac-sys/build-support/test-bridge.patch` no longer applies,
  inspect the upstream API or
  implementation change and update only the affected capture hook.
- If the reference compiles but differential output changes, determine whether
  upstream fixed a bug, changed expected behavior, or exposed a Rust-port
  mismatch. Add a focused regression test before adapting behavior.
- If a source table moves or changes syntax, update the Rust table reader and
  retain validation of the expected table size and contents.
- If the candidate itself is broken, keep the current pin and record the
  upstream issue; do not weaken a test merely to advance the revision.

## C/C++ files

No C/C++ source or header extension is tracked in this repository. The
differential adapter is stored as
`crates/fdk-aac-sys/build-support/qmf-test-wrapper.bridge` and is
written to Cargo's `OUT_DIR` as a temporary `.cpp` file during an FFI build.
The fetched upstream checkout and generated adapter are build artifacts and are
removed by `cargo clean`.
