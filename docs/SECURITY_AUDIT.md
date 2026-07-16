# Security audit report

Audit date: 2026-07-17

Scope: Rust workspace source, the optional C/C++ FFI boundary, Cargo
dependencies, build scripts, GitHub Actions, release automation, and committed
Git history. This is a source-level review, not a formal penetration test or a
claim that arbitrary malformed AAC input is proven safe.

## Findings and remediation

### High — unchecked Rust slice lengths at the C FFI boundary

The safe encoder and decoder wrappers converted `usize` slice lengths to C
`INT`/`UINT` values with `as`. On platforms where `usize` is wider, very large
buffers could be truncated or become negative before reaching C/C++. Returned
C counts were also converted without checking sign or capacity. That could
violate the invariants on which the safe wrapper relies.

Status: fixed. Every slice length now uses a checked conversion before an FFI
call. Negative or out-of-capacity results from C are rejected. Stream sample
count multiplication and decoder consumed-byte subtraction are checked. Unit
tests cover overflow, negative, and over-capacity cases, and the FFI module
denies future `clippy::cast_possible_truncation` findings.

### Medium — cached upstream checkout was trusted by file presence

Build scripts accepted an existing generated checkout when a small set of files
existed. An altered cache could therefore supply modified C/C++ or reference
tables without its Git state being restored to the pinned revision.

Status: fixed. Generated checkouts are reset and cleaned against the full pinned
commit, missing commit objects are fetched explicitly, and `HEAD` is verified
before any source is read or the test-only patch is applied. Checkout and `.git`
paths must be real directories; symlinks are removed without following their
targets before any destructive Git cleanup command runs.

`FDK_AAC_SOURCE_DIR` remains an explicit trusted-local-source override. It is
not reset because doing so would modify a caller-owned checkout. Release and CI
environments must leave it unset unless the supplied tree has been independently
verified.

### Medium — GitHub Actions used mutable version tags

Workflow steps referenced `actions/checkout` by a movable major-version tag.
Compromise or movement of that tag could change code executed with repository
credentials.

Status: fixed. All workflow uses are pinned to the immutable commit for
`actions/checkout` v7.0.0. Dependabot is configured to propose reviewed updates.
The checkout in the Security workflow does not persist credentials.

### Low — no continuous advisory, source-policy, or secret scanning

The lockfile was checked during builds but had no dedicated security gate.

Status: fixed. The Security workflow installs pinned, locked versions of
`cargo-audit` and `cargo-deny`; rejects RustSec warnings, duplicate or wildcard
dependencies, unknown registries and Git sources; and runs a checksum-verified
Gitleaks binary over Git history every week and for every pull request.

### Informational — current scan results

- RustSec: no vulnerabilities or warnings in the five locked dependency crates.
- Dependency graph: no duplicate versions, wildcard requirements, unknown
  registries, or Git dependencies.
- Secret scan: no findings across 697 reachable commits (approximately 28 MB).
- Unsafe Rust: production unsafe operations are concentrated in the optional
  FFI wrapper; other matches are differential-test calls into the reference
  library. The Pure Rust codec does not expose an unsafe public API.
- A full `clippy -D warnings` run reports existing style and complexity debt.
  Those diagnostics are not security findings; the security-relevant FFI
  truncation lint is enforced independently so unrelated cleanup cannot mask it.

## Residual risk and recommended follow-up

AAC is a complex, attacker-controlled binary format. Static review cannot prove
that all decoder paths are resistant to panics, excessive allocation, or
algorithmic denial of service. Coverage-guided fuzzing of ADTS, LATM/LOAS, ASC,
raw AAC, SBR, and USAC entry points should be maintained as a separate dynamic
security effort. The optional upstream C/C++ reference layer inherits the
memory-safety risk of that implementation and should not be enabled merely to
process untrusted input when the Pure Rust path is sufficient.
