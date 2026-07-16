# Security policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch. Older
releases are not maintained unless explicitly stated in their release notes.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
**Report a vulnerability** form in the repository Security tab. Include the
affected revision, attack preconditions, a minimal reproducer when possible,
and whether the issue involves the Pure Rust implementation or the optional
C/C++ FFI reference layer.

Avoid including real credentials, private media, or other sensitive data in a
report. Acknowledgement does not imply that patent or licensing questions can
be handled as security vulnerabilities; those remain governed by [`NOTICE`](NOTICE).

## Automated checks

The `Security` workflow runs on pull requests, pushes to `main`, a weekly
schedule, and manual dispatch. It checks RustSec advisories, dependency sources
and version policy, truncating casts at the C FFI boundary, and Git history for
committed secrets. Action references and downloaded scanner binaries are pinned
to immutable digests.
