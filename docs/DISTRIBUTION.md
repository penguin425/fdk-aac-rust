# Distribution compliance

This project is a **Third-Party Modified Version of the Fraunhofer FDK AAC
Codec Library for Android**. Every distributor is responsible for reading and
complying with the complete license in [`NOTICE`](../NOTICE). This checklist is
operational guidance and is not a replacement for that license.

## Source distributions

Every source archive must include, without alteration:

- `NOTICE`, containing the complete software license;
- `README.md`, containing the dated, prominent modification notice, modified
  version name, summary of changes, and patent warning.

The release checks enforce these requirements for both Rust `.crate` archives,
which are published to crates.io and attached to the matching GitHub Release.
Changes must also receive a dated `CHANGELOG.md` entry and be committed before
the annotated release tag is created.

## Binary distributions

The project release workflow intentionally publishes source `.crate` archives
only. Do not add compiled libraries, executables, application packages,
containers, or other binary artifacts to a release unless all of the following
are provided to every recipient:

1. the complete `NOTICE` text in the accompanying documentation or materials;
2. a free-of-charge copy of the complete corresponding source code for the FDK
   AAC codec and all distributed modifications, using an offer and delivery
   method that recipients can actually access;
3. the prominent modified-version name and dated change notices;
4. no use of the Fraunhofer name to endorse or promote the modified version;
5. no copyright license fee charged for use, copying, or distribution of the
   codec or its modifications.

Record the exact source revision corresponding to each binary and retain the
source-delivery location for as long as the binary remains available. A link to
a moving branch is not an adequate record of corresponding source.

## Patent rights

The software license grants no express or implied patent license. Distribution
or use of an AAC encoder or decoder may require authorization from applicable
patent owners or a licensing administrator. Passing the automated copyright
license checks does not establish patent clearance for any product, territory,
or use case.
