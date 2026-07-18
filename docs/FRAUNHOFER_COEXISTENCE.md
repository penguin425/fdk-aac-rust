# Coexistence with other Fraunhofer static libraries

`fdk-aac-rust-sys` and Fraunhofer IIS `mpeghdec` both embed descendants of
Fraunhofer's common FDK utility code. Their unmodified static archives therefore
cannot be linked into one executable: at the pinned `mpeghdec` r4.0.0 commit
`8149df84a777ea7d0a9a326f3c36067aec39201e`, 339 globally defined symbols
overlap and GNU ld reports multiple definitions.

The reviewed collision set is stored in
[`api/mpeghdec-r4.0.0-collisions.txt`](../api/mpeghdec-r4.0.0-collisions.txt).
CI rebuilds both source trees and requires an explicit review whenever this set
changes. No C/C++ source from either project is committed here.

## Supported integration procedure

On Linux, [`tools/test-fraunhofer-coexistence.sh`](../tools/test-fraunhofer-coexistence.sh)
uses GNU `objcopy` to rename the overlapping symbols in the `mpeghdec` archive.
It then verifies that no global collision remains, links in both archive orders,
and executes a probe that creates and destroys both an AAC encoder and an
MPEG-H decoder. The resulting namespaced archive is written below
`target/fraunhofer-coexistence/`.

Run the complete check with:

```sh
./tools/test-fraunhofer-coexistence.sh
```

This is an integration-time transformation, not a claim that arbitrary raw FDK
archives are directly compatible. Consumers that only use the Pure Rust codec
do not link `fdk-aac-rust-sys` and are unaffected. The current transformation
and execution test is Linux/GNU-specific; other linkers need an equivalent
symbol-renaming or symbol-hiding step before static coexistence can be claimed.
