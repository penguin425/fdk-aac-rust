# Performance tracking

The release-mode `benchmark_decode` example measures deterministic mono AAC-LC
decode throughput, first-frame CPU latency, Linux peak resident memory, and the
benchmark executable size. Five independent decoder instances are measured and
the median is reported as JSON.

[`tools/check-performance.sh`](../tools/check-performance.sh) runs the benchmark
and compares it with the deliberately broad catastrophe guards in
[`benchmarks/aac-lc-decode-guard.json`](../benchmarks/aac-lc-decode-guard.json).
These limits catch multi-fold slowdowns, runaway memory, unexpectedly huge
binaries, and pathological first-frame latency. They are not optimization
targets and must not be tightened to ordinary hosted-runner noise levels.

GitHub Actions runs the measurement for relevant pull requests, weekly, and on
manual dispatch. Every run writes the JSON measurement and environment metadata
to the workflow summary and retains them as an artifact for 90 days. Trends
should be evaluated across multiple runs on the same runner class; a single
small movement is not evidence of a regression.

Run it locally with:

```sh
./tools/check-performance.sh
```

For a quick smoke run:

```sh
PERF_FRAMES=200 PERF_ITERATIONS=3 ./tools/check-performance.sh
```

Codec algorithmic delay is intentionally excluded: encoder/decoder delay is a
stream property documented and tested separately, not a CPU performance metric.
