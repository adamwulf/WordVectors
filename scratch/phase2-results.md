# Word2Vec Phase 2 results

## Full-corpus release benchmark

The Phase 2 measurement used the same ten-book corpus as Phase 1 and timed only
`Word2Vec.train()` with `ProcessInfo.processInfo.systemUptime`. SwiftPM's package sandbox was
disabled because nested `sandbox-exec` is unavailable in the worker harness; release optimization
and benchmark behavior were otherwise unchanged.

| Metric | Original | Phase 1 | Phase 2 Hogwild |
|---|---:|---:|---:|
| Wall clock | 73.093 s | 34.233 s | **7.151 s** |
| Speedup vs original | 1.000x | 2.135x | **10.221x** |
| Speedup vs Phase 1 | — | 1.000x | **4.787x** |
| `the` vector checksum | -1.441858 | -1.442068 | 1.301308 |
| Vocabulary count | 16047 | 16047 | 16047 |

The machine reports **10 active processors**, so the 4.787x improvement over the Accelerate-based
Phase 1 is 47.9% of the active-core count. The clean measurement command was:

```sh
swift run --disable-sandbox -c release --package-path WordVectorKit w2v-bench /Users/adamwulf/Developer/swift/WordVectors/.ittybitty/agents/agent-12743f55/repo/WordVectors/Corpus
```

The checksum moved by 2.743376 and changed sign versus Phase 1. A signed sum of 100 components is
especially cancellation-sensitive, and Hogwild intentionally changes RNG streams and update order,
so this scalar is no longer a useful closeness measure. Dedicated two-run scans of the full model
found:

- vocabulary count unchanged at 16047 in every run;
- **zero NaN or infinite components**;
- mean vector magnitude 1.7174 and maximum magnitude 6.1449 (finite, non-exploding values);
- mean same-word cosine across all 16047 vectors of 0.8875 and 0.8918 in two diagnostic pairs.

The synthetic structured-corpus stability test uses the stricter, reproducible quality gate: mean
same-word cosine across the entire vocabulary must exceed 0.9. Repeated runs passed.

## Hogwild implementation

- Uses `ProcessInfo.processInfo.activeProcessorCount` workers with contiguous sentence chunks.
- Each worker loops over its own chunk for every local iteration and owns its RNG (seeded with the
  thread id), sentence buffer, `neu1`/`neu1e` scratch buffers, word counters, and alpha.
- Workers share `syn0` and `syn1neg` pointers with intentionally lock-free updates, matching
  `TrainModelThread`'s asynchronous SGD approach.
- A small `NSLock` protects only the shared `wordCountActual` progress/alpha counter and serializes
  progress callback delivery. The app already hops UI progress updates to `MainActor`.
- The CBOW, skip-gram, subsampling, and sentence-boundary math remains unchanged.

## Test change and verification

`testDeterministicWithSameSeed` was renamed to `testTrainingIsStableAcrossRuns`. It now verifies
identical vocabulary/order, identical vector dimensions and counts, nonzero vector magnitudes, and
mean per-word cosine similarity above 0.9 across two runs. This replaces impossible bit-exact vector
equality with an aggregate geometry check appropriate for nondeterministic Hogwild training.

Full suite:

```text
swift test --disable-sandbox --package-path WordVectorKit
Executed 48 tests, with 0 failures (0 unexpected).
```

