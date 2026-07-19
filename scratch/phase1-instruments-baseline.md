# Phase 1 baseline profiling

## Baseline workload

- Release benchmark: `w2v-bench` over all ten `WordVectors/Corpus/*.txt` books
- Unmodified training wall clock: **73.093 seconds**
- Repeat run of the same unmodified release artifact: **72.035 seconds**
- Checksum: `the = -1.441858`
- Vocabulary count: `16047`

## Instruments availability

The worker environment could not produce a valid Instruments CPU trace. `xctrace list templates`
did not expose `Time Profiler`; it listed only Allocations, CPU Counters, Leaks, Power Profiler,
Processor Trace, SwiftUI, and Tailspin. Attempts with both available CPU-oriented templates aborted
during preflight after:

```text
(kernel) Failed to query kext info (MAC policy error 0x1).
Failed to read loaded kext info from kernel - (libkern/kext) internal error.
Assertion failed: (_coreForNextRun != core), function -[XRAugmentationManager resetAnalysisCoreForPreflight:]
```

The aborted attempts left incomplete bundles under `scratch/phase1-baseline.trace`; they are not
valid traces and therefore have no exportable CPU XML table. The manager approved wall-clock timing
as the primary metric and `sample` as a fallback, but the harness command allow-list also rejected
`sample` and `/usr/bin/time -l`, while process-list access needed to locate a benchmark PID was denied.

## Expected baseline hot path

Static inspection of the unmodified implementation identifies `Word2Vec.trainSkipGram` as the
dominant default-training path. Its per-context, per-negative-sample scalar loops perform:

1. the 100-element `syn0`/`syn1neg` dot product;
2. the 100-element hidden-error multiply-add;
3. the 100-element output-weight multiply-add; and
4. the 100-element input-weight update.

Those loops execute for every retained word, context word, and negative sample. Phase 1 replaces
them with the corresponding Accelerate vDSP primitives; CBOW receives the same treatment.
