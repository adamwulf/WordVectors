# Word2Vec Phase 1 results

## Full-corpus release benchmark

Both measurements used the ten-book corpus and timed only `Word2Vec.train()` with
`ProcessInfo.processInfo.systemUptime`.

| Metric | Baseline | Phase 1 | Change |
|---|---:|---:|---:|
| Wall clock | 73.093 s | 34.233 s | **2.135x faster** (53.17% less time) |
| `the` vector checksum | -1.441858 | -1.442068 | -0.000210 (0.0146% relative drift) |
| Vocabulary count | 16047 | 16047 | unchanged |

The checksum drift is tiny and expected from vDSP's reordered floating-point additions. The
unchanged vocabulary and 0.0146% checksum movement satisfy the "super close" result requirement.

## Changes measured

- Enabled `-Ounchecked` for release builds of the `WordVectorKit` target.
- Replaced skip-gram and CBOW dot products and multiply-add updates with `vDSP_dotpr` and
  `vDSP_vsma`.
- Replaced the input-weight additions with `vDSP_vadd`.
- Replaced CBOW context accumulation and averaging with `vDSP_vadd` and `vDSP_vsdiv`.
- Reused the sentence subsampling buffer across sentences and epochs.
- Hoisted the repeated `Double(trainWords)` and `sample * trainWords` calculations.

The expected hot work has shifted from Swift scalar array/pointer loops in `trainSkipGram` and
`trainCBOW` into optimized Accelerate vector primitives. A measured top-symbol comparison could not
be exported because this worker's Instruments installation lacks the Time Profiler template and its
CPU templates fail under the kernel MAC policy; `sample` was also rejected by the command allow-list.
See `scratch/phase1-instruments-baseline.md` for the exact profiling diagnostics.

## Verification

- Release build: passed with `swift build --disable-sandbox -c release --package-path WordVectorKit`.
- Full tests: **48 passed, 0 failed** with
  `swift test --disable-sandbox --package-path WordVectorKit`.
- `testDeterministicWithSameSeed` still passes with exact vector equality, so it was **not relaxed**.
- Full-corpus Phase 1 checksum and vocabulary count are reported above.
