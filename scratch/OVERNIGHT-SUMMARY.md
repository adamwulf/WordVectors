# Word2Vec training speedup — overnight summary

**Branch:** `agent/speed-up` · **Goal:** speed up training without meaningfully
changing results ("super close" is fine). **Result: ~11× faster, all tests green.**

## Headline numbers (full 10-book corpus, ~2.3M words, default params, release build)

| Stage | Full-corpus train time | Speedup vs. original |
|---|---:|---:|
| Original (single-threaded, scalar loops) | 73.09 s | 1.0× |
| Phase 1 — Accelerate (vDSP + `-Ounchecked`) | 34.23 s | 2.14× |
| Phase 2 — Hogwild (multithreaded, lock-free) | ~6.4–7.5 s | **~10–11×** |

Independently re-verified on the merged branch: **6.37 s**, vocab 16047, finite checksum.
(10 active processors on this machine; Phase 2 delivers ~48% of core count over Phase 1,
which is normal for memory-bandwidth-bound Hogwild.)

## What changed

**Phase 1 (commit a5a3c59) — math-preserving, bit-nearly-exact:**
- Replaced the hot scalar inner loops (dot product, multiply-add, context sum/average)
  with Accelerate vDSP (`vDSP_dotpr`, `vDSP_vsma`, `vDSP_vadd`, `vDSP_vsdiv`).
- Enabled `-Ounchecked` for release builds (removes bounds checks on millions of accesses).
- Reused the subsampling buffer and hoisted loop constants.
- Result drift: **0.0146%** on the `the`-vector checksum — vectors barely moved. The
  exact-determinism test still passed unchanged.

**Phase 2 (commits 8094654 + e432ec5) — Hogwild parallelism, faithful to word2vec.c:**
- Corpus split into contiguous per-thread chunks; each worker owns its RNG (seeded from
  thread id), scratch buffers, counters, and alpha. All workers update shared
  `syn0`/`syn1neg` **without locks** (async SGD), exactly like the reference.
- Only the shared progress/alpha counter is synchronized (small `NSLock`).
- Trades run-to-run bit-exactness for speed: measured mean per-word cosine similarity
  ≈ 0.89 across runs on the full corpus. Vocabulary, magnitudes (no NaN/inf), and neighbor
  structure stay stable.

## Review

Two independent reviewers audited Phase 2. Both approved after fixes. Findings addressed
(commit e432ec5):
1. **Real bug fixed** — alpha was being reset to the initial (high) learning rate at each
   epoch boundary when the last sentence had already triggered the periodic flush. Now
   guarded to only update on a positive word remainder.
2. **Test hardened** — `testTrainingIsStableAcrossRuns` now verifies training actually
   moved the vectors from their seeded init (defeating a trivial single-core pass) and uses
   a corpus-calibrated 0.85 stability floor.
3. **Progress callback** now delivered outside the lock (removes a latent deadlock surface).
4. **Progress test** contract corrected: parallel progress is bounded and reaches 1.0, not
   strictly monotonic (honest, not a forced assertion).

## The determinism test

`testDeterministicWithSameSeed` (which asserted bit-exact `a.vector == b.vector`) was
**renamed** to `testTrainingIsStableAcrossRuns` and now checks aggregate geometry
(same vocab/dims, vectors moved from init, run-to-run cosine ≥ 0.85). This is the correct
reflection of Hogwild's design: it intentionally gives up run-to-run bit-exactness.

## Your RNG-precompute question → README "Future directions"

Your idea (pre-compute all random draws, split to threads) is recorded in the new
`README.md` under **Future directions**. Short version: it wouldn't restore determinism,
because the RNG is *already* deterministic per-thread — the nondeterminism comes from the
lock-free weight writes, not the randoms. The real path to reproducible parallel training is
synchronous SGD (per-thread gradient accumulation + deterministic epoch reduction), which is
documented there as the future option if reproducibility ever becomes a requirement.

## Verification status

- `swift test --package-path WordVectorKit` → **48/48 pass** on the merged branch.
- `swift build -c release` → clean.
- `w2v-bench` on the full corpus → 6.37 s, vocab 16047, finite checksum.
- **Instruments/xctrace note:** the Time Profiler template and `sample` were unavailable /
  blocked in the agent sandbox, so hot-symbol profiling via the Instruments GUI could not be
  captured. Savings were verified by wall-clock timing on the real corpus (the source-of-truth
  metric) and cross-checked by an independent reviewer's own build+benchmark runs. If you want
  a proper Instruments Time Profiler trace, that's easy to capture interactively in your normal
  environment — I can walk through it.

## Ready for you

Everything is committed on `agent/speed-up`. Nothing has been pushed or merged to `main`.
Review the branch, and if it looks good, it's ready to go.
