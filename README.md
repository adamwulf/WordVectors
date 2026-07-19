# WordVectors

An iOS/macOS app for training and exploring word2vec embeddings, backed by
`WordVectorKit` — a pure-Swift, in-memory port of Google's
[word2vec.c](docs/reference-word2vec.c) (CBOW + skip-gram, negative sampling).

## Layout

- **`WordVectorKit/`** — the Swift package. `Word2Vec.swift` is the training core;
  `WordEmbeddings.swift` holds the trained model and nearest-neighbor / projection
  queries. The `w2v-bench` executable target trains on the full corpus and prints
  wall-clock + a checksum, used for performance measurement.
- **`WordVectors/`** — the app: corpus loading, `ModelStore` (training orchestration),
  and the Explore/Train UI.
- **`WordVectors/Corpus/`** — ten public-domain books (~2.3M words) used as the
  training corpus.
- **`docs/`** — the reference `word2vec.c` and background research.

## Training performance

Training faithfully follows the reference math (exp table, unigram^0.75
negative-sampling table, subsampling, LCG RNG, alpha decay). Two rounds of
optimization sped up a full-corpus run (10 books, default parameters):

| Stage | Full-corpus train time | Speedup vs. original |
|---|---:|---:|
| Original (single-threaded, scalar loops) | ~73 s | 1.0× |
| Accelerate (vDSP inner loops + `-Ounchecked`) | ~34 s | ~2.1× |
| Hogwild (multithreaded, lock-free weights) | ~7 s | ~10× |
| Deterministic (synchronous SGD, bit-exact) | ~7 s | ~10× |

The **Accelerate** pass replaced the hot scalar dot-product / multiply-add loops
with vDSP primitives and enabled `-Ounchecked` for release builds. Results stay
essentially bit-identical (vDSP reorders float adds, so vectors drift by a tiny
rounding amount).

The **Hogwild** pass parallelizes training across CPU cores exactly as the
reference `word2vec.c` does: the corpus is split into contiguous per-thread chunks,
each thread owns its RNG (seeded from the thread id) and scratch buffers, and all
threads update the shared `syn0`/`syn1neg` weight matrices **without locks**
(asynchronous SGD). This trades run-to-run bit-exactness for speed: because the OS
interleaves the lock-free weight writes differently each run, vectors move slightly
between runs (measured mean per-word cosine similarity ≈ 0.89 across runs on the full
corpus). It is now opt-in via `Word2VecParameters.deterministic = false`.

The **Deterministic** pass (the default, `deterministic = true`) makes training
**bit-exact reproducible run-to-run** — train the same corpus with the same
parameters twice and every vector is byte-for-byte identical — while staying about as
fast as Hogwild. See below for how.

## Deterministic parallel training (reproducible + fast)

Training is bit-exact by default: `Word2Vec.train` uses **synchronous SGD** so a full
run on the ~2.3 M-word corpus produces byte-identical vectors every time (verified
with `w2v-bench <corpus> --check-determinism`, which trains twice and compares).

**How it works.** Each epoch, every worker trains on its own **private copy** of
`syn0`/`syn1neg`, seeded from a shared epoch-start snapshot. A worker therefore sees
only its *own* updates during an epoch and can never race another worker's write. At
the epoch barrier the per-worker deltas (`private − snapshot`) are summed back into the
shared weights **in fixed thread order (0, 1, 2, …)**. Float addition isn't
associative, so pinning the reduction order is exactly what makes the merged result
identical on every run regardless of how the OS scheduled the workers. Each worker's
LCG RNG state is carried across epochs (seeded once from its thread id), matching the
reference's `local_iter` loop. Alpha decay is computed as a pure function of
`epoch × train_words + local_word_count`, so the learning rate no longer depends on
the order in which workers reach the shared progress counter either.

**Cost.** One private `syn0`/`syn1neg` copy per worker (extra memory) plus a per-epoch
sync barrier. In exchange the weight race is gone entirely. Because workers read an
epoch-start snapshot rather than each other's mid-epoch updates, the learned vectors
differ slightly from the old lock-free Hogwild output — but they are now identical on
every run, which is the point.

**Why not just pre-compute the random draws?** An earlier design note asked whether
handing each thread a fixed slice of pre-drawn randoms would restore determinism. It
would not, and the reason is worth keeping: the RNG was *already* deterministic (each
thread seeds `nextRandom = threadId` and advances a pure LCG, drawing the same sequence
every run). The nondeterminism came entirely from the **lock-free weight writes** — the
OS interleaved the shared `syn1neg[...] += …` updates differently each run (occasionally
losing one to a torn read-modify-write). Synchronous SGD fixes the actual cause by
removing the shared write during the epoch; the randoms were never the problem.

For very large vocabularies where the per-worker weight copies become too costly, an
alternative with the same guarantee is to **partition the vocabulary** so no two threads
ever write the same word's row — then no private copies or delta merge are needed.
