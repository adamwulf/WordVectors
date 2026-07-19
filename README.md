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

The **Accelerate** pass replaced the hot scalar dot-product / multiply-add loops
with vDSP primitives and enabled `-Ounchecked` for release builds. Results stay
essentially bit-identical (vDSP reorders float adds, so vectors drift by a tiny
rounding amount).

The **Hogwild** pass parallelizes training across CPU cores exactly as the
reference `word2vec.c` does: the corpus is split into contiguous per-thread chunks,
each thread owns its RNG (seeded from the thread id), scratch buffers, and alpha,
and all threads update the shared `syn0`/`syn1neg` weight matrices **without locks**
(asynchronous SGD). Only the shared progress counter is synchronized. This trades
run-to-run bit-exactness for speed: because the OS interleaves the lock-free weight
writes differently each run, vectors move slightly between runs (measured mean
per-word cosine similarity ≈ 0.89 across runs on the full corpus). Vocabulary,
vector magnitudes, and neighbor structure remain stable, which is what the app
relies on.

## Future directions

### Deterministic parallel training (reproducible + fast)

The current Hogwild training is fast but **not reproducible run-to-run**. A natural
question is whether pre-computing all the random draws and handing each thread a
fixed slice would restore determinism. It would not, and it's worth recording why:

- **The RNG is already deterministic.** Each thread seeds `nextRandom = threadId`
  and advances a pure LCG, so every thread already draws the *same* sequence on
  every run. Pre-computing those draws changes nothing.
- **The nondeterminism comes from the weight writes, not the randoms.** Threads
  share `syn0`/`syn1neg` lock-free, so the OS interleaves their
  `syn1neg[...] += g * ...` updates differently each run (and occasionally loses one
  to a torn read-modify-write). That reordering is what moves the vectors — the
  random numbers are identical run-to-run either way.
- **The draw count isn't cleanly knowable in advance anyway.** Negative sampling
  re-draws on collision (`if t == word { continue }`) and the window/subsample draws
  are data-dependent, so the number of RNG calls per thread can't be derived from
  word/sentence counts alone.

If reproducible parallel training is ever wanted (e.g. "train twice, get the same
model" as a product guarantee), the right approach is **synchronous SGD**: give each
thread private gradient accumulators (or partition the vocabulary so no two threads
write the same word), then reduce/average deterministically at each epoch boundary.
This removes the weight race entirely and is bit-reproducible, at the cost of extra
memory and a per-epoch sync barrier — slower than pure Hogwild, but still far faster
than single-threaded.
