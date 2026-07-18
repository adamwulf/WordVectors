# Word2Vec / Word-Embedding Alternatives — App-Store-Shippable Research

**Purpose:** Choose a **permissively-licensed** way to, ON DEVICE (iOS 26, UIKit, Swift):

1. **TRAIN** word vectors from a custom text corpus (~13 MB of Project Gutenberg plain-text books). ← **the hard requirement**
2. Look up the **N nearest words** to an input word.
3. Do word **algebra** (`king − man + woman`) and find nearest words to the result.

**Why we're doing this survey:** The library we were going to use — [StarlangSoftware/WordToVec-Swift](https://github.com/StarlangSoftware/WordToVec-Swift) — is **GPL-3.0**, which is incompatible with closed-source App Store distribution. We need MIT / BSD / Apache-2.0 (or similar).

**Two facts decide everything, so they were verified against primary sources first:**
- the **exact license** (quoted from the actual LICENSE file / license header — never guessed), and
- whether the candidate **trains from a custom corpus** (requirement #1) vs. only doing inference on pretrained vectors.

> ⚠️ **Headline finding:** There is **no maintained, permissively-licensed, pure-Swift word2vec _training_ package** available via the Swift Package Index or CocoaPods. Apple's own frameworks do requirements #2 and #3 beautifully but **cannot train an embedding from a corpus** — they only *consume* vectors you produce elsewhere. The realistic App-Store paths are therefore **(1) wrap Google's Apache-2.0 `word2vec.c`** or **(2) write a small clean-room Swift word2vec.** Details and rationale below.

---

## Comparison table

| Candidate | License (verified & quoted) | Trains from custom corpus? | Nearest-word? | Vector arithmetic? | SPM / iOS-ready? | Maintenance | Overall fit |
|---|---|---|---|---|---|---|---|
| **StarlangSoftware/WordToVec-Swift** | **GPL-3.0** — *"GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007"* [LICENSE](https://github.com/StarlangSoftware/WordToVec-Swift/blob/master/LICENSE) | ✅ Yes (skip-gram, from a `Corpus`) | ⚠️ vectors only (you add lookup) | ⚠️ vectors only (you add cosine) | ✅ SPM, iOS-capable | ✅ Active | ❌ **Unusable — GPL-3.0 kills App Store distribution.** Reference only. |
| **Apple `NLEmbedding`** (Natural Language) | Apple SDK (proprietary, ships with OS — no redistribution issue) | ❌ **No — load/lookup only** | ✅ `neighbors(for:…)` | ✅ `neighbors(for: vector…)` + `vector(for:)` | ✅ iOS 13+ built-in | ✅ Apple | ⚠️ **Perfect for #2 & #3, useless for #1.** Use as the *query layer*, not the trainer. |
| **Apple Create ML `MLWordEmbedding`** | Apple SDK (proprietary) | ❌ **No — compiles a dict you already have** | ✅ (in resulting model) | ✅ (via `NLEmbedding`) | ❌ **macOS 10.15+ only — not on iOS** | ✅ Apple | ❌ **Not a trainer and not on-device.** Dev-time compressor of *existing* vectors. |
| **Google `word2vec.c`** (Mikolov) | **Apache-2.0** — header: *"Copyright 2013 Google Inc… Licensed under the Apache License, Version 2.0"* [word2vec.c](https://github.com/tmikolov/word2vec/blob/master/word2vec.c), [LICENSE](https://github.com/tmikolov/word2vec/blob/master/LICENSE) | ✅ Yes (CBOW + skip-gram) | ⚠️ separate `distance` tool; add your own | ⚠️ separate `word-analogy` tool; add your own | ⚠️ C, must bridge; **no SPM** | ⚠️ Mirror, low churn (2023) | ✅ **Best "wrap it" option.** ~700 lines, plain C, pthreads + posix_memalign (both fine on iOS). |
| **alexsosn/Word2Vec-iOS** | **Apache-2.0** — GitHub license field `Apache-2.0` [repo](https://github.com/alexsosn/Word2Vec-iOS) (bundles Google word2vec.c, also Apache-2.0 — no license conflict) | ✅ Yes (`model.train()`, on-device) | ✅ `distance("cat", numberOfClosest:)` | ✅ `analogy("man woman king", …)` | ❌ **Demo `.xcodeproj` app, NOT a package** (no Package.swift/podspec) | ❌ **Stale — last push Nov 2015** | ⚠️ **Not shippable as-is, but a proven blueprint** — its ObjC-wrapped word2vec.c is exactly the "wrap it" path, pre-done. Lift & modernize the files. |
| **Facebook fastText** | **MIT** — *"MIT License / Copyright (c) 2016-present, Facebook, Inc."* [LICENSE](https://github.com/facebookresearch/fastText/blob/main/LICENSE) | ✅ Yes (skipgram/cbow, subword) | ✅ (`nn` tool) | ✅ (`analogies` tool) | ⚠️ C++11, must bridge; **no SPM**; no iOS story | ⚠️ Low churn | ⚠️ **Best license, heavier lift.** Larger C++ codebase + own build system; `std::thread`; file-input only. |
| **Stanford GloVe** | **Apache-2.0** — *"Apache License / Version 2.0"*, © 2014 Stanford [LICENSE](https://github.com/stanfordnlp/GloVe/blob/master/LICENSE) | ✅ Yes (co-occurrence based) | ⚠️ add your own | ⚠️ add your own | ⚠️ C multi-binary pipeline; **no SPM** | ✅ Active (2025) | ❌ **Poor fit for on-device** — 4-stage shell pipeline (`vocab_count`→`cooccur`→`shuffle`→`glove`) built around files & shell glue. |
| **Write our own (clean-room Swift word2vec)** | **N/A — we own it (MIT/whatever we choose)** | ✅ Yes (that's the point) | ✅ we implement | ✅ we implement | ✅ Pure Swift, native SPM, iOS-native | ✅ We maintain it | ✅ **Recommended.** ~5–8 small Swift files; no license risk; no C bridging; integrates cleanly with `NLEmbedding` for queries. |

---

## Top recommendation

### 🥇 Write our own clean-room, pure-Swift word2vec (skip-gram + negative sampling), and use Apple's `NLEmbedding` for the query layer.

**Rationale:**

1. **No permissive, maintained, pure-Swift training package exists.** The only two Swift-native word2vec projects that surface anywhere (GitHub, Swift Package Index) are:
   - StarlangSoftware/WordToVec-Swift — **GPL-3.0**, disqualified; and
   - alexsosn/Word2Vec-iOS — Apache-2.0 but a **10-year-stale demo app** (last push **Nov 2015**), not a consumable package.

   Everything else that trains is **C/C++** requiring a bridging header, or is **inference-only**.

2. **The training algorithm is genuinely small.** The GPL package we were going to use implements the whole thing in ~**5 core Swift files** (`NeuralNetwork`, `Vocabulary`, `VocabularyWord`, `WordToVecParameter`, `Iteration`) — skip-gram with negative sampling. A clean-room reimplementation (written from the published algorithm and Google's Apache-2.0 reference, **not** by copying the GPL code) is a few days of work, not weeks. This is the same footprint as the code we'd otherwise be bridging.

3. **Zero license risk and zero C-interop tax.** Pure Swift means: no bridging header, no static-lib build for arm64, no `Makefile`/CMake wrangling, no App Review questions about bundled third-party C. We license it however we like. This is the cleanest possible App-Store story.

4. **Apple gives us #2 and #3 for free — for the *query* side.** `NLEmbedding` (iOS 13+) is exactly a "map of strings to vectors, which locates neighboring, similar strings" and exposes `neighbors(for:maximumCount:distanceType:)`, `distance(between:and:distanceType:)`, `vector(for:)`, **and** a `neighbors(for: vector, …)` overload that takes a **raw vector** — which is precisely what `king − man + woman` needs. So after we train, we feed our `[String: [Double]]` into an `NLEmbedding` (compiled at runtime via `writeEmbedding(for:language:revision:to:)` / loaded via `init(contentsOf:)`) and get nearest-word + vector-algebra lookups from Apple's optimized on-disk index. **We only have to write the trainer; Apple already wrote the searcher.**

**What "write our own" entails (rough sketch):**

- **Corpus ingestion & tokenization** — stream the ~13 MB Gutenberg text, lowercase, split on non-letters, drop rare tokens below a min-count. (`NLTokenizer` can help, or a simple regex split.)
- **Vocabulary + subsampling** — build word→index, counts; frequent-word subsampling (the `t≈1e-5` trick).
- **Model** — two weight matrices (`W_in`, `W_out`), dimension ~100–300. `Float` arrays; use **Accelerate/`vDSP`** for the dot products to keep it fast on-device.
- **Training loop** — **skip-gram with negative sampling** (SGNS): for each center/context pair, 5–15 negative samples drawn from the unigram^0.75 distribution; sigmoid; SGD with a linearly-decayed learning rate. This is the core ~150–250 lines.
- **Concurrency** — run training off the main thread (`Task`/`DispatchQueue`); optionally shard across cores like the C version's pthreads. Report progress to the UI.
- **Output** — a `[String: [Double]]` dictionary → hand to `NLEmbedding` (runtime compile) for queries. Persist to the app's Documents dir so we only train once.
- **Nearest-word & algebra** — delegate to `NLEmbedding.neighbors(for:…)` and the vector-taking `neighbors(for:)` overload; or implement cosine over the dictionary ourselves (a few lines with `vDSP`).

**Effort:** ~3–5 focused days for a correct, on-device trainer + query wiring. Risk is low and fully in our control.

---

## Runner-up

### 🥈 Bridge Google's Apache-2.0 `word2vec.c` (using alexsosn/Word2Vec-iOS as the proven blueprint).

**Rationale:** If we want to lean on a battle-tested implementation rather than write SGD ourselves:

- **License is clean and verified:** the `word2vec.c` file header literally reads *"Copyright 2013 Google Inc… Licensed under the Apache License, Version 2.0"* — Apache-2.0 is App-Store-safe (permissive; just preserve the notice).
- **It already runs on iOS.** alexsosn/Word2Vec-iOS (Apache-2.0) wraps this exact C code as Objective-C `.m` files behind a Swift `Word2VecModel`, and its demo **trains on-device from a bundled Project Gutenberg text** (`pg2701.txt` — Moby-Dick, the same corpus family as ours) and exposes `train()`, `distance(_:numberOfClosest:)`, and `analogy(_:numberOfClosest:)`. It's a working template for all three of our requirements.
- **The sandbox concerns are manageable.** `word2vec.c` reads the corpus via `fopen()` on a `-train` path and writes vectors to an `-output` path — **file-IO only, no stdin/in-memory input.** That's fine on iOS: the app sandbox permits reading/writing its *own* container (Documents/`tmp`), so we write our corpus to a temp file and point the trainer at it. It uses `pthread_create` (multithreading) and `posix_memalign` — **both available on iOS/arm64**, no blockers.

**Why it's the runner-up, not the pick:**
- We'd be maintaining a bridging header + vendored C, and the only ready-made wrapper (alexsosn) is a **stale 2015 demo `.xcodeproj`, not a package** — we'd lift the files and modernize them for iOS 26 / current Swift, then own that fork.
- The C API is file-oriented and CLI-shaped; adapting it to a clean async Swift API with progress reporting is roughly as much glue as just writing the trainer in Swift — but with added C-interop and App-Review-of-bundled-C surface area.

**fastText (MIT) is the third option** and has the most permissive license of all the trainers, but it's a **larger C++11 codebase with its own build system** and no iOS story — the heaviest lift of the three C/C++ candidates. Reach for it only if we specifically need fastText's subword/OOV handling.

---

## What explicitly does NOT satisfy requirement #1 (training)

These are excellent for #2/#3 but **cannot learn vectors from our corpus** — do not mistake them for a solution:

- **`NLEmbedding`** *(quoted, Apple docs)*: *"Use an NLEmbedding to find similar strings based on the proximity of their vectors… **Natural Language provides built-in word embeddings** that you can retrieve… You can also **compile your own custom embedding** into an efficient, searchable, on-disk representation."* — It **loads/searches** vectors; it never derives them from text. iOS 13+. **Query layer only.**
- **Create ML `MLWordEmbedding`** *(quoted, Apple docs)*: *"You configure a word embedding **with a dictionary, keyed by strings**… The value for each string is an array of doubles, which represents a vector."* Initializer is `init(dictionary:parameters:)`. — It **compresses/compiles vectors you already have** into a `.mlmodel`; it does **not** train from a corpus. And it's **`macOS 10.15+` only — not available on iOS at all.** It is a *dev-time authoring/compression* tool, not an on-device trainer.

In other words, Apple's pipeline is: *(you train vectors elsewhere)* → `MLWordEmbedding`/`writeEmbedding` compiles them → `NLEmbedding` searches them. **The "train elsewhere" box is the one thing Apple doesn't fill — and it's our requirement #1.**

---

## License verification log (every claim, with source)

| Library | Claimed license | How verified | Quote |
|---|---|---|---|
| StarlangSoftware/WordToVec-Swift | GPL-3.0 | Fetched `LICENSE` raw | *"GNU GENERAL PUBLIC LICENSE / Version 3, 29 June 2007 / Copyright (C) 2007 Free Software Foundation, Inc."* — [LICENSE](https://github.com/StarlangSoftware/WordToVec-Swift/blob/master/LICENSE) |
| facebookresearch/fastText | MIT | Fetched `LICENSE` raw | *"MIT License / Copyright (c) 2016-present, Facebook, Inc."* — [LICENSE](https://github.com/facebookresearch/fastText/blob/main/LICENSE) |
| tmikolov/word2vec | Apache-2.0 | Fetched `LICENSE` **and** `word2vec.c` header | LICENSE: *"Apache License / Version 2.0, January 2004"*; header: *"Copyright 2013 Google Inc… Licensed under the Apache License, Version 2.0"* — [LICENSE](https://github.com/tmikolov/word2vec/blob/master/LICENSE), [word2vec.c](https://github.com/tmikolov/word2vec/blob/master/word2vec.c) |
| stanfordnlp/GloVe | Apache-2.0 | Fetched `LICENSE` raw | *"Apache License / Version 2.0, January 2004"*, © 2014 The Board of Trustees of Leland Stanford Junior University — [LICENSE](https://github.com/stanfordnlp/GloVe/blob/master/LICENSE) |
| alexsosn/Word2Vec-iOS | Apache-2.0 | GitHub API `license` field (`spdx_id: Apache-2.0`) + repo tree (single top-level LICENSE; no separate license on bundled C) | GitHub API `license.spdx_id = "Apache-2.0"` — [repo](https://github.com/alexsosn/Word2Vec-iOS). Bundled word2vec-derived C is itself Apache-2.0 → no conflict. |
| Apple `NLEmbedding` / `MLWordEmbedding` | Apple SDK (proprietary, no redistribution) | Apple developer docs | See "does NOT satisfy #1" section — quoted from [NLEmbedding](https://developer.apple.com/documentation/naturallanguage/nlembedding) and [MLWordEmbedding](https://developer.apple.com/documentation/createml/mlwordembedding) |

**Maintenance / metadata verified via GitHub API:**
- alexsosn/Word2Vec-iOS — `pushed_at` **2015-11-11**, 23 stars, `default_branch: master`, **no** Package.swift/podspec (`.xcodeproj` only). **Stale.**
- tmikolov/word2vec — `pushed_at` **2023-02-28**, 1,590 stars. *"Automatically exported from code.google.com/p/word2vec."*
- stanfordnlp/GloVe — `pushed_at` **2025-07-27**, 7,226 stars. Actively maintained.

**Could-not-verify items:** none material to the decision. All license and train-from-corpus claims above were confirmed against the actual LICENSE file / license header / Apple docs / GitHub API — not inferred.

---

## Bottom line

- **Requirements #2 (nearest word) and #3 (vector algebra) are solved by Apple's `NLEmbedding`** — free, on-device, iOS 13+, and its `neighbors(for: vector, …)` overload does `king − man + woman` directly.
- **Requirement #1 (train from our corpus) has no drop-in permissive Swift package.** Choose between:
  - **Write our own pure-Swift skip-gram/SGNS trainer** *(recommended)* — small, MIT-able, no C interop, cleanest App-Store story; or
  - **Bridge Google's Apache-2.0 `word2vec.c`** *(runner-up)* — proven, using alexsosn/Word2Vec-iOS as a blueprint, at the cost of vendoring & modernizing C.
- **fastText (MIT)** is the fallback if we later need subword/OOV robustness, accepting a heavier C++ integration.
- **StarlangSoftware/WordToVec-Swift stays a read-only reference** — its GPL-3.0 makes any linking or code reuse incompatible with App Store distribution. Do **not** copy from it when writing our own; work from the public algorithm and the Apache-2.0 reference instead.
