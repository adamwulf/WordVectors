# WordToVec-Swift — API Reference & Integration Guide

**Purpose:** Single source of truth for integrating [StarlangSoftware/WordToVec-Swift](https://github.com/StarlangSoftware/WordToVec-Swift) into our UIKit iOS app to (1) train word vectors from a text corpus, (2) find nearest words to a given word, and (3) do word algebra (`king - man + woman`) and find nearest words to the resulting vector.

**Research method:** Every type name, method signature, parameter label, and behavioral note below was read from the **actual Swift source at the exact versions the package resolves to** (not master, not guessed). Source quotes are labeled with `<repo>@<tag> — <file>`. Relevant lines are quoted inline.

> ⚠️ **Read the two "Critical" callouts first** ([§0.1 Build blocker](#01-critical-the-package-does-not-compile-against-its-own-pinned-dependencies) and [§2.2 Corpus loading](#22-critical-how-a-corpus-is-actually-loaded-bundlemodule-not-a-file-path)). They will cost you a day each if discovered late.

---

## 0. TL;DR for the impatient

- Add via SPM: `https://github.com/StarlangSoftware/WordToVec-Swift.git`, only tag is **`1.0.0`**.
- **It will not build as-published** (see §0.1). Plan to either **vendor/patch the sources** or pin a self-consistent older `Math` — details in §0.1 and §6.
- Train:
  ```swift
  let param = WordToVecParameter()          // sensible defaults; tune via setters
  let net   = NeuralNetwork(corpus: corpus, parameter: param)
  let dict  = net.train()                    // -> VectorizedDictionary  (CPU-heavy; off main thread!)
  ```
- **Do not** use `Corpus(fileName:)` to load your own text — it reads from the *Corpus package's* bundle, not your file (see §2.2). Build the corpus **in memory** instead.
- Nearest word: `dict.mostSimilarWord(name: "king")` returns **only one** word and uses **raw dot product** (not cosine). For top-N or cosine, iterate yourself using the `Vector` API (§3, §4).
- Word algebra: use `Vector.addVector(v:)` / `Vector.subtract(v:)` / `Vector.difference(v:)`, then scan all words with `Vector.cosineSimilarity(v:)` (§4).

---

## 0.1. CRITICAL: The package does not compile against its own pinned dependencies

WordToVec-Swift `1.0.0` transitively resolves to **Math-Swift `1.0.11`**, but its source was written against an **older Math** API. Specifically:

- `NeuralNetwork.swift` constructs its weight matrix with a **4-argument** initializer:

  > `WordToVec-Swift@1.0.0 — Sources/WordToVec/NeuralNetwork.swift`
  > ```swift
  > self.__wordVectors = Matrix(row: self.__vocabulary.size(), col: self.__parameter.getLayerSize(), min: -0.5, max: 0.5)
  > self.__wordVectorUpdate = Matrix(row: self.__vocabulary.size(), col: self.__parameter.getLayerSize())
  > ```
  > (no `random:` argument, and no `Random` instance is created anywhere in the file)

- But **Math `1.0.11`** (the version dragged in by the pinned graph) only offers a **5-argument** initializer that *requires* a `Random`:

  > `Math-Swift@1.0.11 — Sources/Math/Matrix.swift`
  > ```swift
  > public class Matrix : NSCopying{
  > ...
  > public init(row: Int, col: Int)
  > public init(row: Int, col: Int, min: Double, max: Double, random: Random)   // <-- only min/max variant
  > public init(size: Int)
  > public init(rowVector: Vector, colVector: Vector)
  > ```

  There is **no** 4-arg `Matrix(row:col:min:max:)` in Math `1.0.11`. → **`error: missing argument for parameter 'random'`** at compile time.

**Why:** The 4-arg initializer existed in **Math ≤ 1.0.8** (which imported only `Foundation`). Between 1.0.8 and 1.0.10, StarlangSoftware refactored the random matrix constructor to take an explicit `Random` (from Util-Swift). WordToVec `1.0.0` predates that refactor but its dependency chain pins the *newer* Math.

**Verified boundary (read from source):**

| Math tag | `init(row:col:min:max:)` (4-arg) | `init(row:col:min:max:random:)` (5-arg) | imports |
|---|---|---|---|
| 1.0.5 | ✅ yes | ❌ no | `Foundation` |
| 1.0.8 | ✅ yes | ❌ no | `Foundation` |
| 1.0.10 | ❌ no | ✅ yes | `Foundation`, `Util` |
| 1.0.11 | ❌ no | ✅ yes | `Foundation`, `Util` |

### How to actually build it — pick one

**Option A — Vendor the WordToVec sources and patch the one call (recommended).**
There are only **5 small files** in `Sources/WordToVec/`. Copy them into the app target (or a local SPM package), keep depending on Corpus/Dictionary/Math via SPM at their normal versions, and change the two matrix constructions to pass a `Random`:

```swift
// Patched NeuralNetwork.init(corpus:parameter:)
let random = Random(seed: parameter.getSeed())     // Util.Random; seed default is 1
self.__wordVectors       = Matrix(row: self.__vocabulary.size(), col: parameter.getLayerSize(),
                                  min: -0.5, max: 0.5, random: random)
self.__wordVectorUpdate  = Matrix(row: self.__vocabulary.size(), col: parameter.getLayerSize())
```

`Random` comes from Util-Swift and is `import`able wherever `Math` is:

> `Util-Swift@1.0.8 — Sources/Util/Random.swift`
> ```swift
> public class Random{
>     public init(seed: Int = 0)
>     public func nextDouble(min: Double = 0.0, max: Double = 1.0) -> Double
>     public func nextInt(maxRange: Int) -> Int
>     public func shuffle(array: inout [Any])
> }
> ```

This is the cleanest path: you get to run training off the main thread, you control the seed, and you are not fighting SPM's exact-version pins. **Note:** vendoring means you lose the empty `catch {}` corpus loader anyway (see §2.2), which you were going to avoid.

**Option B — Fork WordToVec-Swift and relax the pins, then override Math to ≤ 1.0.8.**
Because every StarlangSoftware manifest uses `.exact(...)`, you cannot simply add a top-level Math override — the exact pins conflict (Dictionary `1.0.11` pins Math `1.0.11`). You would have to fork Corpus/Dictionary too and repin them to a Math that has the 4-arg initializer. This is more forks than Option A and is **not recommended**.

**Bottom line:** budget for Option A. The rest of this document documents the *real, current* APIs so that whichever path you take, the call sequence and types are correct.

---

## 1. Package coordinates & dependency graph

### 1.1. WordToVec-Swift

| Field | Value |
|---|---|
| GitHub URL | `https://github.com/StarlangSoftware/WordToVec-Swift.git` |
| Only release tag | **`1.0.0`** (published 2022-04-30; the repo has no other tags) |
| Default branch | `master` |
| `swift-tools-version` | **5.2** |
| Platform requirements | **None declared** in `Package.swift` (no `platforms:` clause) — inherits SwiftPM defaults. Practically it is pure Swift + Foundation, so it runs on iOS. Set your app's own deployment target (iOS 13+ recommended; `Bundle.module` resource bundling used by Corpus requires tools ≥ 5.3 / iOS 13). |
| License | **GPL-3.0** ⚠️ (see §5.6 — this has distribution implications for an App Store app) |
| Product | `.library(name: "WordToVec", targets: ["WordToVec"])` |

> `WordToVec-Swift@1.0.0 — Package.swift`
> ```swift
> // swift-tools-version:5.2
> let package = Package(
>     name: "WordToVec",
>     products: [ .library(name: "WordToVec", targets: ["WordToVec"]) ],
>     dependencies: [
>         .package(name: "Corpus", url: "https://github.com/StarlangSoftware/Corpus-Swift.git", .exact("1.0.17")),
>     ],
>     targets: [
>         .target(name: "WordToVec", dependencies: ["Corpus"]),
>         .testTarget(name: "WordToVecTests", dependencies: ["WordToVec"]),
>     ]
> )
> ```

Source files in the package (`Sources/WordToVec/`): `NeuralNetwork.swift`, `WordToVecParameter.swift`, `Vocabulary.swift`, `VocabularyWord.swift`, `Iteration.swift`.

### 1.2. Full transitive dependency list (as actually resolved)

All StarlangSoftware manifests use **`.exact(...)`** pins, so the resolved graph is fully determined:

```
WordToVec-Swift  1.0.0
└── Corpus-Swift        .exact 1.0.17
    ├── Dictionary-Swift    .exact 1.0.16   ← declared by Corpus 1.0.17's manifest
    │   └── Math-Swift          .exact 1.0.11
    │       └── Util-Swift          .exact 1.0.8
    └── DataStructure-Swift .exact 1.0.4    (no dependencies)
```

> ⚠️ **Pin-version subtlety (verified):** Corpus **`1.0.17`**'s `Package.swift` pins **Dictionary `1.0.16`** *and* **DataStructure `1.0.4`**. (An earlier reading of a stale/other tag showed Dictionary `1.0.11`; the `1.0.17` tag itself declares `1.0.16` — confirmed below.) Dictionary `1.0.16` → Math (verify at resolve time; Dictionary `1.0.11` pins Math `1.0.11`). **Whatever Dictionary version resolves, Math ≥ 1.0.10 is what breaks the build — see §0.1.** Re-confirm the exact `.resolved` numbers after you add the package in Xcode; they are what SwiftPM writes.

**Add-to-Xcode checklist (SPM):** the only URL you enter is WordToVec's; SwiftPM fetches the rest automatically:

| Package | URL | Resolved version |
|---|---|---|
| WordToVec-Swift | `https://github.com/StarlangSoftware/WordToVec-Swift.git` | `1.0.0` (exact — only tag) |
| Corpus-Swift | `https://github.com/StarlangSoftware/Corpus-Swift.git` | `1.0.17` |
| Dictionary-Swift | `https://github.com/StarlangSoftware/Dictionary-Swift.git` | `1.0.16` (per Corpus 1.0.17) |
| DataStructure-Swift | `https://github.com/StarlangSoftware/DataStructure-Swift.git` | `1.0.4` |
| Math-Swift | `https://github.com/StarlangSoftware/Math-Swift.git` | `1.0.11` (via Dictionary) |
| Util-Swift | `https://github.com/StarlangSoftware/Util-Swift.git` | `1.0.8` (via Math) |

**Verifying manifest quotes:**

> `Corpus-Swift@1.0.17 — Package.swift`
> ```swift
> // swift-tools-version:5.5
> dependencies: [
>     .package(name: "Dictionary", url: "https://github.com/StarlangSoftware/Dictionary-Swift.git", .exact("1.0.16")),
>     .package(name: "DataStructure", url: "https://github.com/StarlangSoftware/DataStructure-Swift.git", .exact("1.0.4")),
> ],
> // target "Corpus" resources: [.process("corpus.txt"), .process("simplecorpus.txt")]
> ```

> `Dictionary-Swift@1.0.11 — Package.swift`
> ```swift
> // swift-tools-version:5.5
> dependencies: [
>     .package(name: "Math", url: "https://github.com/StarlangSoftware/Math-Swift.git", .exact("1.0.11")),
> ],
> // Dictionary target resources: turkish_dictionary.txt, turkish_misspellings.txt, lowercase.txt, mixedcase.txt
> ```

> `Math-Swift@1.0.11 — Package.swift`
> ```swift
> // swift-tools-version:5.2
> dependencies: [
>     .package(name: "Util", url: "https://github.com/StarlangSoftware/Util-Swift.git", .exact("1.0.8")),
> ],
> ```

> `DataStructure-Swift@1.0.4 — Package.swift`
> ```swift
> // swift-tools-version:5.2
> dependencies: [ ]   // none
> ```

**Modules you will `import` in app code:**
`import WordToVec` (NeuralNetwork, WordToVecParameter, Vocabulary),
`import Corpus` (Corpus, Sentence),
`import Dictionary` (Word, Dictionary, VectorizedDictionary, VectorizedWord),
`import Math` (Vector, Matrix).

---

## 2. Training / generating vectors from a corpus

### 2.1. The types involved

| Type | Module | Role |
|---|---|---|
| `Corpus` | Corpus | Holds the training sentences/words. Input to training. |
| `Sentence` | Corpus | A list of `Word`s (one line of the corpus). |
| `Word` | Dictionary | A single token; wraps a `String` name. `Hashable`/`Equatable`. |
| `WordToVecParameter` | WordToVec | All training hyperparameters. |
| `NeuralNetwork` | WordToVec | The trainer. `init(corpus:parameter:)`, `train()`. |
| `VectorizedDictionary` | Dictionary | **Output** — the trained model (word → vector). |
| `VectorizedWord` | Dictionary | A `Word` subclass carrying a `Vector`. |
| `Vector` | Math | The embedding itself + all vector math. |

### 2.2. CRITICAL: how a corpus is *actually* loaded (`Bundle.module`, NOT a file path)

`Corpus` exposes these initializers:

> `Corpus-Swift@1.0.17 — Sources/Corpus/Corpus.swift`
> ```swift
> import Foundation
> import DataStructure
> import Dictionary
> import Util
>
> open class Corpus{
>     private var paragraphs: [Paragraph] = []
>     public var sentences: [Sentence] = []
>     public var wordList: CounterHashMap<Word> = CounterHashMap<Word>()
>     public var fileName: String = ""
>
>     public init()
>     public init(fileName: String)
>     public init(fileName: String, sentenceSplitter: SentenceSplitter)
>     public init(fileName: String, languageChecker: LanguageChecker)
>     ...
>     public func addSentence(s: Sentence)
>     public func sentenceCount() -> Int
>     public func getSentence(index: Int) -> Sentence
>     public func numberOfWords() -> Int
>     public func getWordList() -> Set<Word>
>     public func contains(word: String) -> Bool
>     ...
> }
> ```

**The trap:** `init(fileName:)` does **not** open the path you pass. It looks the name up as a resource inside the **Corpus package's own module bundle**, with a **hardcoded `.txt` extension**, and **swallows all errors**:

> `Corpus-Swift@1.0.17 — Sources/Corpus/Corpus.swift — init(fileName:) body`
> ```swift
> self.fileName = fileName
> let url = Bundle.module.url(forResource: fileName, withExtension: "txt")   // <-- Bundle.module = Corpus package bundle
> do{
>     let fileContent = try String(contentsOf: url!, encoding: .utf8)
>     let lines = fileContent.split(whereSeparator: \.isNewline)
>     for line in lines{
>         self.addSentence(s: Sentence(sentence: String(line)))
>     }
> } catch {
>     // <-- EMPTY. Any failure (nil url, unreadable file) is silently ignored.
> }
> ```

Consequences for our iOS app:

1. `Corpus(fileName: "myCorpus")` will look for `myCorpus.txt` **inside the Corpus SPM module's bundle** — which only ships `corpus.txt` and `simplecorpus.txt` (StarlangSoftware's own Turkish/English test data). It **cannot** read a file from *our* app bundle or the Documents directory.
2. If the resource is missing, `url` is `nil`, `url!` would crash — **but** it's inside the `do`/`catch`, and the force-unwrap trap on `nil` is a runtime crash, *not* a thrown error, so behavior is: **if the name doesn't resolve you crash on `url!`; if it resolves but read fails you silently get an empty corpus.** Either way, do not rely on this initializer for app data.
3. `catch {}` is empty → **silent failure**: a read problem yields a `Corpus` with **zero sentences**, and training on it produces an empty/garbage model with **no error surfaced**.

**✅ Correct approach for the app — build the corpus in memory:**

```swift
import Corpus
import Dictionary   // Word lives here; usually transitively visible via Corpus

func makeCorpus(from text: String) -> Corpus {
    let corpus = Corpus()                       // empty, in-memory
    for line in text.split(whereSeparator: \.isNewline) {
        // Sentence(sentence:) tokenizes on the SPACE character (see below)
        corpus.addSentence(s: Sentence(sentence: String(line)))
    }
    return corpus
}
```

If your corpus is a bundled resource in *your* app, load its text yourself and feed the string:

```swift
let url  = Bundle.main.url(forResource: "corpus", withExtension: "txt")!
let text = try String(contentsOf: url, encoding: .utf8)
let corpus = makeCorpus(from: text)
```

### 2.3. Corpus file format & tokenization (verified from source)

- **One sentence per line.** Lines are split on any Unicode newline (`split(whereSeparator: \.isNewline)`).
- **Tokenization is a plain split on the ASCII space character `" "`** — nothing more:

  > `Corpus-Swift@1.0.17 — Sources/Corpus/Sentence.swift`
  > ```swift
  > open class Sentence : Equatable{
  >     public init()
  >     public init(url: URL)
  >     public init(sentence: String)
  >     public init(sentence: String, languageChecker: LanguageChecker)
  >     public func getWord(index: Int) -> Word
  >     public func getWords() -> [Word]
  >     public func addWord(word: Word)
  >     public func wordCount() -> Int
  > }
  > ```
  > `init(sentence:)` body:
  > ```swift
  > public init(sentence: String){
  >     let wordList : [String] = sentence.split(separator: " ").map(String.init)
  >     for word in wordList{
  >         if word.count != 0{
  >             words.append(Word(name: word))
  >         }
  >     }
  > }
  > ```

  > `Dictionary-Swift@1.0.11 — Sources/Dictionary/Word.swift`
  > ```swift
  > open class Word : Comparable, Equatable, Hashable{
  >     private var name: String
  >     public init(name: String)
  >     public func getName() -> String
  >     public func setName(name: String)
  >     ...
  > }
  > ```

**Format implications (these cause silent quality problems, not crashes):**

- **No lowercasing.** `King`, `king`, and `KING` become three different words. Lowercase your corpus yourself if you want case-insensitive vectors.
- **No punctuation handling.** `king,` and `king.` are distinct tokens from `king`. Strip/pad punctuation with spaces before feeding the corpus (e.g. replace `.,!?;:"'()` with a space, then collapse spaces).
- **Only the space `" "` splits tokens.** Tabs and other whitespace are *not* token separators — a tab-separated line becomes one giant "word." Normalize all whitespace to single spaces first.
- Empty tokens (from double spaces) are skipped (`if word.count != 0`).
- The classic word2vec needs a reasonably large corpus; tiny corpora give unstable/nonsense neighbors. This is a data issue, not an API one — but it looks like "the library is broken" if you test with three sentences.

### 2.4. `WordToVecParameter` — every hyperparameter

> `WordToVec-Swift@1.0.0 — Sources/WordToVec/WordToVecParameter.swift`
> ```swift
> public class WordToVecParameter{
>     private var __layerSize: Int = 100
>     private var __cbow: Bool = true
>     private var __alpha: Double = 0.025
>     private var __window: Int = 5
>     private var __hierarchicalSoftMax: Bool = false
>     private var __negativeSamplingSize: Int = 5
>     private var __numberOfIterations: Int = 3
>     private var __seed: Int = 1
>
>     public init()
>
>     public func getLayerSize() -> Int
>     public func isCbow() -> Bool
>     public func getAlpha() -> Double
>     public func getWindow() -> Int
>     public func isHierarchicalSoftMax() -> Bool
>     public func getNegativeSamplingSize() -> Int
>     public func getNumberOfIterations() -> Int
>     public func getSeed() -> Int
>
>     public func setLayerSize(layerSize: Int)
>     public func setCbow(cbow: Bool)
>     public func setAlpha(alpha: Double)
>     public func setWindow(window: Int)
>     public func setHierarchialSoftMax(hierarchicalSoftMax: Bool)   // NOTE spelling: "Hierarchial"
>     public func setNegativeSamplingSize(negativeSamplingSize: Int)
>     public func setNumberOfIterations(numberOfIterations: Int)
>     public func setSeed(seed: Int)
> }
> ```

| Setter / getter | Default | Meaning | Typical values |
|---|---|---|---|
| `layerSize` | **100** | Embedding dimension (vector length; # neurons in hidden layer). | 50–300. Smaller = faster + less memory; use 50–100 on-device. |
| `cbow` | **true** | `true` = **CBOW** (predict word from context, faster). `false` = **skip-gram** (predict context from word, better for rare words). *(README calls the package "SkipGram" but the default is CBOW.)* | CBOW for speed on device; skip-gram for quality on small vocab. |
| `alpha` | **0.025** | Initial learning rate; decays over training internally. | 0.025 (CBOW) / 0.05 (skip-gram) are the classic word2vec values. |
| `window` | **5** | Max context distance (words on each side considered). | 5 typical; 10 for skip-gram; 2–3 for tiny corpora. |
| `hierarchicalSoftMax` | **false** | Use hierarchical softmax (builds a Huffman tree over vocab). If `false`, **negative sampling** is used. | Leave `false` → negative sampling (the common choice). |
| `negativeSamplingSize` | **5** | # negative samples per positive (only used when hierarchicalSoftMax == false). | 5–20 (small data), 2–5 (large data). |
| `numberOfIterations` | **3** | Training epochs over the whole corpus. | 3–15. More epochs = better but slower. |
| `seed` | **1** | RNG seed (used for the `Random` in matrix init / shuffling → **reproducible** vectors). | any Int; fix it for reproducible results. |

> ⚠️ Setter name typo to remember: **`setHierarchialSoftMax(hierarchicalSoftMax:)`** ("Hierarchial", missing the second *c*). The getter is spelled correctly (`isHierarchicalSoftMax()`).

### 2.5. `NeuralNetwork` — the trainer

> `WordToVec-Swift@1.0.0 — Sources/WordToVec/NeuralNetwork.swift`
> ```swift
> import Foundation
> import Math
> import Corpus
> import Dictionary
>
> // (properties __wordVectors: Matrix, __wordVectorUpdate: Matrix, __vocabulary: Vocabulary,
> //  __parameter, __corpus, __expTable; constants EXP_TABLE_SIZE = 1000, MAX_EXP = 6)
>
> public init(corpus: Corpus, parameter: WordToVecParameter){
>     self.__vocabulary = Vocabulary(corpus: corpus)
>     self.__parameter = parameter
>     self.__corpus = corpus
>     self.__wordVectors = Matrix(row: self.__vocabulary.size(), col: self.__parameter.getLayerSize(), min: -0.5, max: 0.5)  // ⚠ §0.1
>     self.__wordVectorUpdate = Matrix(row: self.__vocabulary.size(), col: self.__parameter.getLayerSize())
>     self.__prepareExpTable()
> }
>
> public func train() -> VectorizedDictionary{
>     let result : VectorizedDictionary = VectorizedDictionary()
>     if self.__parameter.isCbow(){
>         self.__trainCbow()
>     } else {
>         self.__trainSkipGram()
>     }
>     for i in 0..<self.__vocabulary.size(){
>         result.addWord(word: VectorizedWord(name: self.__vocabulary.getWord(index: i).getName(),
>                                              vector: self.__wordVectors.getRowVector(row: i)))
>     }
>     return result
> }
> ```

- **`init(corpus: Corpus, parameter: WordToVecParameter)`** — builds the `Vocabulary` from the corpus, random-inits the weight matrix (`min: -0.5, max: 0.5`), and precomputes the sigmoid lookup table. This is the call that fails to compile against Math 1.0.11 (§0.1).
- **`train() -> VectorizedDictionary`** — runs CBOW or skip-gram (per `isCbow()`), then copies each row of the trained weight matrix into a `VectorizedWord` keyed by the vocabulary word's name, and returns them all as a `VectorizedDictionary`. **This is the type that holds your vectors.**
- `Vocabulary(corpus:)` is constructed internally; you don't call it directly. It builds word counts, a Huffman tree (for hierarchical softmax) and a unigram table (for negative sampling).

### 2.6. Minimal working training snippet (app-ready)

```swift
import WordToVec
import Corpus
import Dictionary
import Math

/// Trains word vectors. MUST be called off the main thread (see §5.1).
func trainWordVectors(corpusText: String) -> VectorizedDictionary {
    // 1. Build the corpus IN MEMORY (do NOT use Corpus(fileName:) — see §2.2).
    let corpus = Corpus()
    for line in corpusText.split(whereSeparator: \.isNewline) {
        corpus.addSentence(s: Sentence(sentence: String(line)))
    }

    // 2. Configure hyperparameters.
    let param = WordToVecParameter()
    param.setLayerSize(layerSize: 100)          // embedding dimension
    param.setCbow(cbow: true)                   // CBOW (fast) vs skip-gram (param.setCbow(cbow: false))
    param.setWindow(window: 5)
    param.setAlpha(alpha: 0.025)                // learning rate
    param.setNegativeSamplingSize(negativeSamplingSize: 5)
    param.setNumberOfIterations(numberOfIterations: 5)
    param.setSeed(seed: 1)                       // reproducible
    // param.setHierarchialSoftMax(hierarchicalSoftMax: false)  // note the "Hierarchial" spelling

    // 3. Train.
    let net = NeuralNetwork(corpus: corpus, parameter: param)
    let model: VectorizedDictionary = net.train()   // CPU-heavy
    return model
}

// Usage from UIKit — never block the main thread:
DispatchQueue.global(qos: .userInitiated).async {
    let model = trainWordVectors(corpusText: text)
    DispatchQueue.main.async {
        // stash `model`, enable the UI, etc.
    }
}
```

---

## 3. Nearest-word lookup

### 3.1. Model / word / vector accessors

`VectorizedDictionary` extends the base `Dictionary`; words live in the inherited `words: [Word]` array.

> `Dictionary-Swift@1.0.11 — Sources/Dictionary/VectorizedDictionary.swift`
> ```swift
> public class VectorizedDictionary: Dictionary{
>     public init(){
>         super.init(comparator: { $0.getName() < $1.getName() })
>     }
>     public func addWord(word: VectorizedWord){
>         self.words.append(word)
>     }
>     public func mostSimilarWord(name: String) -> VectorizedWord?{
>         var maxDistance : Double = -1
>         var result : VectorizedWord? = nil
>         let word = self.getWord(name: name)
>         if word == nil{
>             return nil
>         }
>         for currentWord in self.words{
>             if currentWord != word && word is VectorizedWord{
>                 let distance = (word as! VectorizedWord).getVector().dotProduct(v: (currentWord as! VectorizedWord).getVector())
>                 if distance > maxDistance{
>                     maxDistance = distance
>                     result = currentWord as? VectorizedWord
>                 }
>             }
>         }
>         return result
>     }
> }
> ```

> `Dictionary-Swift@master — Sources/Dictionary/VectorizedWord.swift`
> ```swift
> import Foundation
> import Math
> public class VectorizedWord : Word{
>     private var __vector: Vector
>     public init(name: String, vector: Vector)
>     public func getVector() -> Vector
> }
> ```

Base-class methods available on a `VectorizedDictionary`:

> `Dictionary-Swift@master — Sources/Dictionary/Dictionary.swift`
> ```swift
> public class Dictionary{
>     var words: [Word] = []                                  // internal storage (not public)
>     public init(comparator : @escaping (Word, Word) throws -> Bool)
>     public func getWord(name: String) -> Word?
>     public func getWordIndex(name: String) -> Int
>     public func size() -> Int
>     public func getWordWithIndex(index: Int) -> Word        // <-- iterate all words by index
>     ...
> }
> ```

**Get the vector for a word:**
```swift
guard let vw = model.getWord(name: "king") as? VectorizedWord else { /* OOV */ return }
let v: Vector = vw.getVector()
```
`getWord(name:)` returns `Word?`; downcast to `VectorizedWord` to reach `getVector()`. `nil` / failed cast ⇒ the word is **out-of-vocabulary** (never appeared in the corpus). Always handle OOV — it is the #1 source of "why is it empty" bugs.

### 3.2. Built-in nearest word — and its two limitations

`mostSimilarWord(name:)` **exists** but:

1. It returns **only the single** most similar word (`VectorizedWord?`), not a top-N list.
2. It ranks by **raw `dotProduct`** — **not** cosine similarity and **not** length-normalized. Because word2vec vectors have very different magnitudes, dot product is biased toward high-frequency / large-norm words. For quality neighbors you almost always want **cosine** similarity.

Returns `nil` if the query word is OOV.

```swift
let nearest: VectorizedWord? = model.mostSimilarWord(name: "king")
print(nearest?.getName() ?? "out of vocabulary")   // one word only, dot-product ranked
```

### 3.3. Top-N nearest words (recommended — cosine, implement yourself)

Iterate the vocabulary via `size()` + `getWordWithIndex(index:)`, score with `Vector.cosineSimilarity(v:)`, sort, take N:

```swift
import Dictionary
import Math

/// Returns up to `count` nearest words to `word` by cosine similarity. Empty if `word` is OOV.
func nearestWords(to word: String, in model: VectorizedDictionary, count: Int = 10)
    -> [(word: String, score: Double)]
{
    guard let target = model.getWord(name: word) as? VectorizedWord else { return [] }
    let targetVec = target.getVector()
    let targetName = word

    var scored: [(String, Double)] = []
    scored.reserveCapacity(model.size())
    for i in 0..<model.size() {
        guard let vw = model.getWordWithIndex(index: i) as? VectorizedWord else { continue }
        let name = vw.getName()
        if name == targetName { continue }                       // skip the query itself
        let sim = targetVec.cosineSimilarity(v: vw.getVector())  // Math.Vector
        scored.append((name, sim))
    }
    scored.sort { $0.1 > $1.1 }
    return Array(scored.prefix(count)).map { (word: $0.0, score: $0.1) }
}
```

> **Perf note:** this is O(V·D) per query (V = vocab size, D = dimension). For a few-thousand-word vocab this is instant; for large vocabularies precompute once and/or cache the target vector. `cosineSimilarity` internally computes both norms every call, so if you do many queries against a fixed set, consider pre-normalizing (see §4.3).

---

## 4. Vector arithmetic (word algebra: `king - man + woman`)

### 4.1. The `Vector` API (Math module) — full method list

> `Math-Swift@1.0.11 — Sources/Math/Vector.swift`
> ```swift
> import Foundation
> public class Vector : Equatable{
>     private var __size : Int
>     private var values : [Double]
>
>     public init(values: [Double])
>     public init(size: Int, x: Double)
>     public init(size: Int, index: Int, x: Double)
>     public static func == (lhs: Vector, rhs: Vector) -> Bool
>
>     public func biased() -> Vector
>     public func add(x: Double)                         // appends a scalar element (grows the vector)
>     public func insert(pos: Int, x: Double)
>     public func remove(pos: Int)
>     public func clear()
>     public func sumOfElements() -> Double
>     public func maxIndex() -> Int
>     public func sigmoid()
>     public func tanh()
>     public func relu()
>     public func reluDerivative()
>     public func skipVector(mod: Int, value: Int) -> Vector
>     public func addVector(v: Vector)                   // in-place elementwise ADD  (self += v)
>     public func subtract(v: Vector)                    // in-place elementwise SUBTRACT (self -= v)
>     public func difference(v: Vector) -> Vector        // returns NEW vector (self - v)
>     public func dotProduct(v: Vector) -> Double
>     public func dotProductWithSelf() -> Double
>     public func elementProduct(v: Vector) -> Vector    // returns NEW elementwise product
>     public func multiply(v: Vector) -> Matrix          // outer product -> Matrix
>     public func divide(value: Double)                  // in-place scalar divide
>     public func multiply(value: Double)                // in-place scalar multiply
>     public func product(value: Double) -> Vector       // returns NEW scaled vector
>     public func l1Normalize()                          // in-place; divide by sum of elements
>     public func l2Norm() -> Double                     // Euclidean length
>     public func cosineSimilarity(v: Vector) -> Double
>     public func size() -> Int
>     public func getValue(index: Int) -> Double
>     public func setValue(index: Int, value: Double)
>     public func addValue(index: Int, value: Double)
> }
> ```

**Key method → operation map for word algebra:**

| Operation | Method | Mutates? |
|---|---|---|
| `a += b` | `a.addVector(v: b)` | in-place (mutates `a`) |
| `a -= b` | `a.subtract(v: b)` | in-place (mutates `a`) |
| `c = a - b` (new) | `let c = a.difference(v: b)` | returns new |
| `a · b` | `a.dotProduct(v: b) -> Double` | — |
| cosine(a,b) | `a.cosineSimilarity(v: b) -> Double` | — |
| ‖a‖ (L2) | `a.l2Norm() -> Double` | — |
| scale `a * k` (new) | `let s = a.product(value: k)` | returns new |
| scale `a *= k` | `a.multiply(value: k)` | in-place |
| L1 normalize | `a.l1Normalize()` | in-place (divides by **sum**, not L2 norm!) |
| element access | `a.getValue(index:)`, `a.setValue(index:value:)`, `a.addValue(index:value:)` | — |
| length | `a.size() -> Int` | — |

> ⚠️ **Two dangerous surprises:**
> 1. **`add(x: Double)` is NOT vector addition** — it *appends a scalar element* and grows the vector. For "add two vectors," use **`addVector(v:)`**. Confusing `add`/`addVector` will silently corrupt dimensions.
> 2. **`l1Normalize()` divides by the sum of elements (L1), not the L2 norm.** There is **no built-in unit-L2-normalize** method. If you want unit vectors, do it manually with `l2Norm()` (see §4.3). `addVector`, `subtract`, and `product` **mutate or derive** — copy first if you need to preserve the source word's vector.

### 4.2. Copy-safe vector helpers (recommended)

Because `addVector`/`subtract` mutate in place, and `Vector` is a **class (reference type)**, always start algebra from a *copy* so you don't corrupt the model's stored word vectors:

```swift
import Math

extension Vector {
    /// A deep copy (values are Doubles, so copying the array is enough).
    func copied() -> Vector {
        let out = Vector(size: self.size(), x: 0.0)
        for i in 0..<self.size() { out.setValue(index: i, value: self.getValue(index: i)) }
        return out
    }
}

func vector(for word: String, in model: VectorizedDictionary) -> Vector? {
    (model.getWord(name: word) as? VectorizedWord)?.getVector()
}
```

### 4.3. `king - man + woman` and nearest words to the *result vector*

There is **no built-in** "most similar to an arbitrary vector" method (`mostSimilarWord` only takes a *name*). Implement it over the vocabulary using the `Vector` API:

```swift
import Dictionary
import Math

/// Compute an analogy vector: base - minus + plus  (e.g. king - man + woman).
func analogyVector(base: String, minus: String, plus: String,
                   in model: VectorizedDictionary) -> Vector?
{
    guard let b = vector(for: base,  in: model)?.copied(),   // copy: addVector/subtract mutate
          let m = vector(for: minus, in: model),
          let p = vector(for: plus,  in: model)
    else { return nil }                                       // any OOV -> nil
    b.subtract(v: m)      // b = base - man
    b.addVector(v: p)     // b = base - man + woman
    return b
}

/// Nearest words to an ARBITRARY vector (not a named word), by cosine similarity.
/// `exclude` filters out the input words (analogy queries should exclude base/minus/plus).
func nearestWords(to query: Vector, in model: VectorizedDictionary,
                  count: Int = 10, exclude: Set<String> = []) -> [(word: String, score: Double)]
{
    var scored: [(String, Double)] = []
    scored.reserveCapacity(model.size())
    for i in 0..<model.size() {
        guard let vw = model.getWordWithIndex(index: i) as? VectorizedWord else { continue }
        let name = vw.getName()
        if exclude.contains(name) { continue }
        scored.append((name, query.cosineSimilarity(v: vw.getVector())))
    }
    scored.sort { $0.1 > $1.1 }
    return Array(scored.prefix(count)).map { (word: $0.0, score: $0.1) }
}

// ---- Usage: king - man + woman  ~=  queen ----
if let q = analogyVector(base: "king", minus: "man", plus: "woman", in: model) {
    let results = nearestWords(to: q, in: model, count: 5,
                               exclude: ["king", "man", "woman"])   // standard analogy: exclude inputs
    // results.first?.word  ->  ideally "queen" (quality depends on corpus size/training)
} else {
    // one of king/man/woman is out-of-vocabulary
}
```

**Notes:**
- **Exclude the input words.** Classic analogy evaluation removes `base`, `minus`, `plus` from the candidate set — otherwise the top hit is usually just `king` itself.
- **Cosine, not dot product.** We use `cosineSimilarity` for the same reason as §3.2.
- **Optional speed-up (pre-normalize once):** if you run many analogy/nearest queries, precompute unit-L2 copies of every word vector once and use `dotProduct` (equals cosine for unit vectors). There's no built-in L2 normalize, so:
  ```swift
  func l2Normalized(_ v: Vector) -> Vector {
      let n = v.l2Norm()
      let out = v.copied()
      if n > 0 { out.multiply(value: 1.0 / n) }   // in-place scalar multiply on the copy
      return out
  }
  ```

---

## 5. Gotchas & operational notes

### 5.1. Threading — training MUST be off the main thread
`train()` runs the full word2vec loop (all epochs × all sentences × window × negative samples) synchronously and is **CPU-bound**. On device this can take seconds to minutes depending on corpus/vocab size and `numberOfIterations`. **Never call `train()` on the main thread** — it will freeze the UI and can trigger the iOS watchdog. Wrap it in `DispatchQueue.global(qos: .userInitiated).async { ... }` (see §2.6) and hop back to `DispatchQueue.main` to update UI. Nearest-word / analogy scans (§3.3, §4.3) are lighter but still O(V·D); for large vocab, run them off-main too and debounce user input.

### 5.2. Determinism
With a fixed `seed` (default 1) the results are reproducible **given the same corpus and the same code path** (the seed feeds `Random`, which drives matrix init and shuffling). If you parallelize or change corpus order, reproducibility is lost.

### 5.3. Memory
The model holds one `Vector` of `layerSize` `Double`s per vocabulary word, plus (during training) a second `__wordVectorUpdate` matrix of the same size and the `__expTable` (1000 doubles). Rough embedding footprint after training: `vocabSize × layerSize × 8 bytes` (e.g. 10k words × 100 dims ≈ 8 MB). Keep `layerSize` modest (50–100) on device. `Vector` and `VectorizedWord` are **classes (reference types)** — mutating a word's vector in place mutates the model; always `copied()` before doing algebra (§4.2).

### 5.4. Corpus loading & file-reading quirks (the silent-failure zone)
- **`Corpus(fileName:)` reads from `Bundle.module` (the Corpus package's bundle), with a hardcoded `.txt` extension, and an empty `catch {}`.** It cannot read your app's files, and failures are silent (empty corpus) — see §2.2. **Use the in-memory `Corpus()` + `addSentence(s:)` path.**
- **Encoding is fixed to UTF-8** (`encoding: .utf8`). If you load text yourself, decode as UTF-8; non-UTF-8 files will throw when *you* read them (which is good — you control the error, unlike the library's swallowed one).
- **Tokenization splits only on the space character `" "`.** Tabs/newlines-within-a-line/multiple-spaces behave as described in §2.3. Normalize whitespace, lowercase, and strip punctuation yourself *before* building the corpus, or your vocabulary will be full of `word,` `word.` `Word` duplicates.
- An **empty or whitespace-only corpus** yields an empty `Vocabulary`; `NeuralNetwork.init` then builds a 0-row matrix and `train()` returns an empty `VectorizedDictionary` with **no error**. Assert `corpus.sentenceCount() > 0` and `model.size() > 0` in debug builds.

### 5.5. OOV (out-of-vocabulary) words
Any word not seen in the corpus has no vector. `getWord(name:)` returns `nil` for it, and `mostSimilarWord(name:)` returns `nil`. There is **no** subword/fallback handling. Always guard every lookup and surface a friendly "word not in vocabulary" message. (word2vec is also case- and punctuation-sensitive per §2.3, so `"King"` may be OOV even if `"king"` exists.)

### 5.6. ⚠️ Licensing — GPL-3.0
WordToVec-Swift (and the StarlangSoftware dependencies) are **GPL-3.0**. Linking GPL code into a shipped iOS app has copyleft/distribution implications and is generally **incompatible with the App Store's terms** for closed-source apps. **Flag this to the team/legal before committing to this dependency.** If GPL is a blocker, the algorithm is small enough (5 files) that a clean-room reimplementation, or a differently-licensed word2vec (or Core ML / NaturalLanguage's `NLEmbedding` for *pretrained* lookups — though it can't *train* on a custom corpus), may be preferable.

### 5.7. `Bundle.module` and tools version
`Corpus` and `Dictionary` bundle resources via `Bundle.module`, which requires `swift-tools-version ≥ 5.3`. Those packages declare 5.5, so this is fine in a modern Xcode. Just don't be surprised to see `*_Corpus.bundle` / `*_Dictionary.bundle` resource bundles in your app — they carry StarlangSoftware's Turkish/English data files (a few MB) you don't use but will ship unless you trim them.

---

## 6. Recommended integration path (summary)

1. **Resolve the licensing question first** (§5.6). If GPL is acceptable, proceed.
2. **Vendor the 5 `Sources/WordToVec/` files** into a local package/target and **patch the two `Matrix(...)` calls** to pass a `Random(seed: parameter.getSeed())` (§0.1 Option A). Keep Corpus/Dictionary/Math/Util as normal SPM dependencies (or vendor them too if you want to strip their resource bundles).
3. **Build corpora in memory** with `Corpus()` + `Sentence(sentence:)` after your own normalization (lowercase, punctuation → space, collapse whitespace) — never `Corpus(fileName:)` (§2.2, §2.3).
4. **Train off the main thread** (§2.6, §5.1). Cache the returned `VectorizedDictionary`.
5. **Nearest words / analogy:** use the cosine helpers in §3.3 and §4.3 (the built-in `mostSimilarWord` is single-result + dot-product only).
6. Guard **OOV** everywhere (§5.5) and assert non-empty corpus/model in debug (§5.4).

---

### Appendix A — exact source locations quoted in this doc

| API | Package @ tag | File |
|---|---|---|
| Package manifest & deps | WordToVec-Swift @ 1.0.0 | `Package.swift` |
| `NeuralNetwork` init/train, `Matrix(...)` call | WordToVec-Swift @ 1.0.0 | `Sources/WordToVec/NeuralNetwork.swift` |
| `WordToVecParameter` fields/getters/setters | WordToVec-Swift @ 1.0.0 | `Sources/WordToVec/WordToVecParameter.swift` |
| Corpus deps & resources | Corpus-Swift @ 1.0.17 | `Package.swift` |
| `Corpus` inits (incl. `Bundle.module` loader), methods | Corpus-Swift @ 1.0.17 | `Sources/Corpus/Corpus.swift` |
| `Sentence` inits, tokenization on `" "` | Corpus-Swift @ 1.0.17 | `Sources/Corpus/Sentence.swift` |
| Dictionary deps | Dictionary-Swift @ 1.0.11 | `Package.swift` |
| `VectorizedDictionary` (`mostSimilarWord` = dotProduct) | Dictionary-Swift @ 1.0.11/master | `Sources/Dictionary/VectorizedDictionary.swift` |
| `VectorizedWord` (`getVector()`) | Dictionary-Swift @ master | `Sources/Dictionary/VectorizedWord.swift` |
| `Dictionary` base (`size`, `getWord(name:)`, `getWordWithIndex`) | Dictionary-Swift @ master | `Sources/Dictionary/Dictionary.swift` |
| `Word` (`getName()`) | Dictionary-Swift @ 1.0.11 | `Sources/Dictionary/Word.swift` |
| Math deps (→ Util 1.0.8) | Math-Swift @ 1.0.11 | `Package.swift` |
| `Vector` full API | Math-Swift @ 1.0.11 | `Sources/Math/Vector.swift` |
| `Matrix` inits (4-arg vs 5-arg boundary) | Math-Swift @ 1.0.5 / 1.0.8 / 1.0.10 / 1.0.11 | `Sources/Math/Matrix.swift` |
| `Random` API | Util-Swift @ 1.0.8 | `Sources/Util/Random.swift` |
| DataStructure (no deps) | DataStructure-Swift @ 1.0.4 | `Package.swift` |

*All signatures and bodies above were read directly from the GitHub-hosted source at the stated tags; nothing here is inferred from documentation alone.*
