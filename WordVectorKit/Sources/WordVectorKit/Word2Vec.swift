//  Word2Vec.swift
//
//  The training algorithm in this file is a faithful Swift port of Google's
//  word2vec.c (Tomas Mikolov et al.), which is licensed under the Apache
//  License, Version 2.0:
//
//      Copyright 2013 Google Inc. All Rights Reserved.
//      Licensed under the Apache License, Version 2.0 (the "License").
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  The math (exp table, unigram^0.75 negative-sampling table, subsampling,
//  the LCG RNG constants, alpha decay, and both CBOW & skip-gram
//  negative-sampling gradient updates) is translated directly from that
//  reference. See the reference at docs/reference-word2vec.c and the
//  `// ref: word2vec.c line NNN` markers below.
//
//  Key deliberate difference from the C: NO file I/O. The C streams the corpus
//  from disk with a `</s>` sentinel word (index 0) to mark sentence ends. We
//  instead receive already-tokenized sentences as `[String]` and treat each
//  sentence as a bounded unit — context windows never cross a sentence
//  boundary. This is equivalent in effect to the C's `</s>` handling (which
//  clips the window at sentence ends) but cleaner, so there is no `</s>`
//  entry in our vocabulary.

import Foundation
import Accelerate

/// Shared progress counter. Tracks the running total of processed words so the `progress` closure
/// can report a monotonic 0…1 fraction across workers. Only this scalar is synchronized; in the
/// Hogwild path the model weights intentionally remain lock-free. Alpha decay is computed
/// separately as a deterministic function of words processed (see `deterministicAlpha`), so it no
/// longer depends on the order in which workers reach this counter. ref: word2vec.c lines 387-399
private final class TrainingProgressCounter {
    private let lock = NSLock()
    private var wordCountActual = 0

    /// Adds `count` processed words and reports the new progress fraction. `denominator` is accepted
    /// for symmetry with the C's alpha math but only clamps the reported fraction to `totalTrainWords`.
    func add(_ count: Int, denominator: Int, totalTrainWords: Int, progress: ((Double) -> Void)?) {
        lock.lock()
        wordCountActual += count
        let fraction = min(1.0, Double(wordCountActual) / Double(totalTrainWords))
        lock.unlock()

        progress?(fraction)
    }
}

/// Pure-Swift, in-memory port of word2vec (CBOW + skip-gram, negative sampling only;
/// hierarchical softmax is intentionally omitted). Training uses the reference implementation's
/// Hogwild parallelism: workers update shared weights asynchronously without locks.
public final class Word2Vec {

    // MARK: - Ported constants (ref: word2vec.c lines 22-23, 49)

    private static let expTableSize = 1000   // ref: word2vec.c line 22 (EXP_TABLE_SIZE)
    private static let maxExpInt = 6          // ref: word2vec.c line 23 (MAX_EXP, an int #define)
    private static let maxExp: Float = Float(maxExpInt) // Float form for the +/- clamp comparisons.
    private static let unigramPower = 0.75    // ref: word2vec.c line 55 (power)

    /// Sigmoid-table lookup scale. In the C this is `EXP_TABLE_SIZE / MAX_EXP / 2` where both
    /// are int #defines, so it is INTEGER division: 1000 / 6 / 2 = 83 (NOT 83.333). Computing
    /// it as float division diverges on ~87.5% of lookups and, at f == MAX_EXP exactly, would
    /// index expTable[1000] (the uninitialized slot). We reproduce the integer math here.
    /// ref: word2vec.c lines 456, 481, 512, 537
    private static let expScale = Float(expTableSize / maxExpInt / 2) // == 83

    // The negative-sampling table size (C: `table_size = 1e8`, ref: word2vec.c line 49)
    // is taken from parameters. We default it lower than the C (see Word2VecParameters) to
    // avoid a large single allocation on device; it can also be reduced further in tests.

    // MARK: - Configuration

    private let params: Word2VecParameters
    private let vectorSize: Int

    // MARK: - Precomputed sigmoid/exp lookup table

    /// f(x) = e^x / (e^x + 1) sampled over [-MAX_EXP, MAX_EXP]. ref: word2vec.c lines 708-712
    private let expTable: [Float]

    // MARK: - Init

    public init(parameters: Word2VecParameters) {
        self.params = parameters
        self.vectorSize = parameters.vectorSize

        // Precompute the exp() table exactly as the C does. ref: word2vec.c lines 708-712
        var table = [Float](repeating: 0, count: Word2Vec.expTableSize + 1)
        for i in 0..<Word2Vec.expTableSize {
            // expTable[i] = exp((i / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
            let x = (Float(i) / Float(Word2Vec.expTableSize) * 2 - 1) * Word2Vec.maxExp
            let e = expf(x)
            table[i] = e / (e + 1) // f(x) = x / (x + 1)
        }
        self.expTable = table
    }

    // MARK: - Test-only accessors (internal; used by @testable tests to verify the port)

    /// The integer-math sigmoid-table scale (must equal 83 to match the C). ref: fix #4
    internal static var expScaleForTesting: Int { expTableSize / maxExpInt / 2 }

    /// Reads a value from the precomputed sigmoid/exp table (for bounds/zero-slot tests).
    internal func expTableValueForTesting(_ idx: Int) -> Float { expTable[idx] }

    // MARK: - Vocabulary

    /// A vocabulary word with its corpus count. Mirrors `struct vocab_word` (cn only;
    /// we drop the Huffman `point`/`code` fields since we skip hierarchical softmax).
    private struct VocabWord {
        var word: String
        var cn: Int
    }

    // MARK: - Training

    /// Trains on already-preprocessed sentences (each element = one space-tokenized sentence).
    /// `progress` is called serially from training worker threads with a value in `0.0...1.0`.
    /// Callers that update UI must dispatch to the main thread.
    public func train(sentences: [String], progress: ((Double) -> Void)?) -> WordEmbeddings {
        // 1) Build vocabulary with counts, apply min-count, sort by frequency descending.
        let (vocab, wordIndex) = buildVocabulary(sentences: sentences)

        let vocabSize = vocab.count
        guard vocabSize > 0 else {
            // Nothing survived min-count filtering — return an empty model.
            return WordEmbeddings(words: [], vectorSize: vectorSize, storage: [])
        }

        // Tokenize sentences once into arrays of vocab indices (dropping OOV tokens).
        // Each inner array is a "sentence" that bounds context windows.
        var tokenizedSentences: [[Int]] = []
        tokenizedSentences.reserveCapacity(sentences.count)
        var trainWords = 0
        for sentence in sentences {
            var indices: [Int] = []
            for token in sentence.split(separator: " ") {
                if let idx = wordIndex[String(token)] {
                    indices.append(idx)
                    trainWords += 1
                }
            }
            if !indices.isEmpty {
                tokenizedSentences.append(indices)
            }
        }

        // 2) Init network weights. ref: word2vec.c lines 350-372
        //    syn0 (input vectors) random via LCG; syn1neg (output vectors) zeroed.
        var syn0 = [Float](repeating: 0, count: vocabSize * vectorSize)
        var syn1neg = [Float](repeating: 0, count: vocabSize * vectorSize)

        // ref: word2vec.c lines 367-370 — the LCG seed here is 1 in the C; we seed from params.
        var initRandom: UInt64 = params.seed == 0 ? 1 : params.seed
        for a in 0..<vocabSize {
            for b in 0..<vectorSize {
                initRandom = initRandom &* 25214903917 &+ 11
                // syn0 = (((next_random & 0xFFFF)/65536) - 0.5) / layer1_size
                let r = Float(initRandom & 0xFFFF) / Float(65536)
                syn0[a * vectorSize + b] = (r - 0.5) / Float(vectorSize)
            }
        }
        // syn1neg is already all-zero (ref: word2vec.c lines 364-365).

        // 3) Build the unigram negative-sampling table (cn^0.75). ref: word2vec.c lines 52-68
        let unigramTable = buildUnigramTable(vocab: vocab)

        // 4) Training loop — port of TrainModelThread. ref: word2vec.c lines 374-555
        let numThreads = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let config = TrainingConfig(
            startingAlpha: params.initialAlpha,
            iterCount: params.iterations,
            // Clamp window to >= 1: `nextRandom % UInt64(window)` would trap on window == 0.
            // A window of 0 has no context anyway, so 1 is the smallest meaningful value.
            window: max(1, params.window),
            negative: params.negativeSamples,
            sample: params.subsample,
            useCBOW: params.useCBOW,
            vocabSize: vocabSize,
            trainWords: trainWords,
            numThreads: numThreads
        )

        if params.deterministic {
            trainDeterministic(config: config, vocab: vocab, tokenizedSentences: tokenizedSentences,
                               unigramTable: unigramTable, syn0: &syn0, syn1neg: &syn1neg,
                               progress: progress)
        } else {
            trainHogwild(config: config, vocab: vocab, tokenizedSentences: tokenizedSentences,
                         unigramTable: unigramTable, syn0: &syn0, syn1neg: &syn1neg,
                         progress: progress)
        }

        progress?(1.0)

        // Package syn0 (the input word vectors) as the model, exactly like the C's output.
        // ref: word2vec.c lines 574-580 — it writes syn0 rows as the word vectors.
        let words = vocab.map { $0.word }
        return WordEmbeddings(words: words, vectorSize: vectorSize, storage: syn0)
    }

    /// Immutable per-run training constants, shared by both the Hogwild and deterministic paths.
    private struct TrainingConfig {
        let startingAlpha: Double
        let iterCount: Int
        let window: Int
        let negative: Int
        let sample: Double
        let useCBOW: Bool
        let vocabSize: Int
        let trainWords: Int
        let numThreads: Int

        /// Total words the C would process across all epochs (iter * train_words).
        var totalTrainWords: Int { max(1, iterCount * trainWords) }
        /// Denominator for alpha decay. ref: word2vec.c line 393
        var alphaDenominator: Int { iterCount * trainWords + 1 }
        var sampleTrainWords: Double { sample * Double(trainWords) }
    }

    /// Runs one worker's pass over its sentence chunk for a single epoch, updating `syn0`/`syn1neg`
    /// through the supplied pointers. Both training paths share this so the *only* difference between
    /// them is whether those pointers address the shared matrices (Hogwild) or a per-worker private
    /// copy (deterministic). `alpha` is computed deterministically from `baseWordCount + localWordCount`
    /// so it never depends on thread interleaving. Returns the number of (pre-subsample) train words
    /// this worker consumed, for progress reporting.
    private func runEpochChunk(threadId: Int, epoch: Int, config: TrainingConfig,
                               vocab: [VocabWord], tokenizedSentences: [[Int]],
                               unigramTable: [Int32], nextRandom: inout UInt64,
                               syn0: UnsafeMutablePointer<Float>, syn1neg: UnsafeMutablePointer<Float>,
                               neu1: UnsafeMutablePointer<Float>, neu1e: UnsafeMutablePointer<Float>,
                               sen: inout [Int], reportChunkWords: (Int) -> Void) -> Int {
        let numThreads = config.numThreads
        // Contiguous sentence chunks are the in-memory equivalent of the C port's per-thread
        // file offsets. ref: word2vec.c lines 385-386
        let chunkStart = tokenizedSentences.count * threadId / numThreads
        let chunkEnd = tokenizedSentences.count * (threadId + 1) / numThreads

        // Words fully processed before this epoch begins. Used to make alpha a pure function of
        // (epoch, local progress) rather than of the shared, order-dependent progress counter.
        let baseWordCount = epoch * config.trainWords
        var localWordCount = 0
        var lastReported = 0

        expTable.withUnsafeBufferPointer { expBuf in
            let expp = expBuf.baseAddress!

            for sentenceIndex in chunkStart..<chunkEnd {
                let sentence = tokenizedSentences[sentenceIndex]

                // 3a) Apply subsampling to build this sentence's kept-token list.
                //     ref: word2vec.c lines 407-412
                sen.removeAll(keepingCapacity: true)
                sen.reserveCapacity(sentence.count)
                for word in sentence {
                    localWordCount += 1
                    if config.sample > 0 {
                        let cn = Double(vocab[word].cn)
                        // ran = (sqrt(cn / (sample*train_words)) + 1) * (sample*train_words) / cn
                        let ran = (sqrt(cn / config.sampleTrainWords) + 1) * config.sampleTrainWords / cn
                        nextRandom = nextRandom &* 25214903917 &+ 11
                        if ran < Double(nextRandom & 0xFFFF) / Double(65536) {
                            continue // discard this frequent word
                        }
                    }
                    sen.append(word)
                }

                let sentenceLength = sen.count

                // Deterministic alpha decay + periodic progress. ref: word2vec.c lines 387-399.
                // alpha depends only on (baseWordCount + localWordCount), never on scheduling.
                let alpha = deterministicAlpha(processed: baseWordCount + localWordCount, config: config)
                if localWordCount - lastReported > 10000 {
                    reportChunkWords(localWordCount - lastReported)
                    lastReported = localWordCount
                }

                // 3b) For each position in the sentence, run the CBOW or skip-gram update.
                for sentencePosition in 0..<sentenceLength {
                    let word = sen[sentencePosition]

                    // Zero the hidden accumulators. ref: word2vec.c lines 431-432
                    for c in 0..<vectorSize { neu1[c] = 0 }
                    for c in 0..<vectorSize { neu1e[c] = 0 }

                    // Random window shrink b in [0, window). ref: word2vec.c lines 433-434
                    nextRandom = nextRandom &* 25214903917 &+ 11
                    let b = Int(nextRandom % UInt64(config.window))

                    if config.useCBOW {
                        trainCBOW(word: word, b: b, window: config.window, negative: config.negative,
                                  sentencePosition: sentencePosition, sentenceLength: sentenceLength,
                                  sen: sen, vocabSize: config.vocabSize, alpha: Float(alpha),
                                  unigramTable: unigramTable, nextRandom: &nextRandom,
                                  syn0: syn0, syn1neg: syn1neg, expTable: expp,
                                  neu1: neu1, neu1e: neu1e)
                    } else {
                        trainSkipGram(word: word, b: b, window: config.window, negative: config.negative,
                                      sentencePosition: sentencePosition, sentenceLength: sentenceLength,
                                      sen: sen, vocabSize: config.vocabSize, alpha: Float(alpha),
                                      unigramTable: unigramTable, nextRandom: &nextRandom,
                                      syn0: syn0, syn1neg: syn1neg, expTable: expp,
                                      neu1e: neu1e)
                    }
                }
            }

            if localWordCount - lastReported > 0 {
                reportChunkWords(localWordCount - lastReported)
            }
        }

        return localWordCount
    }

    /// Alpha decay as a pure function of words processed so far. ref: word2vec.c lines 391-394
    private func deterministicAlpha(processed: Int, config: TrainingConfig) -> Double {
        var alpha = config.startingAlpha * (1 - Double(processed) / Double(config.alphaDenominator))
        if alpha < config.startingAlpha * 0.0001 { alpha = config.startingAlpha * 0.0001 }
        return alpha
    }

    // MARK: - Hogwild training (fast, lock-free, not run-to-run reproducible)

    /// Original Hogwild path: all workers update the shared weights lock-free. ref: word2vec.c lines 374-555
    private func trainHogwild(config: TrainingConfig, vocab: [VocabWord],
                              tokenizedSentences: [[Int]], unigramTable: [Int32],
                              syn0: inout [Float], syn1neg: inout [Float],
                              progress: ((Double) -> Void)?) {
        let progressCounter = TrainingProgressCounter()

        syn0.withUnsafeMutableBufferPointer { syn0Buf in
        syn1neg.withUnsafeMutableBufferPointer { syn1Buf in
            let syn0p = syn0Buf.baseAddress!
            let syn1p = syn1Buf.baseAddress!

            DispatchQueue.concurrentPerform(iterations: config.numThreads) { threadId in
                var nextRandom = UInt64(threadId) // ref: word2vec.c line 378
                var neu1 = [Float](repeating: 0, count: vectorSize)
                var neu1e = [Float](repeating: 0, count: vectorSize)
                var sen: [Int] = []

                neu1.withUnsafeMutableBufferPointer { neu1Buf in
                neu1e.withUnsafeMutableBufferPointer { neu1eBuf in
                    let neu1p = neu1Buf.baseAddress!
                    let neu1ep = neu1eBuf.baseAddress!

                    // `local_iter` belongs to each thread in the C. ref: word2vec.c line 377
                    for epoch in 0..<config.iterCount {
                        _ = runEpochChunk(
                            threadId: threadId, epoch: epoch, config: config, vocab: vocab,
                            tokenizedSentences: tokenizedSentences, unigramTable: unigramTable,
                            nextRandom: &nextRandom, syn0: syn0p, syn1neg: syn1p,
                            neu1: neu1p, neu1e: neu1ep, sen: &sen,
                            reportChunkWords: { chunkWords in
                                progressCounter.add(
                                    chunkWords, denominator: config.alphaDenominator,
                                    totalTrainWords: config.totalTrainWords, progress: progress)
                            })
                    }
                }}
            }
        }}
    }

    // MARK: - Deterministic training (synchronous SGD, bit-exact run-to-run)

    /// Bit-exact reproducible path (README: "Deterministic parallel training"). Each worker trains
    /// on a **private copy** of `syn0`/`syn1neg`, so it sees only its own updates during an epoch and
    /// never races another worker. At each epoch barrier the per-worker deltas (private − snapshot)
    /// are summed into the shared weights **in fixed thread order 0,1,2,…**. Float addition isn't
    /// associative, so the fixed order is what makes the merged result identical on every run.
    private func trainDeterministic(config: TrainingConfig, vocab: [VocabWord],
                                    tokenizedSentences: [[Int]], unigramTable: [Int32],
                                    syn0: inout [Float], syn1neg: inout [Float],
                                    progress: ((Double) -> Void)?) {
        let numThreads = config.numThreads
        let matrixCount = config.vocabSize * vectorSize
        let progressCounter = TrainingProgressCounter()

        // Per-worker private weight copies, flattened into one contiguous buffer each so workers can
        // address disjoint slices through raw pointers (concurrent access to distinct regions of one
        // buffer is safe; concurrent CoW access to an [[Float]] of independent arrays is not). Slice
        // `t` is `[t*matrixCount ..< (t+1)*matrixCount)`. Reused across epochs; refilled from the
        // shared snapshot at the start of every epoch so each epoch starts from the merged model.
        var privateSyn0 = [Float](repeating: 0, count: numThreads * matrixCount)
        var privateSyn1 = [Float](repeating: 0, count: numThreads * matrixCount)

        // Each worker's RNG state persists across epochs, seeded once from its thread id, exactly as
        // the reference does across its `local_iter` loop. Stored in a shared buffer; each worker only
        // ever touches its own disjoint element. ref: word2vec.c lines 377-378
        var rngState = [UInt64](repeating: 0, count: numThreads)
        for threadId in 0..<numThreads { rngState[threadId] = UInt64(threadId) }

        privateSyn0.withUnsafeMutableBufferPointer { priv0Buf in
        privateSyn1.withUnsafeMutableBufferPointer { priv1Buf in
        rngState.withUnsafeMutableBufferPointer { rngBuf in
            let priv0Base = priv0Buf.baseAddress!
            let priv1Base = priv1Buf.baseAddress!
            let rngBase = rngBuf.baseAddress!

            for epoch in 0..<config.iterCount {
                // Snapshot the shared weights so every worker reads the SAME epoch-start model.
                let snapshot0 = syn0
                let snapshot1 = syn1neg

                snapshot0.withUnsafeBufferPointer { snap0 in
                snapshot1.withUnsafeBufferPointer { snap1 in
                    let snap0p = snap0.baseAddress!
                    let snap1p = snap1.baseAddress!

                    DispatchQueue.concurrentPerform(iterations: numThreads) { threadId in
                        let priv0p = priv0Base + threadId * matrixCount
                        let priv1p = priv1Base + threadId * matrixCount
                        // Start this epoch from the shared snapshot: private = snapshot.
                        priv0p.update(from: snap0p, count: matrixCount)
                        priv1p.update(from: snap1p, count: matrixCount)

                        var nextRandom = rngBase[threadId]
                        var neu1 = [Float](repeating: 0, count: vectorSize)
                        var neu1e = [Float](repeating: 0, count: vectorSize)
                        var sen: [Int] = []

                        neu1.withUnsafeMutableBufferPointer { neu1Buf in
                        neu1e.withUnsafeMutableBufferPointer { neu1eBuf in
                            _ = runEpochChunk(
                                threadId: threadId, epoch: epoch, config: config, vocab: vocab,
                                tokenizedSentences: tokenizedSentences, unigramTable: unigramTable,
                                nextRandom: &nextRandom, syn0: priv0p, syn1neg: priv1p,
                                neu1: neu1Buf.baseAddress!, neu1e: neu1eBuf.baseAddress!, sen: &sen,
                                // Intra-epoch progress. Safe to report from workers: the counter only
                                // drives the UI fraction here — alpha is computed deterministically, so
                                // this does NOT feed back into the trained weights.
                                reportChunkWords: { chunkWords in
                                    progressCounter.add(
                                        chunkWords, denominator: config.alphaDenominator,
                                        totalTrainWords: config.totalTrainWords, progress: progress)
                                })
                        }}
                        rngBase[threadId] = nextRandom // carry this worker's RNG into the next epoch
                    }
                }}

                // Deterministic reduction: shared += sum over threads of (private − snapshot), summed
                // in fixed thread order. Because each worker's private matrix already equals
                // snapshot + its own delta, adding (private − snapshot) for threads 0,1,2,… in order
                // yields a result identical on every run regardless of how the workers were scheduled.
                syn0.withUnsafeMutableBufferPointer { shared0 in
                syn1neg.withUnsafeMutableBufferPointer { shared1 in
                    let shared0p = shared0.baseAddress!
                    let shared1p = shared1.baseAddress!
                    snapshot0.withUnsafeBufferPointer { snap0 in
                    snapshot1.withUnsafeBufferPointer { snap1 in
                        let snap0p = snap0.baseAddress!
                        let snap1p = snap1.baseAddress!
                        for threadId in 0..<numThreads {
                            accumulateDelta(shared: shared0p, priv: priv0Base + threadId * matrixCount,
                                            snapshot: snap0p, count: matrixCount)
                            accumulateDelta(shared: shared1p, priv: priv1Base + threadId * matrixCount,
                                            snapshot: snap1p, count: matrixCount)
                        }
                    }}
                }}
                // Progress was reported incrementally by the workers above; nothing to add here.
            }
        }}}
    }

    /// `shared[i] += priv[i] - snapshot[i]` for every element. Called once per thread per matrix in
    /// fixed thread order to keep the reduction bit-exact. `delta = priv - snapshot; shared += delta`.
    private func accumulateDelta(shared: UnsafeMutablePointer<Float>, priv: UnsafePointer<Float>,
                                 snapshot: UnsafePointer<Float>, count: Int) {
        for i in 0..<count {
            shared[i] += priv[i] - snapshot[i]
        }
    }

    // MARK: - CBOW update (ref: word2vec.c lines 435-494)

    private func trainCBOW(word: Int, b: Int, window: Int, negative: Int,
                           sentencePosition: Int, sentenceLength: Int, sen: [Int],
                           vocabSize: Int, alpha: Float, unigramTable: [Int32],
                           nextRandom: inout UInt64,
                           syn0: UnsafeMutablePointer<Float>,
                           syn1neg: UnsafeMutablePointer<Float>,
                           expTable: UnsafePointer<Float>,
                           neu1: UnsafeMutablePointer<Float>,
                           neu1e: UnsafeMutablePointer<Float>) {
        let layer1Size = vectorSize

        // in -> hidden: sum the context word vectors into neu1. ref: word2vec.c lines 437-446
        var cw = 0
        var a = b
        while a < window * 2 + 1 - b {
            if a != window {
                let c = sentencePosition - window + a
                if c >= 0 && c < sentenceLength {
                    let lastWord = sen[c]
                    let base = lastWord * layer1Size
                    vDSP_vadd(neu1, 1, syn0 + base, 1, neu1, 1, vDSP_Length(layer1Size))
                    cw += 1
                }
            }
            a += 1
        }

        guard cw > 0 else { return }

        // Average the context vectors. ref: word2vec.c line 448
        var cwFloat = Float(cw)
        vDSP_vsdiv(neu1, 1, &cwFloat, neu1, 1, vDSP_Length(layer1Size))

        // NEGATIVE SAMPLING. ref: word2vec.c lines 465-484
        for d in 0..<(negative + 1) {
            let target: Int
            let label: Float
            if d == 0 {
                target = word
                label = 1
            } else {
                // With a 1-word vocab there are no valid negatives; `% UInt64(vocabSize - 1)`
                // would trap on `% 0`. Skip the negative draw entirely in that case.
                guard vocabSize > 1 else { continue }
                nextRandom = nextRandom &* 25214903917 &+ 11
                var t = Int(unigramTable[Int((nextRandom >> 16) % UInt64(unigramTable.count))])
                if t == 0 { t = Int(nextRandom % UInt64(vocabSize - 1)) + 1 }
                if t == word { continue }
                target = t
                label = 0
            }
            let l2 = target * layer1Size

            // f = dot(neu1, syn1neg[target]) ref: word2vec.c lines 477-478
            var f: Float = 0
            vDSP_dotpr(neu1, 1, syn1neg + l2, 1, &f, vDSP_Length(layer1Size))

            // g = (label - sigmoid(f)) * alpha, with the clipped-exp-table lookup.
            // ref: word2vec.c lines 479-481
            var g: Float
            if f > Word2Vec.maxExp {
                g = (label - 1) * alpha
            } else if f < -Word2Vec.maxExp {
                g = (label - 0) * alpha
            } else {
                let idx = Int((f + Word2Vec.maxExp) * Word2Vec.expScale)
                g = (label - expTable[idx]) * alpha
            }

            // Accumulate hidden error and update output weights. ref: word2vec.c lines 482-483
            vDSP_vsma(syn1neg + l2, 1, &g, neu1e, 1, neu1e, 1, vDSP_Length(layer1Size))
            vDSP_vsma(neu1, 1, &g, syn1neg + l2, 1, syn1neg + l2, 1, vDSP_Length(layer1Size))
        }

        // hidden -> in: apply the accumulated error to each context word. ref: word2vec.c lines 486-493
        a = b
        while a < window * 2 + 1 - b {
            if a != window {
                let c = sentencePosition - window + a
                if c >= 0 && c < sentenceLength {
                    let lastWord = sen[c]
                    let base = lastWord * layer1Size
                    vDSP_vadd(syn0 + base, 1, neu1e, 1, syn0 + base, 1, vDSP_Length(layer1Size))
                }
            }
            a += 1
        }
    }

    // MARK: - Skip-gram update (ref: word2vec.c lines 495-544)

    private func trainSkipGram(word: Int, b: Int, window: Int, negative: Int,
                               sentencePosition: Int, sentenceLength: Int, sen: [Int],
                               vocabSize: Int, alpha: Float, unigramTable: [Int32],
                               nextRandom: inout UInt64,
                               syn0: UnsafeMutablePointer<Float>,
                               syn1neg: UnsafeMutablePointer<Float>,
                               expTable: UnsafePointer<Float>,
                               neu1e: UnsafeMutablePointer<Float>) {
        let layer1Size = vectorSize

        // For each context word around the center `word`. ref: word2vec.c lines 496-502
        var a = b
        while a < window * 2 + 1 - b {
            if a != window {
                let c = sentencePosition - window + a
                if c >= 0 && c < sentenceLength {
                    let lastWord = sen[c]
                    let l1 = lastWord * layer1Size

                    // Zero the error accumulator for this context word. ref: word2vec.c line 503
                    for j in 0..<layer1Size { neu1e[j] = 0 }

                    // NEGATIVE SAMPLING. ref: word2vec.c lines 521-540
                    for d in 0..<(negative + 1) {
                        let target: Int
                        let label: Float
                        if d == 0 {
                            target = word
                            label = 1
                        } else {
                            // With a 1-word vocab there are no valid negatives; `% UInt64(vocabSize - 1)`
                            // would trap on `% 0`. Skip the negative draw entirely in that case.
                            guard vocabSize > 1 else { continue }
                            nextRandom = nextRandom &* 25214903917 &+ 11
                            var t = Int(unigramTable[Int((nextRandom >> 16) % UInt64(unigramTable.count))])
                            if t == 0 { t = Int(nextRandom % UInt64(vocabSize - 1)) + 1 }
                            if t == word { continue }
                            target = t
                            label = 0
                        }
                        let l2 = target * layer1Size

                        // f = dot(syn0[context], syn1neg[target]) ref: word2vec.c lines 533-534
                        var f: Float = 0
                        vDSP_dotpr(syn0 + l1, 1, syn1neg + l2, 1, &f, vDSP_Length(layer1Size))

                        // g = (label - sigmoid(f)) * alpha. ref: word2vec.c lines 535-537
                        var g: Float
                        if f > Word2Vec.maxExp {
                            g = (label - 1) * alpha
                        } else if f < -Word2Vec.maxExp {
                            g = (label - 0) * alpha
                        } else {
                            let idx = Int((f + Word2Vec.maxExp) * Word2Vec.expScale)
                            g = (label - expTable[idx]) * alpha
                        }

                        // Accumulate error and update output weights. ref: word2vec.c lines 538-539
                        vDSP_vsma(syn1neg + l2, 1, &g, neu1e, 1, neu1e, 1, vDSP_Length(layer1Size))
                        vDSP_vsma(syn0 + l1, 1, &g, syn1neg + l2, 1, syn1neg + l2, 1, vDSP_Length(layer1Size))
                    }

                    // Learn input -> hidden: apply accumulated error to the context word.
                    // ref: word2vec.c line 542
                    vDSP_vadd(syn0 + l1, 1, neu1e, 1, syn0 + l1, 1, vDSP_Length(layer1Size))
                }
            }
            a += 1
        }
    }

    // MARK: - Vocabulary build (ref: word2vec.c lines 272-308, 155-182)

    /// Builds the vocabulary with counts, drops words below `minCount`, and sorts by
    /// frequency descending. Returns the vocab plus a word→index lookup.
    private func buildVocabulary(sentences: [String]) -> (vocab: [VocabWord], index: [String: Int]) {
        // Count occurrences. ref: word2vec.c lines 284-300
        var counts: [String: Int] = [:]
        for sentence in sentences {
            for token in sentence.split(separator: " ") {
                counts[String(token), default: 0] += 1
            }
        }

        // Drop words below min-count. ref: word2vec.c lines 164-167
        var kept: [VocabWord] = []
        kept.reserveCapacity(counts.count)
        for (word, cn) in counts where cn >= params.minCount {
            kept.append(VocabWord(word: word, cn: cn))
        }

        // Sort by frequency descending; tie-break on word for determinism (the C's qsort
        // is not stable, but a deterministic tie-break gives reproducible Swift results).
        // ref: word2vec.c lines 147-159 (VocabCompare / SortVocab)
        kept.sort { a, b in
            if a.cn != b.cn { return a.cn > b.cn }
            return a.word < b.word
        }

        var index: [String: Int] = [:]
        index.reserveCapacity(kept.count)
        for (i, vw) in kept.enumerated() {
            index[vw.word] = i
        }

        return (kept, index)
    }

    // MARK: - Unigram negative-sampling table (ref: word2vec.c lines 52-68)

    /// Builds the negative-sampling table using the `cn^0.75` distribution.
    /// Stored as `Int32` to match the C's `int *table`; the entry count comes from
    /// `params.unigramTableSize` (default 1e7 = ~40 MB, vs. the C's 1e8 = ~400 MB).
    private func buildUnigramTable(vocab: [VocabWord]) -> [Int32] {
        let vocabSize = vocab.count
        // Clamp to >= 1: a zero-length table would make `% UInt64(unigramTable.count)` trap
        // at every negative-sampling draw.
        let tableSize = max(1, params.unigramTableSize)
        var table = [Int32](repeating: 0, count: tableSize)

        let power = Word2Vec.unigramPower

        // train_words_pow = sum_a pow(cn_a, power). ref: word2vec.c line 57
        var trainWordsPow: Double = 0
        for a in 0..<vocabSize {
            trainWordsPow += pow(Double(vocab[a].cn), power)
        }

        // Fill the table proportionally to cn^0.75. ref: word2vec.c lines 58-67
        var i = 0
        var d1 = pow(Double(vocab[0].cn), power) / trainWordsPow
        for a in 0..<tableSize {
            table[a] = Int32(i)
            if Double(a) / Double(tableSize) > d1 {
                i += 1
                if i < vocabSize {
                    d1 += pow(Double(vocab[i].cn), power) / trainWordsPow
                }
            }
            if i >= vocabSize { i = vocabSize - 1 }
        }

        return table
    }
}
