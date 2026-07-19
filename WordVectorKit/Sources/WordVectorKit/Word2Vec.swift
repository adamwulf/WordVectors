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

/// Pure-Swift, in-memory port of word2vec (CBOW + skip-gram, negative sampling only;
/// hierarchical softmax is intentionally omitted). Single-threaded for correctness.
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
    /// `progress` is called periodically on the calling thread with a value in `0.0...1.0`.
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
        let startingAlpha = params.initialAlpha
        var alpha = startingAlpha
        let iterCount = params.iterations
        // Total words the C would process across all epochs (iter * train_words).
        let totalTrainWords = max(1, iterCount * trainWords)

        // Per-thread RNG in the C is seeded from the thread id (0 for a single thread).
        // ref: word2vec.c line 378 — `next_random = (long long)id;`
        var nextRandom: UInt64 = 0

        var wordCountActual = 0
        var wordCount = 0
        var lastWordCount = 0

        // Scratch buffers reused across updates (ref: word2vec.c lines 382-383).
        var neu1 = [Float](repeating: 0, count: vectorSize)
        var neu1e = [Float](repeating: 0, count: vectorSize)

        // Clamp window to >= 1: `nextRandom % UInt64(window)` would trap on window == 0.
        // A window of 0 has no context anyway, so 1 is the smallest meaningful value.
        let window = max(1, params.window)
        let negative = params.negativeSamples
        let sample = params.subsample
        let useCBOW = params.useCBOW
        let trainWordsDouble = Double(trainWords)
        let sampleTrainWords = sample * trainWordsDouble

        // Reuse the kept-token buffer across every sentence and epoch.
        var sen: [Int] = []

        syn0.withUnsafeMutableBufferPointer { syn0Buf in
        syn1neg.withUnsafeMutableBufferPointer { syn1Buf in
        expTable.withUnsafeBufferPointer { expBuf in
        neu1.withUnsafeMutableBufferPointer { neu1Buf in
        neu1e.withUnsafeMutableBufferPointer { neu1eBuf in

            let syn0p = syn0Buf.baseAddress!
            let syn1p = syn1Buf.baseAddress!
            let expp = expBuf.baseAddress!
            let neu1p = neu1Buf.baseAddress!
            let neu1ep = neu1eBuf.baseAddress!

            for _ in 0..<iterCount {
                for sentence in tokenizedSentences {
                    // 3a) Apply subsampling to build this sentence's kept-token list.
                    //     ref: word2vec.c lines 407-412
                    sen.removeAll(keepingCapacity: true)
                    sen.reserveCapacity(sentence.count)
                    for word in sentence {
                        wordCount += 1
                        if sample > 0 {
                            let cn = Double(vocab[word].cn)
                            // ran = (sqrt(cn / (sample*train_words)) + 1) * (sample*train_words) / cn
                            let ran = (sqrt(cn / sampleTrainWords) + 1) * sampleTrainWords / cn
                            nextRandom = nextRandom &* 25214903917 &+ 11
                            if ran < Double(nextRandom & 0xFFFF) / Double(65536) {
                                continue // discard this frequent word
                            }
                        }
                        sen.append(word)
                    }

                    let sentenceLength = sen.count

                    // Periodic alpha decay + progress. ref: word2vec.c lines 387-399
                    if wordCount - lastWordCount > 10000 {
                        wordCountActual += wordCount - lastWordCount
                        lastWordCount = wordCount
                        // alpha = starting_alpha * (1 - word_count_actual / (iter*train_words + 1))
                        alpha = startingAlpha * (1 - Double(wordCountActual) / Double(iterCount * trainWords + 1))
                        if alpha < startingAlpha * 0.0001 { alpha = startingAlpha * 0.0001 }
                        progress?(min(1.0, Double(wordCountActual) / Double(totalTrainWords)))
                    }

                    // 3b) For each position in the sentence, run the CBOW or skip-gram update.
                    for sentencePosition in 0..<sentenceLength {
                        let word = sen[sentencePosition]

                        // Zero the hidden accumulators. ref: word2vec.c lines 431-432
                        for c in 0..<vectorSize { neu1p[c] = 0 }
                        for c in 0..<vectorSize { neu1ep[c] = 0 }

                        // Random window shrink b in [0, window). ref: word2vec.c lines 433-434
                        nextRandom = nextRandom &* 25214903917 &+ 11
                        let b = Int(nextRandom % UInt64(window))

                        if useCBOW {
                            trainCBOW(word: word, b: b, window: window, negative: negative,
                                      sentencePosition: sentencePosition, sentenceLength: sentenceLength,
                                      sen: sen, vocabSize: vocabSize, alpha: Float(alpha),
                                      unigramTable: unigramTable, nextRandom: &nextRandom,
                                      syn0: syn0p, syn1neg: syn1p, expTable: expp,
                                      neu1: neu1p, neu1e: neu1ep)
                        } else {
                            trainSkipGram(word: word, b: b, window: window, negative: negative,
                                          sentencePosition: sentencePosition, sentenceLength: sentenceLength,
                                          sen: sen, vocabSize: vocabSize, alpha: Float(alpha),
                                          unigramTable: unigramTable, nextRandom: &nextRandom,
                                          syn0: syn0p, syn1neg: syn1p, expTable: expp,
                                          neu1e: neu1ep)
                        }
                    }
                }
            }

        }}}}}

        progress?(1.0)

        // Package syn0 (the input word vectors) as the model, exactly like the C's output.
        // ref: word2vec.c lines 574-580 — it writes syn0 rows as the word vectors.
        let words = vocab.map { $0.word }
        return WordEmbeddings(words: words, vectorSize: vectorSize, storage: syn0)
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
