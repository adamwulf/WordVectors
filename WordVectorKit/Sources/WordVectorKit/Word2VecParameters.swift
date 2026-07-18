import Foundation

/// Training hyperparameters for `Word2Vec`. Names mirror the C reference's globals.
public struct Word2VecParameters {
    /// Dimensionality of the word vectors (C: `layer1_size`).
    public var vectorSize: Int = 100
    /// Maximum skip length between words — the context window (C: `window`).
    public var window: Int = 5
    /// Discard words appearing fewer than this many times (C: `min_count`).
    public var minCount: Int = 5
    /// Number of negative samples per positive example (C: `negative`).
    public var negativeSamples: Int = 5
    /// Training epochs (C: `iter`).
    public var iterations: Int = 5
    /// Subsampling threshold for frequent words (C: `sample`).
    public var subsample: Double = 1e-3
    /// Starting learning rate (C: `alpha`). 0.025 for skip-gram, 0.05 for CBOW.
    public var initialAlpha: Double = 0.025
    /// `false` = skip-gram (better small-corpus quality), `true` = CBOW (C: `cbow`).
    public var useCBOW: Bool = false
    /// Seed for the linear-congruential RNG so results are reproducible.
    public var seed: UInt64 = 1

    /// Resolution of the unigram negative-sampling table (C: `table_size`, fixed at 1e8).
    ///
    /// This is the number of slots the `cn^0.75` distribution is quantized into. The C
    /// reference hard-codes `1e8`, and that is the default here for faithful numerical
    /// behavior. It is exposed (internal) only so tests can use a smaller resolution for
    /// speed — a smaller table still encodes the same distribution, just at coarser
    /// quantization, so it does not change the algorithm, only the sampling granularity.
    internal var unigramTableSize: Int = Int(1e8)

    public init() {}
}
