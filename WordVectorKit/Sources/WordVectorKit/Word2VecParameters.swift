import Foundation

/// Training hyperparameters for `Word2Vec`. Names mirror the C reference's globals.
///
/// `Sendable` because instances are handed from the main actor to the off-main training task
/// (see `ModelStore.train(stems:parameters:)`). Every stored property is a value type, so the
/// conformance is sound rather than a suppression; it also mirrors the repo's `CorpusBook`.
public struct Word2VecParameters: Sendable {
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

    /// Resolution of the unigram negative-sampling table (C: `table_size`).
    ///
    /// This is the number of slots the `cn^0.75` distribution is quantized into. A larger
    /// table gives finer sampling granularity but costs `4 * unigramTableSize` bytes (the
    /// table is `Int32`). The C reference hard-codes `1e8` (~400 MB), which was fine for a
    /// desktop but risks an OOM/jetsam kill on an iOS device, so we default LOWER — `1e7`
    /// (10M entries, ~40 MB) — which is still ample resolution for on-device corpora.
    /// Raise it toward `1e8` to match the reference exactly, or lower it further for tests.
    /// Values are clamped to `>= 1` at build time to avoid a divide-by-zero in sampling.
    public var unigramTableSize: Int = Int(1e7)

    public init() {}
}
