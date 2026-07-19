import Foundation
import Accelerate

/// An immutable map of words to their learned vectors, with cosine-similarity queries.
///
/// This type is self-sufficient and platform-portable: it does NOT depend on Apple's
/// NLEmbedding, so it can be unit-tested on macOS and reused anywhere. The app layer may
/// optionally use NLEmbedding for queries later, but the package computes cosine itself.
public final class WordEmbeddings {

    /// Dimensionality of each word vector.
    public let vectorSize: Int

    /// Word at index `i` owns the vector in `storage[i]`.
    private let words: [String]
    private let index: [String: Int]

    /// Row-major flat storage of all vectors: `storage[row * vectorSize + col]`.
    private let storage: [Float]

    /// Precomputed L2 norms per row (0 is stored for zero vectors to avoid division by zero).
    private let norms: [Float]

    // MARK: - Designated initializer

    /// Builds embeddings from an ordered list of words and a matching flat storage array.
    /// `storage` must contain `words.count * vectorSize` elements in row-major order.
    init(words: [String], vectorSize: Int, storage: [Float]) {
        precondition(storage.count == words.count * vectorSize,
                     "storage size must equal words.count * vectorSize")
        self.words = words
        self.vectorSize = vectorSize
        self.storage = storage

        var idx: [String: Int] = [:]
        idx.reserveCapacity(words.count)
        for (i, w) in words.enumerated() {
            idx[w] = i
        }
        self.index = idx

        // Precompute L2 norms so cosine queries don't recompute the query-set norms repeatedly.
        var computedNorms = [Float](repeating: 0, count: words.count)
        storage.withUnsafeBufferPointer { buf in
            for row in 0..<words.count {
                let base = buf.baseAddress! + row * vectorSize
                var norm: Float = 0
                vDSP_svesq(base, 1, &norm, vDSP_Length(vectorSize)) // sum of squares
                computedNorms[row] = norm.squareRoot()
            }
        }
        self.norms = computedNorms
    }

    /// Test/convenience initializer that takes a `[String: [Float]]` dictionary.
    /// The word order is sorted for deterministic iteration; all vectors must share a length.
    public convenience init(dictionary: [String: [Float]]) {
        let sortedWords = dictionary.keys.sorted()
        let dim = sortedWords.first.map { dictionary[$0]!.count } ?? 0
        var flat = [Float]()
        flat.reserveCapacity(sortedWords.count * dim)
        for w in sortedWords {
            let v = dictionary[w]!
            precondition(v.count == dim, "all vectors in dictionary must share the same length")
            flat.append(contentsOf: v)
        }
        self.init(words: sortedWords, vectorSize: dim, storage: flat)
    }

    // MARK: - Public accessors

    /// All words in the vocabulary (order is stable but unspecified beyond that).
    public var vocabulary: [String] {
        return words
    }

    /// The vector for `word`, or `nil` if out of vocabulary. Returns an immutable copy.
    public func vector(for word: String) -> [Float]? {
        guard let row = index[word] else { return nil }
        let start = row * vectorSize
        // Array(slice) copies, so callers can never mutate internal storage.
        return Array(storage[start..<(start + vectorSize)])
    }

    /// Whether `word` is in the vocabulary.
    public func contains(_ word: String) -> Bool {
        return index[word] != nil
    }

    // MARK: - Nearest-neighbor queries

    /// N nearest words to `word` by cosine similarity, excluding the query itself.
    /// Returns an empty array if `word` is out of vocabulary.
    public func nearest(to word: String, count: Int) -> [(word: String, similarity: Float)] {
        nearest(to: word, count: count, metric: .cosine)
            .map { (word: $0.word, similarity: $0.score) }
    }

    /// N nearest words to an arbitrary `vector` by cosine similarity.
    /// Words in `excluding` are omitted from the result (used for word algebra).
    public func nearest(to vector: [Float], count: Int, excluding: Set<String>) -> [(word: String, similarity: Float)] {
        nearest(to: vector, count: count, excluding: excluding, metric: .cosine)
            .map { (word: $0.word, similarity: $0.score) }
    }

    /// Word algebra: `base - minus + plus`, then the nearest words to the result,
    /// excluding the three input words. Returns empty if any input is out of vocabulary.
    public func analogy(base: String, minus: String, plus: String, count: Int) -> [(word: String, similarity: Float)] {
        analogy(base: base, minus: minus, plus: plus, count: count, metric: .cosine)
            .map { (word: $0.word, similarity: $0.score) }
    }

    // MARK: - Nearest-neighbor queries (metric-aware)

    /// N nearest words to `word` under `metric`, excluding the query itself. The `score` is the
    /// metric's natural value (similarity for cosine/dot, distance for Euclidean). Results are
    /// returned best-first per the metric. Empty if `word` is out of vocabulary.
    public func nearest(to word: String, count: Int, metric: DistanceMetric) -> [(word: String, score: Float)] {
        guard let row = index[word] else { return [] }
        let start = row * vectorSize
        let query = Array(storage[start..<(start + vectorSize)])
        return nearest(to: query, count: count, excluding: [word], metric: metric)
    }

    /// N nearest words to an arbitrary `vector` under `metric`. Words in `excluding` are omitted
    /// (used for word algebra). The `score` is the metric's natural value; results are best-first.
    public func nearest(to vector: [Float], count: Int, excluding: Set<String>, metric: DistanceMetric) -> [(word: String, score: Float)] {
        guard count > 0, vector.count == vectorSize, !words.isEmpty else { return [] }

        // Norm of the query vector. Only cosine needs it; skip the work for the other metrics.
        var queryNorm: Float = 0
        if metric == .cosine {
            vector.withUnsafeBufferPointer { qb in
                vDSP_svesq(qb.baseAddress!, 1, &queryNorm, vDSP_Length(vectorSize))
            }
            queryNorm = queryNorm.squareRoot()
            guard queryNorm > 0 else { return [] }
        }

        var scored: [(word: String, score: Float)] = []
        scored.reserveCapacity(words.count)

        vector.withUnsafeBufferPointer { qb in
            storage.withUnsafeBufferPointer { sb in
                for row in 0..<words.count {
                    let w = words[row]
                    if excluding.contains(w) { continue }
                    let base = sb.baseAddress! + row * vectorSize

                    let score: Float
                    switch metric {
                    case .cosine:
                        let rowNorm = norms[row]
                        if rowNorm == 0 { continue } // zero vector has undefined cosine; skip.
                        var dot: Float = 0
                        vDSP_dotpr(qb.baseAddress!, 1, base, 1, &dot, vDSP_Length(vectorSize))
                        score = dot / (queryNorm * rowNorm)
                    case .dotProduct:
                        var dot: Float = 0
                        vDSP_dotpr(qb.baseAddress!, 1, base, 1, &dot, vDSP_Length(vectorSize))
                        score = dot
                    case .euclidean:
                        // vDSP_distancesq gives the squared L2 distance; its square root is the
                        // actual distance. Ordering by squared distance is identical, but the
                        // reported score is the true distance so the UI shows real units.
                        var distanceSquared: Float = 0
                        vDSP_distancesq(qb.baseAddress!, 1, base, 1, &distanceSquared, vDSP_Length(vectorSize))
                        score = distanceSquared.squareRoot()
                    }
                    scored.append((word: w, score: score))
                }
            }
        }

        // Sort best-first per the metric (similarities descending, distances ascending);
        // break ties on the word for determinism.
        let higherIsBetter = metric.isHigherBetter
        scored.sort { a, b in
            if a.score != b.score {
                return higherIsBetter ? a.score > b.score : a.score < b.score
            }
            return a.word < b.word
        }

        if scored.count > count {
            return Array(scored[0..<count])
        }
        return scored
    }

    /// Word algebra under `metric`: `base - minus + plus`, then the nearest words to the result,
    /// excluding the three inputs. Empty if any input is out of vocabulary.
    public func analogy(base: String, minus: String, plus: String, count: Int, metric: DistanceMetric) -> [(word: String, score: Float)] {
        guard let vb = vector(for: base),
              let vm = vector(for: minus),
              let vp = vector(for: plus) else {
            return []
        }
        var result = [Float](repeating: 0, count: vectorSize)
        for i in 0..<vectorSize {
            result[i] = vb[i] - vm[i] + vp[i]
        }
        return nearest(to: result, count: count, excluding: [base, minus, plus], metric: metric)
    }
}
