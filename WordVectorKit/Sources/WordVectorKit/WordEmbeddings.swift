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
        guard let row = index[word] else { return [] }
        let start = row * vectorSize
        let query = Array(storage[start..<(start + vectorSize)])
        return nearest(to: query, count: count, excluding: [word])
    }

    /// N nearest words to an arbitrary `vector` by cosine similarity.
    /// Words in `excluding` are omitted from the result (used for word algebra).
    public func nearest(to vector: [Float], count: Int, excluding: Set<String>) -> [(word: String, similarity: Float)] {
        guard count > 0, vector.count == vectorSize, !words.isEmpty else { return [] }

        // Norm of the query vector.
        var queryNorm: Float = 0
        vector.withUnsafeBufferPointer { qb in
            vDSP_svesq(qb.baseAddress!, 1, &queryNorm, vDSP_Length(vectorSize))
        }
        queryNorm = queryNorm.squareRoot()
        guard queryNorm > 0 else { return [] }

        var scored: [(word: String, similarity: Float)] = []
        scored.reserveCapacity(words.count)

        vector.withUnsafeBufferPointer { qb in
            storage.withUnsafeBufferPointer { sb in
                for row in 0..<words.count {
                    let w = words[row]
                    if excluding.contains(w) { continue }
                    let rowNorm = norms[row]
                    if rowNorm == 0 { continue } // zero vector has undefined cosine; skip.

                    var dot: Float = 0
                    let base = sb.baseAddress! + row * vectorSize
                    vDSP_dotpr(qb.baseAddress!, 1, base, 1, &dot, vDSP_Length(vectorSize))
                    let cosine = dot / (queryNorm * rowNorm)
                    scored.append((word: w, similarity: cosine))
                }
            }
        }

        // Sort descending by similarity; break ties on the word for determinism.
        scored.sort { a, b in
            if a.similarity != b.similarity { return a.similarity > b.similarity }
            return a.word < b.word
        }

        if scored.count > count {
            return Array(scored[0..<count])
        }
        return scored
    }

    /// Word algebra: `base - minus + plus`, then the nearest words to the result,
    /// excluding the three input words. Returns empty if any input is out of vocabulary.
    public func analogy(base: String, minus: String, plus: String, count: Int) -> [(word: String, similarity: Float)] {
        guard let vb = vector(for: base),
              let vm = vector(for: minus),
              let vp = vector(for: plus) else {
            return []
        }
        var result = [Float](repeating: 0, count: vectorSize)
        for i in 0..<vectorSize {
            result[i] = vb[i] - vm[i] + vp[i]
        }
        return nearest(to: result, count: count, excluding: [base, minus, plus])
    }
}
