import XCTest
@testable import WordVectorKit

final class WordEmbeddingsTests: XCTestCase {

    // MARK: - Basic accessors

    func testVectorLookupAndContains() {
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0, 0],
            "b": [0, 1, 0]
        ])
        XCTAssertEqual(emb.vectorSize, 3)
        XCTAssertTrue(emb.contains("a"))
        XCTAssertFalse(emb.contains("z"))
        XCTAssertEqual(emb.vector(for: "a"), [1, 0, 0])
        XCTAssertNil(emb.vector(for: "z"))
        XCTAssertEqual(Set(emb.vocabulary), ["a", "b"])
    }

    func testVectorIsAnImmutableCopy() {
        let emb = WordEmbeddings(dictionary: ["a": [1, 2, 3]])
        var v = emb.vector(for: "a")!
        v[0] = 999
        // Mutating the returned copy must not change the stored vector.
        XCTAssertEqual(emb.vector(for: "a"), [1, 2, 3])
    }

    // MARK: - Cosine / nearest

    func testNearestOrdersByCosineSimilarity() {
        // "a" points along x. "b" is closest in angle, "c" orthogonal, "d" opposite.
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0],
            "b": [1, 0.1],   // very close angle to a
            "c": [0, 1],     // orthogonal (cosine ~0)
            "d": [-1, 0]     // opposite (cosine -1)
        ])
        let neighbors = emb.nearest(to: "a", count: 3)
        // Query itself excluded; order should be b, c, d.
        XCTAssertEqual(neighbors.map { $0.word }, ["b", "c", "d"])
        // b should have highest similarity, d the lowest.
        XCTAssertGreaterThan(neighbors[0].similarity, neighbors[1].similarity)
        XCTAssertGreaterThan(neighbors[1].similarity, neighbors[2].similarity)
        // Cosine to the opposite vector is -1.
        XCTAssertEqual(neighbors[2].similarity, -1, accuracy: 1e-5)
        // Cosine to the orthogonal vector is ~0.
        XCTAssertEqual(neighbors[1].similarity, 0, accuracy: 1e-5)
    }

    func testNearestExcludesQueryItself() {
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0],
            "b": [0, 1]
        ])
        let neighbors = emb.nearest(to: "a", count: 5)
        XCTAssertFalse(neighbors.contains { $0.word == "a" })
    }

    func testNearestOOVReturnsEmpty() {
        let emb = WordEmbeddings(dictionary: ["a": [1, 0]])
        XCTAssertTrue(emb.nearest(to: "missing", count: 3).isEmpty)
    }

    func testNearestToArbitraryVectorWithExclusions() {
        let emb = WordEmbeddings(dictionary: [
            "x": [1, 0, 0],
            "y": [0, 1, 0],
            "z": [0, 0, 1]
        ])
        // Query vector aligned with x, but exclude x -> should not appear.
        let neighbors = emb.nearest(to: [1, 0, 0], count: 3, excluding: ["x"])
        XCTAssertFalse(neighbors.contains { $0.word == "x" })
        // y and z are both orthogonal (~0 similarity); tie-break is alphabetical.
        XCTAssertEqual(neighbors.map { $0.word }, ["y", "z"])
    }

    func testCosineSimilarityValueIsCorrect() {
        // Two vectors at 45 degrees: cosine = 1/sqrt(2) ~ 0.7071.
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0],
            "b": [1, 1]
        ])
        let neighbors = emb.nearest(to: "a", count: 1)
        XCTAssertEqual(neighbors.count, 1)
        XCTAssertEqual(neighbors[0].similarity, 1 / 2.0.squareRoot().magnitude.float, accuracy: 1e-5)
    }

    // MARK: - Distance metrics

    func testDotProductRewardsMagnitude() {
        // All three candidates point the same direction as the query, so cosine ties them,
        // but dot product should rank the largest-magnitude vector first.
        let emb = WordEmbeddings(dictionary: [
            "q":     [1, 0],
            "small": [1, 0],
            "mid":   [3, 0],
            "big":   [9, 0]
        ])
        let neighbors = emb.nearest(to: "q", count: 3, metric: .dotProduct)
        XCTAssertEqual(neighbors.map { $0.word }, ["big", "mid", "small"])
        // Dot product with q = [1,0] is just the candidate's x component.
        XCTAssertEqual(neighbors[0].score, 9, accuracy: 1e-5)
        XCTAssertEqual(neighbors[2].score, 1, accuracy: 1e-5)
    }

    func testEuclideanOrdersByAscendingDistanceBestFirst() {
        // Distances from q=[0,0]: near=1, mid=2, far=5. Best (smallest) must come first.
        let emb = WordEmbeddings(dictionary: [
            "q":    [0, 0],
            "near": [1, 0],
            "mid":  [0, 2],
            "far":  [3, 4]   // distance 5
        ])
        let neighbors = emb.nearest(to: "q", count: 3, metric: .euclidean)
        XCTAssertEqual(neighbors.map { $0.word }, ["near", "mid", "far"])
        XCTAssertEqual(neighbors[0].score, 1, accuracy: 1e-5)
        XCTAssertEqual(neighbors[1].score, 2, accuracy: 1e-5)
        XCTAssertEqual(neighbors[2].score, 5, accuracy: 1e-5)
    }

    func testCosineMetricMatchesLegacyNearest() {
        // The metric-aware cosine path must agree with the original similarity-based API.
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0],
            "b": [1, 0.1],
            "c": [0, 1],
            "d": [-1, 0]
        ])
        let legacy = emb.nearest(to: "a", count: 3)
        let metric = emb.nearest(to: "a", count: 3, metric: .cosine)
        XCTAssertEqual(legacy.map { $0.word }, metric.map { $0.word })
        for (l, m) in zip(legacy, metric) {
            XCTAssertEqual(l.similarity, m.score, accuracy: 1e-6)
        }
    }

    func testAnalogyWithMetricExcludesInputs() {
        // Same grid as the cosine analogy test, but scored by Euclidean distance: the exact
        // match still wins (distance 0) and the three inputs are still excluded.
        let emb = WordEmbeddings(dictionary: [
            "man":   [1, 0],
            "woman": [1, 1],
            "king":  [5, 0],
            "queen": [5, 1],
            "noise": [-3, -3]
        ])
        let result = emb.analogy(base: "king", minus: "man", plus: "woman", count: 3, metric: .euclidean)
        XCTAssertFalse(result.contains { ["king", "man", "woman"].contains($0.word) })
        XCTAssertEqual(result.first?.word, "queen")
        XCTAssertEqual(result.first?.score ?? -1, 0, accuracy: 1e-5)
    }

    func testMetricDisplayMetadata() {
        XCTAssertEqual(DistanceMetric.allCases, [.cosine, .dotProduct, .euclidean])
        XCTAssertTrue(DistanceMetric.cosine.isHigherBetter)
        XCTAssertTrue(DistanceMetric.dotProduct.isHigherBetter)
        XCTAssertFalse(DistanceMetric.euclidean.isHigherBetter)
        XCTAssertEqual(DistanceMetric.cosine.scoreColumnTitle, "Score")
        XCTAssertEqual(DistanceMetric.euclidean.scoreColumnTitle, "Distance")
    }

    // MARK: - Analogy (word algebra)

    func testAnalogyReturnsUnambiguousAnswer() {
        // Construct a clean gender/royalty grid where king - man + woman == queen exactly.
        //   man   = [1, 0]
        //   woman = [1, 1]
        //   king  = [5, 0]
        //   queen = [5, 1]
        // king - man + woman = [5,0] - [1,0] + [1,1] = [5, 1] == queen.
        let emb = WordEmbeddings(dictionary: [
            "man":   [1, 0],
            "woman": [1, 1],
            "king":  [5, 0],
            "queen": [5, 1],
            "noise": [-3, -3]
        ])
        let result = emb.analogy(base: "king", minus: "man", plus: "woman", count: 1)
        XCTAssertEqual(result.first?.word, "queen")
        // Exact match -> cosine similarity 1.
        XCTAssertEqual(result.first?.similarity ?? 0, 1, accuracy: 1e-5)
    }

    func testAnalogyExcludesTheThreeInputs() {
        // Even if an input word is the closest to the computed vector, it must be excluded.
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0],
            "b": [0, 1],
            "c": [1, 0],  // identical to a
            "d": [2, 0]   // the intended answer direction
        ])
        // a - b + c = [1,0] - [0,1] + [1,0] = [2, -1]; nearest excluding a,b,c should be d.
        let result = emb.analogy(base: "a", minus: "b", plus: "c", count: 3)
        XCTAssertFalse(result.contains { ["a", "b", "c"].contains($0.word) })
        XCTAssertEqual(result.first?.word, "d")
    }

    func testAnalogyOOVReturnsEmpty() {
        let emb = WordEmbeddings(dictionary: [
            "a": [1, 0],
            "b": [0, 1]
        ])
        XCTAssertTrue(emb.analogy(base: "a", minus: "b", plus: "missing", count: 3).isEmpty)
    }
}

private extension Double {
    /// Small helper to cast a computed Double constant to Float for accuracy comparisons.
    var float: Float { Float(self) }
}
