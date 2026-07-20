import XCTest
@testable import WordVectorKit

final class WordEmbeddingsClusteringTests: XCTestCase {

    // MARK: - Determinism

    func testSameSeedGivesIdenticalClustering() {
        let embeddings = makeBlobEmbeddings()

        let first = embeddings.cluster(k: 3, wordCount: 100, labelsPerCluster: 4, seed: 42)
        let second = embeddings.cluster(k: 3, wordCount: 100, labelsPerCluster: 4, seed: 42)

        XCTAssertEqual(first.clusterCount, second.clusterCount)
        XCTAssertEqual(first.sizes, second.sizes)
        XCTAssertEqual(first.representatives, second.representatives)

        XCTAssertEqual(first.assignments.count, second.assignments.count)
        for index in first.assignments.indices {
            XCTAssertEqual(first.assignments[index].word, second.assignments[index].word)
            XCTAssertEqual(first.assignments[index].cluster, second.assignments[index].cluster)
        }
    }

    func testAssignmentsAreOrderedLikeVocabularyPrefix() {
        let embeddings = makeBlobEmbeddings()
        let wordCount = 20

        let result = embeddings.cluster(k: 3, wordCount: wordCount, labelsPerCluster: 4, seed: 7)

        let expectedWords = Array(embeddings.vocabulary.prefix(wordCount))
        XCTAssertEqual(result.assignments.map { $0.word }, expectedWords)
    }

    // MARK: - Recovery of well-separated blobs

    func testWellSeparatedBlobsAreRecovered() {
        // Three tight blobs placed far apart in 8-D. Every point in a blob must land in the
        // same cluster, and the three clusters must be distinct. Because cluster *labels* are
        // arbitrary, we assert on the induced partition, not on specific label values.
        let (embeddings, blobOf) = makeSeparatedBlobs(
            dimension: 8,
            blobCount: 3,
            pointsPerBlob: 15,
            separation: 50,
            jitter: 0.05
        )

        let result = embeddings.cluster(k: 3, wordCount: 1000, labelsPerCluster: 3, seed: 123)

        XCTAssertEqual(result.clusterCount, 3)

        // Map each word to its assigned cluster, then verify every blob is internally uniform.
        var clusterForWord: [String: Int] = [:]
        for assignment in result.assignments {
            clusterForWord[assignment.word] = assignment.cluster
        }

        var clusterForBlob: [Int: Int] = [:]
        var blobForCluster: [Int: Int] = [:]
        for assignment in result.assignments {
            let blob = blobOf(assignment.word)
            let cluster = assignment.cluster
            if let existing = clusterForBlob[blob] {
                XCTAssertEqual(existing, cluster, "blob \(blob) split across clusters")
            } else {
                clusterForBlob[blob] = cluster
            }
            if let existingBlob = blobForCluster[cluster] {
                XCTAssertEqual(existingBlob, blob, "cluster \(cluster) mixed two blobs")
            } else {
                blobForCluster[cluster] = blob
            }
        }

        // Three blobs → three distinct clusters, one-to-one.
        XCTAssertEqual(Set(clusterForBlob.values).count, 3)
    }

    // MARK: - k clamping

    func testKGreaterThanWordCountClampsToWordCount() {
        let embeddings = makeBlobEmbeddings()
        let wordCount = 5

        let result = embeddings.cluster(k: 50, wordCount: wordCount, labelsPerCluster: 2, seed: 1)

        // k is clamped to wordCount, and with distinct points each word is its own cluster.
        XCTAssertEqual(result.clusterCount, wordCount)
        XCTAssertEqual(result.sizes.reduce(0, +), wordCount)
        XCTAssertTrue(result.sizes.allSatisfy { $0 == 1 })
    }

    func testKLessThanOneClampsToOne() {
        let embeddings = makeBlobEmbeddings()
        let wordCount = 12

        let zeroK = embeddings.cluster(k: 0, wordCount: wordCount, labelsPerCluster: 3, seed: 9)
        XCTAssertEqual(zeroK.clusterCount, 1)
        XCTAssertEqual(zeroK.sizes, [wordCount])
        XCTAssertTrue(zeroK.assignments.allSatisfy { $0.cluster == 0 })

        let negativeK = embeddings.cluster(k: -5, wordCount: wordCount, labelsPerCluster: 3, seed: 9)
        XCTAssertEqual(negativeK.clusterCount, 1)
        XCTAssertEqual(negativeK.sizes, [wordCount])
    }

    // MARK: - Internal consistency

    func testResultShapeIsInternallyConsistent() {
        let embeddings = makeBlobEmbeddings()
        let wordCount = 40
        let labelsPerCluster = 5

        let result = embeddings.cluster(
            k: 4,
            wordCount: wordCount,
            labelsPerCluster: labelsPerCluster,
            seed: 2024
        )

        // Sizes sum to the number of clustered words.
        XCTAssertEqual(result.sizes.reduce(0, +), wordCount)
        XCTAssertEqual(result.sizes.count, result.clusterCount)

        // One representative list per cluster, each non-empty and capped at labelsPerCluster.
        XCTAssertEqual(result.representatives.count, result.clusterCount)
        for cluster in 0..<result.clusterCount {
            XCTAssertFalse(result.representatives[cluster].isEmpty,
                           "cluster \(cluster) has no representatives")
            XCTAssertLessThanOrEqual(result.representatives[cluster].count, labelsPerCluster)
        }

        // Every assignment names a real word and a valid cluster index.
        let vocabularySet = Set(embeddings.vocabulary.prefix(wordCount))
        for assignment in result.assignments {
            XCTAssertTrue(vocabularySet.contains(assignment.word))
            XCTAssertTrue((0..<result.clusterCount).contains(assignment.cluster))
        }

        // Representatives are drawn from the words actually in that cluster.
        var wordsInCluster = [Set<String>](repeating: [], count: result.clusterCount)
        for assignment in result.assignments {
            wordsInCluster[assignment.cluster].insert(assignment.word)
        }
        for cluster in 0..<result.clusterCount {
            for word in result.representatives[cluster] {
                XCTAssertTrue(wordsInCluster[cluster].contains(word),
                              "representative \(word) is not a member of cluster \(cluster)")
            }
        }

        // Every cluster is non-empty: with the re-seeding strategy, no returned cluster is empty.
        XCTAssertTrue(result.sizes.allSatisfy { $0 >= 1 })
    }

    // MARK: - Degenerate vocabularies

    func testEmptyVocabularyReturnsEmptyResult() {
        let empty = WordEmbeddings(dictionary: [:])

        let result = empty.cluster(k: 5, wordCount: 10, labelsPerCluster: 3, seed: 1)

        XCTAssertTrue(result.assignments.isEmpty)
        XCTAssertEqual(result.clusterCount, 0)
        XCTAssertTrue(result.representatives.isEmpty)
        XCTAssertTrue(result.sizes.isEmpty)
    }

    func testNonPositiveWordCountReturnsEmptyResult() {
        let embeddings = makeBlobEmbeddings()

        for wordCount in [0, -1, -100] {
            let result = embeddings.cluster(k: 3, wordCount: wordCount, labelsPerCluster: 3, seed: 5)
            XCTAssertTrue(result.assignments.isEmpty)
            XCTAssertEqual(result.clusterCount, 0)
            XCTAssertTrue(result.representatives.isEmpty)
            XCTAssertTrue(result.sizes.isEmpty)
        }
    }

    func testSingleWordVocabularyDoesNotTrap() {
        let embeddings = WordEmbeddings(dictionary: ["only": [1, 2, 3, 4]])

        let result = embeddings.cluster(k: 4, wordCount: 10, labelsPerCluster: 3, seed: 3)

        XCTAssertEqual(result.clusterCount, 1)
        XCTAssertEqual(result.assignments.count, 1)
        XCTAssertEqual(result.assignments.first?.word, "only")
        XCTAssertEqual(result.assignments.first?.cluster, 0)
        XCTAssertEqual(result.sizes, [1])
        XCTAssertEqual(result.representatives.count, 1)
        XCTAssertEqual(result.representatives.first, ["only"])
    }

    func testZeroVectorsDoNotTrap() {
        // All-zero vectors normalize to the origin; the method must stay total (no divide-by-zero,
        // no crash) and still return a well-formed result.
        let embeddings = WordEmbeddings(dictionary: [
            "a": [0, 0, 0],
            "b": [0, 0, 0],
            "c": [0, 0, 0],
            "d": [0, 0, 0]
        ])

        let result = embeddings.cluster(k: 2, wordCount: 4, labelsPerCluster: 2, seed: 11)

        XCTAssertEqual(result.sizes.reduce(0, +), 4)
        XCTAssertEqual(result.representatives.count, result.clusterCount)
        XCTAssertTrue(result.representatives.allSatisfy { !$0.isEmpty })
    }

    func testLabelsPerClusterZeroYieldsEmptyRepresentativeLists() {
        // A caller may ask for zero labels; the result should be well-formed with empty lists,
        // never a trap.
        let embeddings = makeBlobEmbeddings()

        let result = embeddings.cluster(k: 3, wordCount: 30, labelsPerCluster: 0, seed: 6)

        XCTAssertEqual(result.representatives.count, result.clusterCount)
        XCTAssertTrue(result.representatives.allSatisfy { $0.isEmpty })
        XCTAssertEqual(result.sizes.reduce(0, +), 30)
    }

    // MARK: - Re-seed and zero-total edge paths

    func testEmptyClusterReseedProducesWellFormedResult() {
        // Many identical points along one direction plus a couple of outliers, with k far larger
        // than the number of distinct directions. k-means++ and Lloyd's will repeatedly empty
        // clusters here, so the re-seed path is exercised hard. The result must still be total and
        // well-formed: every returned cluster non-empty, sizes summing to wordCount, and one
        // non-empty representative list per cluster.
        var dictionary: [String: [Float]] = [:]
        // 30 coincident points all pointing the same way.
        for index in 0..<30 {
            dictionary["same\(index)"] = [1, 0, 0, 0]
        }
        // Two lone outliers along other axes.
        dictionary["out_a"] = [0, 1, 0, 0]
        dictionary["out_b"] = [0, 0, 1, 0]

        let embeddings = WordEmbeddings(dictionary: dictionary)
        let wordCount = dictionary.count

        // k = 8 vastly exceeds the 3 distinct directions, forcing repeated empty-cluster re-seeds.
        let result = embeddings.cluster(k: 8, wordCount: wordCount, labelsPerCluster: 3, seed: 99)

        XCTAssertEqual(result.sizes.reduce(0, +), wordCount)
        XCTAssertEqual(result.sizes.count, result.clusterCount)
        XCTAssertTrue(result.sizes.allSatisfy { $0 >= 1 }, "a returned cluster was empty")

        XCTAssertEqual(result.representatives.count, result.clusterCount)
        XCTAssertTrue(result.representatives.allSatisfy { !$0.isEmpty },
                      "a cluster produced no representatives")

        // Determinism: the same seed reproduces the same partition even through the re-seed path.
        let again = embeddings.cluster(k: 8, wordCount: wordCount, labelsPerCluster: 3, seed: 99)
        XCTAssertEqual(result.sizes, again.sizes)
        XCTAssertEqual(result.representatives, again.representatives)
        for index in result.assignments.indices {
            XCTAssertEqual(result.assignments[index].cluster, again.assignments[index].cluster)
        }
    }

    func testAllIdenticalUnitVectorsHitZeroTotalPathWithoutTrapping() {
        // Every vector points the exact same direction, so after the first k-means++ centroid every
        // point's nearest-distance² is 0 and the D²-weighted sampling total is 0. This drives the
        // zero-total fallback branch for k > 1. The method must stay total (no divide-by-zero, no
        // index trap) and return a well-formed result.
        var dictionary: [String: [Float]] = [:]
        for index in 0..<12 {
            dictionary["u\(index)"] = [3, 4, 0] // all the same direction (normalizes to the same unit vector)
        }

        let embeddings = WordEmbeddings(dictionary: dictionary)
        let wordCount = dictionary.count

        let result = embeddings.cluster(k: 5, wordCount: wordCount, labelsPerCluster: 3, seed: 17)

        XCTAssertEqual(result.sizes.reduce(0, +), wordCount)
        XCTAssertEqual(result.sizes.count, result.clusterCount)
        XCTAssertTrue(result.sizes.allSatisfy { $0 >= 1 }, "a returned cluster was empty")

        XCTAssertEqual(result.representatives.count, result.clusterCount)
        XCTAssertTrue(result.representatives.allSatisfy { !$0.isEmpty },
                      "a cluster produced no representatives")

        // Every assignment references a real word and a valid cluster index.
        let vocabularySet = Set(embeddings.vocabulary.prefix(wordCount))
        for assignment in result.assignments {
            XCTAssertTrue(vocabularySet.contains(assignment.word))
            XCTAssertTrue((0..<result.clusterCount).contains(assignment.cluster))
        }

        // Deterministic through the zero-total path too.
        let again = embeddings.cluster(k: 5, wordCount: wordCount, labelsPerCluster: 3, seed: 17)
        XCTAssertEqual(result.sizes, again.sizes)
        XCTAssertEqual(result.representatives, again.representatives)
    }

    // MARK: - Fixtures

    /// A modest deterministic vocabulary: three coarse groups in 6-D so real clustering runs.
    private func makeBlobEmbeddings() -> WordEmbeddings {
        var dictionary: [String: [Float]] = [:]
        var seed: UInt64 = 0xDEADBEEF
        func nextFloat() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let bits = seed >> 40
            return Float(bits) * (1.0 / Float(1 << 24))
        }

        let centers: [[Float]] = [
            [10, 10, 0, 0, 0, 0],
            [0, 0, 10, 10, 0, 0],
            [0, 0, 0, 0, 10, 10]
        ]
        for group in 0..<3 {
            for member in 0..<20 {
                var vector = centers[group]
                for dimension in 0..<vector.count {
                    vector[dimension] += (nextFloat() - 0.5) * 0.5
                }
                dictionary["g\(group)_w\(member)"] = vector
            }
        }
        return WordEmbeddings(dictionary: dictionary)
    }

    /// Builds `blobCount` tight gaussian-ish blobs, well separated along distinct axes so cosine
    /// clustering cleanly recovers them. Returns the embeddings plus a lookup from word to blob.
    private func makeSeparatedBlobs(
        dimension: Int,
        blobCount: Int,
        pointsPerBlob: Int,
        separation: Float,
        jitter: Float
    ) -> (WordEmbeddings, (String) -> Int) {
        precondition(blobCount <= dimension, "each blob needs its own axis for this fixture")

        var dictionary: [String: [Float]] = [:]
        var blobByWord: [String: Int] = [:]
        var seed: UInt64 = 0x1234_5678_9ABC_DEF0
        func nextFloat() -> Float {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let bits = seed >> 40
            return Float(bits) * (1.0 / Float(1 << 24))
        }

        for blob in 0..<blobCount {
            for member in 0..<pointsPerBlob {
                var vector = [Float](repeating: 0, count: dimension)
                // Push this blob far out along its own axis; other axes carry only tiny jitter.
                vector[blob] = separation
                for axis in 0..<dimension {
                    vector[axis] += (nextFloat() - 0.5) * jitter
                }
                let word = "blob\(blob)_pt\(member)"
                dictionary[word] = vector
                blobByWord[word] = blob
            }
        }

        let embeddings = WordEmbeddings(dictionary: dictionary)
        return (embeddings, { word in blobByWord[word] ?? -1 })
    }
}
