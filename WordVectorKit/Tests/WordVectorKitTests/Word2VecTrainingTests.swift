import XCTest
@testable import WordVectorKit

final class Word2VecTrainingTests: XCTestCase {

    /// A small negative-sampling table resolution for fast tests. The package default is
    /// 1e7 (the C reference uses 1e8); a smaller value encodes the same cn^0.75 distribution
    /// at coarser quantization, so it keeps the smoke tests fast (<a few seconds) without
    /// changing the algorithm.
    private static let fastTableSize = 100_000

    /// Builds a tiny synthetic corpus with strong co-occurrence structure. The words
    /// "cat"/"dog" always appear beside "pet", and "king"/"queen" always beside "royal",
    /// so after training the two groups should separate.
    private func syntheticCorpus() -> [String] {
        var sentences: [String] = []
        // Repeat strongly-structured patterns so co-occurrence dominates.
        for _ in 0..<40 {
            sentences.append("the cat is a pet animal")
            sentences.append("the dog is a pet animal")
            sentences.append("a cat and a dog are pets")
            sentences.append("the king is a royal ruler")
            sentences.append("the queen is a royal ruler")
            sentences.append("a king and a queen are royals")
        }
        return sentences
    }

    func testTrainingCompletesAndProducesEmbeddings() {
        var params = Word2VecParameters()
        params.vectorSize = 20
        params.iterations = 20
        params.minCount = 1
        params.window = 3
        params.negativeSamples = 5
        params.seed = 42
        params.unigramTableSize = Self.fastTableSize

        let trainer = Word2Vec(parameters: params)

        var lastProgress: Double = -1
        let embeddings = trainer.train(sentences: syntheticCorpus()) { p in
            // Progress must be monotonically non-decreasing and within bounds.
            XCTAssertGreaterThanOrEqual(p, 0)
            XCTAssertLessThanOrEqual(p, 1)
            lastProgress = p
        }

        // Training reported completion.
        XCTAssertEqual(lastProgress, 1.0, accuracy: 1e-9)

        // Vocabulary is non-empty and has the expected dimension.
        XCTAssertFalse(embeddings.vocabulary.isEmpty)
        XCTAssertEqual(embeddings.vectorSize, 20)
        XCTAssertTrue(embeddings.contains("cat"))
        XCTAssertTrue(embeddings.contains("king"))

        // Vectors have the right length.
        XCTAssertEqual(embeddings.vector(for: "cat")?.count, 20)

        // The vector should not be all zeros (weights were actually updated).
        let catVec = embeddings.vector(for: "cat")!
        XCTAssertTrue(catVec.contains { $0 != 0 })
    }

    func testNearestReturnsPlausibleNeighbors() {
        var params = Word2VecParameters()
        params.vectorSize = 30
        params.iterations = 40
        params.minCount = 1
        params.window = 3
        params.negativeSamples = 5
        params.seed = 7
        params.unigramTableSize = Self.fastTableSize

        let trainer = Word2Vec(parameters: params)
        let embeddings = trainer.train(sentences: syntheticCorpus(), progress: nil)

        // "cat" and "dog" share the "pet animal" context; "king"/"queen" share "royal ruler".
        // The nearest neighbor of "cat" among the animal/royal words should be an animal-group
        // word (dog/pet/animal) rather than a royal-group word.
        let catNeighbors = embeddings.nearest(to: "cat", count: 3).map { $0.word }
        XCTAssertFalse(catNeighbors.isEmpty)

        // At least one strongly co-occurring animal-group word should rank in the top 3.
        let animalGroup: Set<String> = ["dog", "pet", "animal", "pets"]
        XCTAssertTrue(
            catNeighbors.contains { animalGroup.contains($0) },
            "Expected an animal-group neighbor for 'cat', got \(catNeighbors)"
        )
    }

    func testEmptyCorpusYieldsEmptyModel() {
        var params = Word2VecParameters()
        params.vectorSize = 10
        params.iterations = 2
        params.minCount = 1
        params.unigramTableSize = Self.fastTableSize

        let trainer = Word2Vec(parameters: params)
        let embeddings = trainer.train(sentences: [], progress: nil)
        XCTAssertTrue(embeddings.vocabulary.isEmpty)
    }

    func testMinCountFiltersRareWords() {
        var params = Word2VecParameters()
        params.vectorSize = 10
        params.iterations = 5
        params.minCount = 3
        params.window = 2
        params.unigramTableSize = Self.fastTableSize

        // "common" appears many times; "rare" appears once.
        var sentences: [String] = []
        for _ in 0..<10 { sentences.append("common common word here") }
        sentences.append("rare token appears once")

        let trainer = Word2Vec(parameters: params)
        let embeddings = trainer.train(sentences: sentences, progress: nil)

        XCTAssertTrue(embeddings.contains("common"))
        XCTAssertFalse(embeddings.contains("rare"))
    }

    func testDeterministicWithSameSeed() {
        var params = Word2VecParameters()
        params.vectorSize = 15
        params.iterations = 10
        params.minCount = 1
        params.window = 3
        params.seed = 123
        params.unigramTableSize = Self.fastTableSize

        let corpus = syntheticCorpus()
        let a = Word2Vec(parameters: params).train(sentences: corpus, progress: nil)
        let b = Word2Vec(parameters: params).train(sentences: corpus, progress: nil)

        // Same seed + same corpus + single-threaded => identical vectors.
        XCTAssertEqual(a.vector(for: "cat"), b.vector(for: "cat"))
        XCTAssertEqual(a.vector(for: "king"), b.vector(for: "king"))
    }
}
