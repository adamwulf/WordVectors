import XCTest
@testable import WordVectorKit

final class WordEmbeddingsPersistenceTests: XCTestCase {

    private static let fastTableSize = 100_000

    private func tempURL() -> URL {
        // Unique temp file per test to avoid cross-test interference.
        let dir = FileManager.default.temporaryDirectory
        return dir.appendingPathComponent("wvk-\(UUID().uuidString).bin")
    }

    // MARK: - Round-trip exactness

    func testSaveAndLoadRoundTripsExactly() throws {
        let original = WordEmbeddings(dictionary: [
            "alpha": [0.1, -0.2, 3.5, 0],
            "beta":  [1e-8, -1e8, 0.333333, 42],
            "gamma": [-0.0, 0.0, .leastNonzeroMagnitude, 1]
        ])

        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try original.save(to: url)
        let reloaded = try WordEmbeddings(contentsOf: url)

        XCTAssertEqual(reloaded.vectorSize, original.vectorSize)
        XCTAssertEqual(Set(reloaded.vocabulary), Set(original.vocabulary))
        for word in original.vocabulary {
            // Exact bit-for-bit equality of every stored float.
            XCTAssertEqual(reloaded.vector(for: word), original.vector(for: word),
                           "vector mismatch for \(word)")
        }
    }

    func testInMemorySerializationRoundTrips() throws {
        let original = WordEmbeddings(dictionary: [
            "x": [1, 2, 3],
            "y": [4, 5, 6]
        ])
        let data = original.serialized()
        let reloaded = try WordEmbeddings(serialized: data)
        XCTAssertEqual(reloaded.vector(for: "x"), [1, 2, 3])
        XCTAssertEqual(reloaded.vector(for: "y"), [4, 5, 6])
    }

    func testTrainedModelSurvivesRoundTrip() throws {
        var params = Word2VecParameters()
        params.vectorSize = 12
        params.iterations = 10
        params.minCount = 1
        params.window = 3
        params.seed = 99
        params.unigramTableSize = Self.fastTableSize

        var sentences: [String] = []
        for _ in 0..<20 {
            sentences.append("the cat sat on the mat")
            sentences.append("the dog ran on the log")
        }
        let trained = Word2Vec(parameters: params).train(sentences: sentences, progress: nil)

        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try trained.save(to: url)
        let reloaded = try WordEmbeddings(contentsOf: url)

        XCTAssertEqual(Set(reloaded.vocabulary), Set(trained.vocabulary))
        for word in trained.vocabulary {
            XCTAssertEqual(reloaded.vector(for: word), trained.vector(for: word))
        }
        // Nearest-neighbor queries must behave identically after reload.
        XCTAssertEqual(reloaded.nearest(to: "cat", count: 3).map { $0.word },
                       trained.nearest(to: "cat", count: 3).map { $0.word })
    }

    // MARK: - Unicode words

    func testRoundTripsUnicodeWords() throws {
        let original = WordEmbeddings(dictionary: [
            "café": [1, 0],
            "naïve": [0, 1],
            "日本語": [1, 1]
        ])
        let data = original.serialized()
        let reloaded = try WordEmbeddings(serialized: data)
        XCTAssertEqual(Set(reloaded.vocabulary), Set(original.vocabulary))
        XCTAssertEqual(reloaded.vector(for: "日本語"), [1, 1])
    }

    // MARK: - Error handling

    func testBadMagicThrows() {
        let garbage = Data([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        XCTAssertThrowsError(try WordEmbeddings(serialized: garbage)) { error in
            XCTAssertEqual(error as? WordEmbeddingsIOError, .badMagic)
        }
    }

    func testTruncatedDataThrows() throws {
        let original = WordEmbeddings(dictionary: ["a": [1, 2, 3]])
        let full = original.serialized()
        let truncated = full.prefix(full.count - 4) // drop the last float
        XCTAssertThrowsError(try WordEmbeddings(serialized: Data(truncated))) { error in
            XCTAssertEqual(error as? WordEmbeddingsIOError, .truncated)
        }
    }

    func testEmptyDataThrows() {
        XCTAssertThrowsError(try WordEmbeddings(serialized: Data())) { error in
            XCTAssertEqual(error as? WordEmbeddingsIOError, .truncated)
        }
    }
}
