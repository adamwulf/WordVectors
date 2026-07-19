import XCTest
@testable import WordVectorKit

final class WordEmbeddingsProjectionTests: XCTestCase {

    func testProjectionCapturesKnownDominantDirection() {
        // Vocabulary sorting preserves this order, while the vectors vary primarily along
        // their first dimension. PCA's first projected coordinate should retain that order.
        let embeddings = WordEmbeddings(dictionary: [
            "a": [-30, 0.2],
            "b": [-10, -0.1],
            "c": [10, 0.1],
            "d": [30, -0.2]
        ])

        let projection = embeddings.projected2D(wordCount: 4)
        let xValues = projection.map { $0.x }
        let yValues = projection.map { $0.y }

        XCTAssertEqual(projection.map { $0.word }, ["a", "b", "c", "d"])
        XCTAssertGreaterThan(variance(of: xValues), variance(of: yValues))
        XCTAssertLessThan(projection[0].x, projection[1].x)
        XCTAssertLessThan(projection[1].x, projection[2].x)
        XCTAssertLessThan(projection[2].x, projection[3].x)
    }

    func testEmptyModelAndNonPositiveCountsReturnEmpty() {
        let empty = WordEmbeddings(dictionary: [:])
        XCTAssertTrue(empty.projected2D(wordCount: 10).isEmpty)

        let embeddings = WordEmbeddings(dictionary: ["a": [1, 2]])
        XCTAssertTrue(embeddings.projected2D(wordCount: 0).isEmpty)
        XCTAssertTrue(embeddings.projected2D(wordCount: -1).isEmpty)
    }

    func testCountLargerThanVocabularyReturnsEveryWordInOrder() {
        let embeddings = WordEmbeddings(dictionary: [
            "c": [3, 0],
            "a": [1, 0],
            "b": [2, 0]
        ])

        let projection = embeddings.projected2D(wordCount: 100)

        XCTAssertEqual(projection.count, embeddings.vocabulary.count)
        XCTAssertEqual(projection.map { $0.word }, embeddings.vocabulary)
    }

    func testOneAndTwoWordModelsDoNotCrash() {
        let oneWord = WordEmbeddings(dictionary: ["only": [4, 8]])
        let oneWordProjection = oneWord.projected2D(wordCount: 1)
        XCTAssertEqual(oneWordProjection.count, 1)
        XCTAssertEqual(oneWordProjection[0].x, 0)
        XCTAssertEqual(oneWordProjection[0].y, 0)

        let twoWords = WordEmbeddings(dictionary: [
            "a": [-2, 1],
            "b": [2, 1]
        ])
        let twoWordProjection = twoWords.projected2D(wordCount: 2)
        XCTAssertEqual(twoWordProjection.count, 2)
        XCTAssertTrue(twoWordProjection.allSatisfy { $0.x.isFinite && $0.y.isFinite })
        XCTAssertEqual(twoWordProjection[0].y, 0)
        XCTAssertEqual(twoWordProjection[1].y, 0)
    }

    func testOneDimensionalVectorsUseZeroForSecondCoordinate() {
        let embeddings = WordEmbeddings(dictionary: [
            "a": [-3],
            "b": [0],
            "c": [3]
        ])

        let projection = embeddings.projected2D(wordCount: 3)

        XCTAssertEqual(projection.count, 3)
        XCTAssertLessThan(projection[0].x, projection[1].x)
        XCTAssertLessThan(projection[1].x, projection[2].x)
        XCTAssertTrue(projection.allSatisfy { $0.y == 0 })
    }

    func testAllZeroVectorsReturnZeroCoordinates() {
        let embeddings = WordEmbeddings(dictionary: [
            "a": [0, 0, 0],
            "b": [0, 0, 0],
            "c": [0, 0, 0]
        ])

        let projection = embeddings.projected2D(wordCount: 3)

        XCTAssertEqual(projection.count, 3)
        XCTAssertTrue(projection.allSatisfy { $0.x == 0 && $0.y == 0 })
    }

    func testProjectionIsDeterministic() {
        let embeddings = WordEmbeddings(dictionary: [
            "a": [-4, 2, 1],
            "b": [-1, -1, 0],
            "c": [2, 1, -1],
            "d": [5, -2, 2]
        ])

        let first = embeddings.projected2D(wordCount: 4)
        let second = embeddings.projected2D(wordCount: 4)

        XCTAssertEqual(first.count, second.count)
        for index in first.indices {
            XCTAssertEqual(first[index].word, second[index].word)
            XCTAssertEqual(first[index].x, second[index].x)
            XCTAssertEqual(first[index].y, second[index].y)
        }
    }

    private func variance(of values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let mean = values.reduce(0, +) / Float(values.count)
        return values.reduce(0) { sum, value in
            let difference = value - mean
            return sum + difference * difference
        } / Float(values.count)
    }
}
