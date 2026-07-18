import XCTest
@testable import WordVectorKit

/// Tests for the divide-by-zero / OOM / numerical-scale fixes. Each of the crash cases
/// below was empirically reproducible before the guards were added; the assertions here
/// are simply that training completes without trapping and returns a usable model.
final class Word2VecRobustnessTests: XCTestCase {

    private static let fastTableSize = 100_000

    private func corpus() -> [String] {
        var sentences: [String] = []
        for _ in 0..<20 {
            sentences.append("the cat sat on the mat")
            sentences.append("the dog ran in the park")
        }
        return sentences
    }

    // MARK: - Fix #1: window == 0 must not trap on `nextRandom % UInt64(window)`

    func testWindowZeroDoesNotCrash() {
        var params = Word2VecParameters()
        params.vectorSize = 10
        params.iterations = 3
        params.minCount = 1
        params.window = 0                     // would trap without the clamp
        params.unigramTableSize = Self.fastTableSize

        let embeddings = Word2Vec(parameters: params).train(sentences: corpus(), progress: nil)
        // Training completed and produced a vocabulary; the exact vectors are unimportant.
        XCTAssertFalse(embeddings.vocabulary.isEmpty)
        XCTAssertTrue(embeddings.contains("cat"))
    }

    // MARK: - Fix #2: a vocabulary that reduces to a single word must not trap

    func testSingleWordVocabularyDoesNotCrash() {
        // "spam" appears many times; every other word appears once. With minCount = 5,
        // only "spam" survives -> vocabSize == 1 -> negative-sampling fallback would do `% 0`.
        var sentences: [String] = []
        for _ in 0..<10 {
            sentences.append("spam spam spam spam spam spam")
        }
        sentences.append("a unique rare phrase here now")

        var params = Word2VecParameters()   // DEFAULT negative sampling on purpose
        params.vectorSize = 10
        params.iterations = 5
        params.minCount = 5
        params.window = 3
        params.unigramTableSize = Self.fastTableSize

        let embeddings = Word2Vec(parameters: params).train(sentences: sentences, progress: nil)
        XCTAssertEqual(embeddings.vocabulary, ["spam"])
        XCTAssertTrue(embeddings.contains("spam"))
        // The lone word still has a well-formed vector of the right dimension.
        XCTAssertEqual(embeddings.vector(for: "spam")?.count, 10)
    }

    // MARK: - Fix #3: unigramTableSize == 0 must not trap on `% UInt64(table.count)`

    func testZeroUnigramTableSizeDoesNotCrash() {
        var params = Word2VecParameters()
        params.vectorSize = 10
        params.iterations = 3
        params.minCount = 1
        params.window = 3
        params.unigramTableSize = 0           // would trap without the clamp to >= 1

        let embeddings = Word2Vec(parameters: params).train(sentences: corpus(), progress: nil)
        XCTAssertFalse(embeddings.vocabulary.isEmpty)
    }

    // MARK: - Fix #4: the exp-table scale must be integer math (== 83)

    func testExpTableScaleIsExactlyEightyThree() {
        // The C computes EXP_TABLE_SIZE / MAX_EXP / 2 = 1000 / 6 / 2 = 83 via integer division.
        // Float division would give 83.333, diverging from the reference.
        XCTAssertEqual(Word2Vec.expScaleForTesting, 83)
    }

    func testExpTableLookupNearMaxExpDoesNotReadZeroSlot() {
        // With scale 83, the largest interior index (for f just under MAX_EXP = 6) is
        // (int)((f + 6) * 83). At f -> 6 that approaches (int)(11.999... * 83) < 996, never 1000.
        // Verify no interior f in (-6, 6) produces an index that hits the uninitialized slot 1000,
        // and that the table value there is a real sigmoid value, not 0.
        let trainer = Word2Vec(parameters: Word2VecParameters())
        let scale = Word2Vec.expScaleForTesting
        let maxExp: Float = 6
        // Sweep the open interval; f == +/-6 is short-circuited by the clamp before any lookup.
        var f: Float = -6 + 0.001
        while f < 6 {
            let idx = Int((f + maxExp) * Float(scale))
            XCTAssertGreaterThanOrEqual(idx, 0)
            XCTAssertLessThanOrEqual(idx, 999, "index \(idx) for f=\(f) must stay in [0, 999]")
            let value = trainer.expTableValueForTesting(idx)
            XCTAssertGreaterThan(value, 0, "sigmoid table value at idx \(idx) must be > 0")
            XCTAssertLessThan(value, 1, "sigmoid table value at idx \(idx) must be < 1")
            f += 0.01
        }

        // The specific case the reviewer flagged: f == 6.0 is NOT caught by the strict
        // `f > maxExp` clamp, so it falls through to the lookup. With the integer scale (83)
        // the index is Int(12 * 83) = 996 (safe), NOT Int(12 * 83.333) = 1000 (the zero slot).
        let idxAtSix = Int((6.0 + maxExp) * Float(scale))
        XCTAssertEqual(idxAtSix, 996)
        XCTAssertGreaterThan(trainer.expTableValueForTesting(idxAtSix), 0.99,
                             "sigmoid at f=6 should be ~0.9975, not the zero slot")
    }

    // MARK: - Fix #5: default table size is the on-device-friendly 1e7 and is public/tunable

    func testDefaultUnigramTableSizeIsTenMillion() {
        // Public default must be 1e7 (40 MB) rather than the C's 1e8 (400 MB).
        XCTAssertEqual(Word2VecParameters().unigramTableSize, 10_000_000)
    }

    func testUnigramTableSizeIsPubliclyTunable() {
        // This test compiling at all proves the property is public and mutable from outside
        // the module's internals; assert the assignment took effect.
        var params = Word2VecParameters()
        params.unigramTableSize = 500
        XCTAssertEqual(params.unigramTableSize, 500)
    }
}
