import XCTest
@testable import WordVectorKit

final class CorpusPreprocessorTests: XCTestCase {

    // MARK: - Boilerplate stripping

    func testStripGutenbergBoilerplateRemovesHeaderAndFooter() {
        let text = """
        This is a preamble the Gutenberg license blah blah.
        Copyright notices and metadata.
        *** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***
        Call me Ishmael.
        Some years ago never mind how long.
        *** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***
        Trailing license text that should be dropped.
        """
        let stripped = CorpusPreprocessor.stripGutenbergBoilerplate(text)
        XCTAssertTrue(stripped.contains("Call me Ishmael."))
        XCTAssertTrue(stripped.contains("Some years ago"))
        XCTAssertFalse(stripped.contains("preamble"))
        XCTAssertFalse(stripped.contains("Trailing license"))
    }

    func testStripGutenbergHandlesTHISVariant() {
        let text = """
        header
        *** START OF THIS PROJECT GUTENBERG EBOOK SOMETHING ***
        the body line one
        *** END OF THIS PROJECT GUTENBERG EBOOK SOMETHING ***
        footer
        """
        let stripped = CorpusPreprocessor.stripGutenbergBoilerplate(text)
        XCTAssertTrue(stripped.contains("the body line one"))
        XCTAssertFalse(stripped.contains("header"))
        XCTAssertFalse(stripped.contains("footer"))
    }

    func testStripGutenbergReturnsUnchangedWhenNoMarkers() {
        let text = "Just some plain text with no markers at all.\nSecond line."
        let stripped = CorpusPreprocessor.stripGutenbergBoilerplate(text)
        XCTAssertEqual(stripped, text)
    }

    func testStripGutenbergHandlesMissingEndMarker() {
        let text = """
        header
        *** START OF THE PROJECT GUTENBERG EBOOK X ***
        body content survives
        """
        let stripped = CorpusPreprocessor.stripGutenbergBoilerplate(text)
        XCTAssertTrue(stripped.contains("body content survives"))
        XCTAssertFalse(stripped.contains("header"))
    }

    // MARK: - Normalization

    func testNormalizeLowercases() {
        let sentences = CorpusPreprocessor.normalize("The Cat SAT.")
        XCTAssertEqual(sentences, ["the cat sat"])
    }

    func testNormalizePunctuationBecomesSpaceAndCollapses() {
        // Commas and extra spaces should collapse into single-space token separators.
        let sentences = CorpusPreprocessor.normalize("hello,   world;;;foo")
        XCTAssertEqual(sentences, ["hello world foo"])
    }

    func testNormalizeSplitsSentencesOnTerminators() {
        let sentences = CorpusPreprocessor.normalize("the cat sat. the dog ran! did it? yes indeed")
        XCTAssertEqual(sentences, [
            "the cat sat",
            "the dog ran",
            "did it",
            "yes indeed"
        ])
    }

    func testNormalizeSplitsOnNewlines() {
        let sentences = CorpusPreprocessor.normalize("first line here\nsecond line here")
        XCTAssertEqual(sentences, [
            "first line here",
            "second line here"
        ])
    }

    func testNormalizeDropsEmptyAndSingleTokenSentences() {
        // "." -> empty; "hi." -> single token (dropped); "one two." kept.
        let sentences = CorpusPreprocessor.normalize("hi. . one two. also!")
        // "hi" is 1 token -> dropped; "" -> dropped; "one two" kept; "also" 1 token -> dropped.
        XCTAssertEqual(sentences, ["one two"])
    }

    func testNormalizeStripsDigitsAsSeparators() {
        // Digits are not letters, so they act as separators (word2vec-style letter-only tokens).
        let sentences = CorpusPreprocessor.normalize("chapter 12 begins now")
        XCTAssertEqual(sentences, ["chapter begins now"])
    }

    func testNormalizeEmptyInputYieldsNoSentences() {
        XCTAssertEqual(CorpusPreprocessor.normalize(""), [])
        XCTAssertEqual(CorpusPreprocessor.normalize("   \n  \n"), [])
    }

    // MARK: - Convenience composition

    func testSentencesFromGutenbergComposesBothSteps() {
        let text = """
        preamble to drop
        *** START OF THE PROJECT GUTENBERG EBOOK T ***
        The Cat Sat. The Dog Ran!
        *** END OF THE PROJECT GUTENBERG EBOOK T ***
        footer to drop
        """
        let sentences = CorpusPreprocessor.sentences(fromGutenberg: text)
        XCTAssertEqual(sentences, ["the cat sat", "the dog ran"])
    }
}
