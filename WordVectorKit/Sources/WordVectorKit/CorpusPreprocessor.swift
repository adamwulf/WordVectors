import Foundation

/// Turns raw Project Gutenberg text into training-ready sentences.
///
/// The pipeline is intentionally simple and deterministic so that downstream
/// tokenization is a plain `split(separator: " ")`. Each emitted sentence is a
/// space-joined, lowercased, punctuation-stripped token stream.
public enum CorpusPreprocessor {

    /// Removes Project Gutenberg boilerplate that wraps the actual book text.
    ///
    /// Everything before the line matching `*** START OF THE PROJECT GUTENBERG EBOOK ... ***`
    /// and everything after `*** END OF ... ***` is dropped. Both the "THE" and "THIS"
    /// spellings of the marker are handled (Gutenberg has used both over the years).
    /// If the markers are absent, the text is returned unchanged.
    public static func stripGutenbergBoilerplate(_ text: String) -> String {
        let lines = text.components(separatedBy: "\n")

        var startIndex: Int? = nil
        var endIndex: Int? = nil

        for (i, line) in lines.enumerated() {
            let upper = line.uppercased()
            if startIndex == nil, isStartMarker(upper) {
                // The book body begins on the line AFTER the START marker.
                startIndex = i + 1
            } else if startIndex != nil, endIndex == nil, isEndMarker(upper) {
                // The book body ends on the line BEFORE the END marker.
                endIndex = i
                break
            }
        }

        // If neither marker is present, return the text unchanged.
        guard startIndex != nil || endIndex != nil else {
            return text
        }

        let lower = startIndex ?? 0
        let upper = endIndex ?? lines.count
        guard lower <= upper else {
            // Malformed markers (END before START) — fail safe by returning original.
            return text
        }

        return lines[lower..<upper].joined(separator: "\n")
    }

    /// Returns an array of sentences, one per element. Each sentence is a
    /// space-joined, lowercased, punctuation-stripped token stream.
    ///
    /// Processing order (order matters — sentence boundaries must survive stripping):
    /// 1. Lowercase.
    /// 2. Split into sentences on sentence-ending punctuation (`.`, `!`, `?`, newlines)
    ///    BEFORE stripping punctuation, so boundaries are preserved.
    /// 3. Within each sentence: replace any character that is not a letter or whitespace
    ///    with a space, then collapse runs of whitespace to single spaces.
    /// 4. Drop empty sentences and 1-token sentences.
    public static func normalize(_ text: String) -> [String] {
        let lowered = text.lowercased()

        // Split into sentence fragments on sentence-ending punctuation and newlines,
        // done BEFORE we strip punctuation so the boundaries still exist.
        var sentences: [String] = []
        var current = String.UnicodeScalarView()
        for scalar in lowered.unicodeScalars {
            if scalar == "." || scalar == "!" || scalar == "?" || scalar == "\n" || scalar == "\r" {
                sentences.append(String(current))
                current = String.UnicodeScalarView()
            } else {
                current.append(scalar)
            }
        }
        // Trailing fragment (text after the last terminator, or the whole text if none).
        sentences.append(String(current))

        var result: [String] = []
        result.reserveCapacity(sentences.count)

        for sentence in sentences {
            // Replace any character that is not a letter or whitespace with a space,
            // then collapse whitespace runs to single spaces.
            var tokens: [String] = []
            var token = String.UnicodeScalarView()
            for scalar in sentence.unicodeScalars {
                if CharacterSet.letters.contains(scalar) {
                    token.append(scalar)
                } else {
                    // Whitespace or any non-letter acts as a token boundary.
                    if !token.isEmpty {
                        tokens.append(String(token))
                        token = String.UnicodeScalarView()
                    }
                }
            }
            if !token.isEmpty {
                tokens.append(String(token))
            }

            // Drop empty and 1-token sentences (no useful context window in a lone token).
            if tokens.count >= 2 {
                result.append(tokens.joined(separator: " "))
            }
        }

        return result
    }

    /// Convenience that strips Gutenberg boilerplate then normalizes into sentences.
    public static func sentences(fromGutenberg text: String) -> [String] {
        return normalize(stripGutenbergBoilerplate(text))
    }

    // MARK: - Marker matching

    private static func isStartMarker(_ upperLine: String) -> Bool {
        // e.g. "*** START OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***"
        //  or  "*** START OF THIS PROJECT GUTENBERG EBOOK ... ***"
        guard upperLine.contains("*** START OF ") else { return false }
        return upperLine.contains("PROJECT GUTENBERG")
    }

    private static func isEndMarker(_ upperLine: String) -> Bool {
        // e.g. "*** END OF THE PROJECT GUTENBERG EBOOK MOBY DICK ***"
        //  or  "*** END OF THIS PROJECT GUTENBERG EBOOK ... ***"
        guard upperLine.contains("*** END OF ") else { return false }
        return upperLine.contains("PROJECT GUTENBERG")
    }
}
