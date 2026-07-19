import Foundation
import WordVectorKit

let bookNames = [
    "pg100.txt", "pg1342.txt", "pg1661.txt", "pg1727.txt", "pg2554.txt",
    "pg2641.txt", "pg2701.txt", "pg3268.txt", "pg65238.txt", "pg67979.txt",
]

// First non-flag argument is the corpus directory; flags (e.g. "--check-determinism") may follow.
let positional = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("-") }
guard let corpusPath = positional.first, positional.count == 1 else {
    fputs("usage: w2v-bench <corpus-directory> [--check-determinism]\n", stderr)
    exit(2)
}

let corpusDirectory = URL(fileURLWithPath: corpusPath, isDirectory: true)
var sentences: [String] = []

do {
    for bookName in bookNames {
        let url = corpusDirectory.appendingPathComponent(bookName)
        let text = try String(contentsOf: url, encoding: .utf8)
        sentences.append(contentsOf: CorpusPreprocessor.sentences(fromGutenberg: text))
    }
} catch {
    fputs("failed to load corpus: \(error)\n", stderr)
    exit(1)
}

// Pass "--check-determinism" (or "-d") as an extra trailing arg to train TWICE and compare the
// resulting vectors element-by-element — a full-corpus proof that the deterministic path is
// bit-exact under real, contended multithreading.
let checkDeterminism = CommandLine.arguments.contains("--check-determinism")
    || CommandLine.arguments.contains("-d")

let parameters = Word2VecParameters()
let start = ProcessInfo.processInfo.systemUptime
let model = Word2Vec(parameters: parameters).train(sentences: sentences, progress: nil)
let seconds = ProcessInfo.processInfo.systemUptime - start

let checksumWord = model.contains("the") ? "the" : model.vocabulary.first
guard let checksumWord, let vector = model.vector(for: checksumWord) else {
    fputs("training produced an empty vocabulary\n", stderr)
    exit(1)
}
let checksum = vector.reduce(Float.zero, +)

print(String(format: "train_seconds=%.3f", seconds))
print(String(format: "checksum_word=%@ checksum=%.6f vocab_count=%d", checksumWord, checksum, model.vocabulary.count))
print("deterministic=\(parameters.deterministic)")

if checkDeterminism {
    // Train a second time with identical parameters and compare every vector.
    let model2 = Word2Vec(parameters: parameters).train(sentences: sentences, progress: nil)

    guard model.vocabulary == model2.vocabulary else {
        fputs("vocabulary differs between runs\n", stderr)
        exit(1)
    }

    var maxAbsDiff: Float = 0
    var mismatchedWords = 0
    for word in model.vocabulary {
        guard let a = model.vector(for: word), let b = model2.vector(for: word) else { continue }
        var wordMismatch = false
        for (x, y) in zip(a, b) {
            let diff = abs(x - y)
            if diff > maxAbsDiff { maxAbsDiff = diff }
            if diff != 0 { wordMismatch = true }
        }
        if wordMismatch { mismatchedWords += 1 }
    }

    let bitExact = maxAbsDiff == 0 && mismatchedWords == 0
    print(String(format: "bit_exact=%@ max_abs_diff=%.9g mismatched_words=%d/%d",
                 bitExact ? "true" : "false", maxAbsDiff, mismatchedWords, model.vocabulary.count))
    exit(bitExact ? 0 : 3)
}
