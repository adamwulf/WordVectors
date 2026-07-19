import Foundation
import WordVectorKit

let bookNames = [
    "pg100.txt", "pg1342.txt", "pg1661.txt", "pg1727.txt", "pg2554.txt",
    "pg2641.txt", "pg2701.txt", "pg3268.txt", "pg65238.txt", "pg67979.txt",
]

guard CommandLine.arguments.count == 2 else {
    fputs("usage: w2v-bench <corpus-directory>\n", stderr)
    exit(2)
}

let corpusDirectory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
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

let trainer = Word2Vec(parameters: Word2VecParameters())
let start = ProcessInfo.processInfo.systemUptime
let model = trainer.train(sentences: sentences, progress: nil)
let seconds = ProcessInfo.processInfo.systemUptime - start

let checksumWord = model.contains("the") ? "the" : model.vocabulary.first
guard let checksumWord, let vector = model.vector(for: checksumWord) else {
    fputs("training produced an empty vocabulary\n", stderr)
    exit(1)
}
let checksum = vector.reduce(Float.zero, +)

print(String(format: "train_seconds=%.3f", seconds))
print(String(format: "checksum_word=%@ checksum=%.6f vocab_count=%d", checksumWord, checksum, model.vocabulary.count))
