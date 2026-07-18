import Foundation

/// Errors thrown while saving or loading a `WordEmbeddings` model.
public enum WordEmbeddingsIOError: Error, Equatable {
    /// The file's magic bytes did not match the expected format.
    case badMagic
    /// The file's format version is newer/older than this build understands.
    case unsupportedVersion(UInt32)
    /// The data ended before the declared structure was fully read.
    case truncated
    /// A stored word's bytes were not valid UTF-8.
    case invalidWordEncoding
    /// The declared sizes are inconsistent (e.g. storage length != count * vectorSize).
    case inconsistentSizes
}

/// Compact, exact, little-endian binary persistence for `WordEmbeddings`.
///
/// Layout:
///   - magic:      4 bytes, ASCII "WVK1"
///   - version:    UInt32 (little-endian) — currently 1
///   - vectorSize: UInt32
///   - wordCount:  UInt32
///   - words:      wordCount entries of [ UInt32 byteLen | UTF-8 bytes ]
///   - storage:    wordCount * vectorSize Float32 values (bit pattern, little-endian)
///
/// The round-trip is exact: Float bit patterns are written verbatim, so reloaded vectors
/// are bit-for-bit identical to the trained ones. JSON is intentionally avoided (a large
/// [Float] as JSON is huge and lossy in round-tripping).
extension WordEmbeddings {

    private static let magic: [UInt8] = Array("WVK1".utf8)
    private static let formatVersion: UInt32 = 1

    /// Serializes the model to a compact binary `Data`.
    public func serialized() -> Data {
        let words = self.vocabulary
        let dim = self.vectorSize

        var data = Data()
        data.append(contentsOf: WordEmbeddings.magic)
        appendUInt32(WordEmbeddings.formatVersion, to: &data)
        appendUInt32(UInt32(dim), to: &data)
        appendUInt32(UInt32(words.count), to: &data)

        for word in words {
            let bytes = Array(word.utf8)
            appendUInt32(UInt32(bytes.count), to: &data)
            data.append(contentsOf: bytes)
        }

        // Append the flat storage as raw little-endian Float32 bit patterns.
        for word in words {
            // vector(for:) returns a defensive copy in vocabulary order.
            let vec = self.vector(for: word)!
            for value in vec {
                appendUInt32(value.bitPattern, to: &data)
            }
        }

        return data
    }

    /// Writes the model to `url` (atomically) using the compact binary format.
    public func save(to url: URL) throws {
        try serialized().write(to: url, options: .atomic)
    }

    /// Reconstructs a model from binary `data` produced by `serialized()` / `save(to:)`.
    public convenience init(serialized data: Data) throws {
        var cursor = 0

        // Magic.
        guard data.count >= 4 else { throw WordEmbeddingsIOError.truncated }
        let magic = Array(data[data.startIndex..<data.index(data.startIndex, offsetBy: 4)])
        guard magic == WordEmbeddings.magic else { throw WordEmbeddingsIOError.badMagic }
        cursor = 4

        // Version.
        let version = try WordEmbeddings.readUInt32(data, &cursor)
        guard version == WordEmbeddings.formatVersion else {
            throw WordEmbeddingsIOError.unsupportedVersion(version)
        }

        let dim = Int(try WordEmbeddings.readUInt32(data, &cursor))
        let count = Int(try WordEmbeddings.readUInt32(data, &cursor))

        // Words.
        var words = [String]()
        words.reserveCapacity(count)
        for _ in 0..<count {
            let byteLen = Int(try WordEmbeddings.readUInt32(data, &cursor))
            guard cursor + byteLen <= data.count else { throw WordEmbeddingsIOError.truncated }
            let start = data.index(data.startIndex, offsetBy: cursor)
            let end = data.index(start, offsetBy: byteLen)
            let bytes = data[start..<end]
            guard let word = String(bytes: bytes, encoding: .utf8) else {
                throw WordEmbeddingsIOError.invalidWordEncoding
            }
            words.append(word)
            cursor += byteLen
        }

        // Storage.
        let floatCount = count * dim
        var storage = [Float]()
        storage.reserveCapacity(floatCount)
        for _ in 0..<floatCount {
            let bits = try WordEmbeddings.readUInt32(data, &cursor)
            storage.append(Float(bitPattern: bits))
        }

        guard storage.count == count * dim else { throw WordEmbeddingsIOError.inconsistentSizes }

        self.init(words: words, vectorSize: dim, storage: storage)
    }

    /// Loads a model from a file previously written by `save(to:)`.
    public convenience init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        try self.init(serialized: data)
    }

    // MARK: - Little-endian helpers

    private func appendUInt32(_ value: UInt32, to data: inout Data) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    private static func readUInt32(_ data: Data, _ cursor: inout Int) throws -> UInt32 {
        guard cursor + 4 <= data.count else { throw WordEmbeddingsIOError.truncated }
        var value: UInt32 = 0
        let start = data.index(data.startIndex, offsetBy: cursor)
        for i in 0..<4 {
            value |= UInt32(data[data.index(start, offsetBy: i)]) << (8 * i)
        }
        cursor += 4
        return value // reconstructed as little-endian above
    }
}
