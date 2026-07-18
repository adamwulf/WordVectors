//
//  CorpusLoader.swift
//  WordVectors
//
//  Locates the bundled Project Gutenberg corpus .txt files. The `WordVectors/Corpus/`
//  folder is filesystem-synchronized into the app target, so the files ship inside the
//  app bundle. Because Xcode may flatten the folder or preserve it as a subdirectory
//  depending on how the sync group is bundled, we look in both places.
//

import Foundation

/// One bundled corpus book: the file that backs it plus a human-readable title for
/// display in the picker. `Sendable`/`Hashable` so the UI can carry an ordered selection
/// and the background training code can be handed the chosen files without ceremony.
///
/// `stem` is the filename without extension (e.g. `pg1727`) — a stable identity used for
/// selection defaults and equality. `title` is read from the Gutenberg header when the
/// book is discovered, so it stays correct if the bundled books change.
nonisolated struct CorpusBook: Hashable, Sendable {
    let stem: String
    let title: String
    let url: URL
}

/// Pure bundle lookups with no shared mutable state, so they are safe to call from the
/// background training thread. Marked `nonisolated` because the project defaults types to
/// `@MainActor` isolation, which we don't want for these read-only helpers.
nonisolated enum CorpusLoader {

    /// Every bundled book, sorted by title, each with its display title resolved from the
    /// Gutenberg header. This is what the Train screen shows as a checkbox list.
    static func allBooks() -> [CorpusBook] {
        let books = allCorpusFileURLs().map { url -> CorpusBook in
            let stem = url.deletingPathExtension().lastPathComponent
            return CorpusBook(stem: stem, title: title(for: url) ?? stem, url: url)
        }
        // Sort by title so the picker reads naturally; ties broken by stem for stability.
        return books.sorted {
            $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending
                || ($0.title.localizedCaseInsensitiveCompare($1.title) == .orderedSame && $0.stem < $1.stem)
        }
    }

    /// Reads the `Title:` line from a Project Gutenberg file's header. Returns `nil` if the
    /// file can't be read or has no such line. Only the head of the file is scanned so we
    /// never load a multi-megabyte book just to show its name.
    static func title(for url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        // The header always lives near the top; 8 KB is plenty and bounds the read.
        let data = (try? handle.read(upToCount: 8 * 1024)) ?? Data()
        guard let head = String(data: data, encoding: .utf8) else { return nil }

        for line in head.components(separatedBy: "\n") {
            // Gutenberg files are CRLF-terminated, so trim newlines (incl. the trailing
            // `\r`) as well as spaces — `.whitespaces` alone leaves `\r` on the value.
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.lowercased().hasPrefix("title:") else { continue }
            let value = trimmed.dropFirst("title:".count).trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? nil : value
        }
        return nil
    }

    /// All bundled corpus .txt file URLs, regardless of whether Xcode preserved the
    /// `Corpus` subdirectory or flattened the files into the bundle root.
    static func allCorpusFileURLs() -> [URL] {
        var urls: [URL] = []

        // Preferred: files preserved under a `Corpus` subdirectory.
        if let subdirURLs = Bundle.main.urls(forResourcesWithExtension: "txt", subdirectory: "Corpus") {
            urls.append(contentsOf: subdirURLs)
        }

        // Fallback: files flattened into the bundle root. Only include the Gutenberg
        // "pg####" books so we never accidentally pick up unrelated .txt resources.
        if urls.isEmpty, let rootURLs = Bundle.main.urls(forResourcesWithExtension: "txt", subdirectory: nil) {
            urls.append(contentsOf: rootURLs.filter { $0.lastPathComponent.hasPrefix("pg") })
        }

        // Stable order so training is deterministic across launches.
        return urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Corpus file URLs for the given selected book stems. Falls back to the first
    /// available book if none of the requested stems are present, so training never
    /// silently ends up with zero files.
    static func corpusFileURLs(forStems stems: [String]) -> [URL] {
        let all = allCorpusFileURLs()
        let wanted = Set(stems)

        let selected = all.filter { wanted.contains($0.deletingPathExtension().lastPathComponent) }
        // If none of the named books are present (unexpected), fall back to the first
        // available book rather than returning nothing.
        if selected.isEmpty, let first = all.first {
            return [first]
        }
        return selected
    }
}
