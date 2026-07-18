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

/// Pure bundle lookups with no shared mutable state, so they are safe to call from the
/// background training thread. Marked `nonisolated` because the project defaults types to
/// `@MainActor` isolation, which we don't want for these read-only helpers.
nonisolated enum CorpusLoader {

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

    /// Corpus file URLs for the requested scope. Falls back to whatever is bundled if a
    /// requested stem is missing, so training never silently ends up with zero files.
    static func corpusFileURLs(for scope: CorpusScope) -> [URL] {
        let all = allCorpusFileURLs()
        guard let stems = scope.bookStems else { return all }

        let selected = all.filter { url in
            let stem = url.deletingPathExtension().lastPathComponent
            return stems.contains(stem)
        }
        // If none of the named books are present (unexpected), fall back to the first
        // available book rather than returning nothing.
        if selected.isEmpty, let first = all.first {
            return [first]
        }
        return selected
    }
}
