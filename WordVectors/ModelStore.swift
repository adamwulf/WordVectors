//
//  ModelStore.swift
//  WordVectors
//
//  Owns the trained WordEmbeddings model: loads a cached model on launch,
//  trains from the bundled Gutenberg corpus off the main thread, caches the
//  result to disk, and exposes the queries used by the three feature screens.
//

import Foundation
import OSLog
import WordVectorKit

/// Shared logger so training + query milestones show up in the simulator console
/// (filter by subsystem `com.milestonemade.WordVectors`). `Logger` is `Sendable`, and this
/// is `nonisolated` so the off-main training code can log without crossing an actor boundary.
nonisolated let appLog = Logger(subsystem: "com.milestonemade.WordVectors", category: "WordVectors")

/// The stem of the book the picker selects by default. The Odyssey (`pg1727`, ~717 KB)
/// is a good starting point: it's short (a fresh model is ready in well under a minute on
/// a simulator) and it contains king/queen/man/woman, so the classic `king − man + woman`
/// analogy has all its inputs in vocabulary.
nonisolated let defaultBookStem = "pg1727"

/// The high-level state the UI reflects.
enum ModelState {
    case idle                     // nothing loaded yet
    case loading                  // reading a cached model from disk
    case training(progress: Double)
    case ready(WordEmbeddings)
    case failed(String)
}

/// A snapshot description of the corpus actually used to train the live model.
struct TrainingInfo {
    /// Titles of the books that were trained on, in the order they were loaded.
    let bookTitles: [String]
    let sentenceCount: Int
    let vocabularyCount: Int
    let duration: TimeInterval

    /// A short human-readable summary of the corpus (e.g. "1 book" or "3 books").
    var scopeSummary: String {
        bookTitles.count == 1 ? "1 book" : "\(bookTitles.count) books"
    }
}

@MainActor
final class ModelStore {

    /// Singleton shared by the three feature tabs so they all see the same model.
    static let shared = ModelStore()

    private(set) var state: ModelState = .idle {
        didSet { notifyObservers() }
    }

    /// Info about the last successful training run (nil if loaded from cache without retraining).
    private(set) var lastTrainingInfo: TrainingInfo?

    /// The ready model, if any.
    var embeddings: WordEmbeddings? {
        if case let .ready(model) = state { return model }
        return nil
    }

    // MARK: - Observation

    /// One registered observer: a weak owner plus its state handler. Holding the owner
    /// weakly means observers never need to unregister in `deinit` (which would require
    /// capturing `self` in a closure that outlives deinit — a Swift 6 error).
    private struct Observer {
        weak var owner: AnyObject?
        let handler: (ModelState) -> Void
    }

    private var observers: [Observer] = []

    /// Registers `handler`, called immediately with the current state and on every change.
    /// The registration lives as long as `owner` does; no manual removal is required.
    func addObserver(_ owner: AnyObject, _ handler: @escaping (ModelState) -> Void) {
        observers.append(Observer(owner: owner, handler: handler))
        handler(state)
    }

    private func notifyObservers() {
        // Drop observers whose owners have deallocated, then fire the rest.
        observers.removeAll { $0.owner == nil }
        for observer in observers {
            observer.handler(state)
        }
    }

    // MARK: - Cache location

    /// File used to cache the trained model between launches (in Caches — safe to be purged).
    private var cacheURL: URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("word-embeddings.wvk", isDirectory: false)
    }

    // MARK: - Lifecycle

    private init() {}

    /// Called once at launch. Loads the cached model if present; otherwise leaves state `.idle`
    /// so the user can kick off the first training run from the Train tab.
    func bootstrap() {
        guard case .idle = state else { return }
        loadCachedModel()
    }

    /// Attempts to load a previously-cached model. On success, state becomes `.ready`.
    private func loadCachedModel() {
        let url = cacheURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            appLog.info("No cached model found; starting idle.")
            state = .idle
            return
        }
        appLog.info("Loading cached model from \(url.lastPathComponent, privacy: .public)…")
        state = .loading
        Task.detached(priority: .userInitiated) {
            do {
                let model = try WordEmbeddings(contentsOf: url)
                // A cached-but-empty model is useless; treat it as no cache.
                guard !model.vocabulary.isEmpty else {
                    appLog.info("Cached model was empty; ignoring it.")
                    await MainActor.run { ModelStore.shared.state = .idle }
                    return
                }
                appLog.info("Loaded cached model: vocab size = \(model.vocabulary.count, privacy: .public).")
                await MainActor.run { ModelStore.shared.state = .ready(model) }
            } catch {
                // Corrupt/incompatible cache — fall back to idle so the user can retrain.
                appLog.error("Failed to load cached model: \(String(describing: error), privacy: .public)")
                await MainActor.run { ModelStore.shared.state = .idle }
            }
        }
    }

    /// Clears any cached model from disk and resets to idle.
    func clearCache() {
        try? FileManager.default.removeItem(at: cacheURL)
        lastTrainingInfo = nil
        appLog.info("Cleared cached model.")
        state = .idle
    }

    // MARK: - Training

    /// Kicks off training on a background queue. Progress and completion are delivered on main.
    /// Re-entrancy is guarded: a call while already training is ignored.
    ///
    /// `stems` is the set of selected book filename stems (e.g. `["pg1727"]`). The caller is
    /// responsible for ensuring at least one book is selected; an empty list falls back to
    /// the first bundled book so training never runs on nothing.
    func train(stems: [String]) {
        if case .training = state { return }
        if case .loading = state { return }

        let cacheURL = self.cacheURL
        appLog.info("Training requested for \(stems.count, privacy: .public) book(s): \(stems.joined(separator: ", "), privacy: .public).")
        state = .training(progress: 0)

        Task.detached(priority: .userInitiated) {
            let result = ModelStore.performTraining(stems: stems) { progress in
                // `progress` fires on the background thread — hop to main to update UI.
                Task { @MainActor in
                    // Only reflect progress while we're still in the training phase.
                    if case .training = ModelStore.shared.state {
                        ModelStore.shared.state = .training(progress: progress)
                    }
                }
            }

            await MainActor.run {
                switch result {
                case let .success(model, info):
                    ModelStore.shared.lastTrainingInfo = info
                    ModelStore.shared.state = .ready(model)
                    // Cache for instant subsequent launches (best-effort; failure is non-fatal).
                    do {
                        try model.save(to: cacheURL)
                        appLog.info("Cached trained model to disk.")
                    } catch {
                        appLog.error("Could not cache model: \(String(describing: error), privacy: .public)")
                    }
                case let .failure(message):
                    appLog.error("Training failed: \(message, privacy: .public)")
                    ModelStore.shared.state = .failed(message)
                }
            }
        }
    }

    // MARK: - Off-main training work (nonisolated — never touches main-actor state)

    private enum TrainingResult {
        case success(WordEmbeddings, TrainingInfo)
        case failure(String)
    }

    /// Pure, main-actor-free training. Loads the selected corpus files from the app bundle,
    /// preprocesses them, and runs Word2Vec. `progress` is invoked on the calling thread.
    nonisolated private static func performTraining(
        stems: [String],
        progress: @escaping (Double) -> Void
    ) -> TrainingResult {
        let start = Date()

        // 1) Locate the corpus files in the app bundle.
        let urls = CorpusLoader.corpusFileURLs(forStems: stems)
        guard !urls.isEmpty else {
            return .failure("No corpus files found in the app bundle.")
        }
        appLog.info("Loading \(urls.count, privacy: .public) corpus file(s): \(urls.map { $0.lastPathComponent }.joined(separator: ", "), privacy: .public)")

        // 2) Read + preprocess each book into sentences, capturing its title for the summary.
        var sentences: [String] = []
        var bookTitles: [String] = []
        for url in urls {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                appLog.error("Could not read corpus file \(url.lastPathComponent, privacy: .public)")
                continue
            }
            bookTitles.append(CorpusLoader.title(for: url) ?? url.deletingPathExtension().lastPathComponent)
            sentences.append(contentsOf: CorpusPreprocessor.sentences(fromGutenberg: text))
        }
        guard !sentences.isEmpty else {
            return .failure("The corpus produced no usable sentences.")
        }
        let wordCount = sentences.reduce(0) { $0 + $1.split(separator: " ").count }
        appLog.info("Loaded \(sentences.count, privacy: .public) sentences (~\(wordCount, privacy: .public) tokens). Training started…")

        // 3) Train. Defaults are tuned for a fast, usable on-device demo.
        var params = Word2VecParameters()
        params.vectorSize = 100
        params.window = 5
        params.minCount = 5
        params.iterations = 5
        params.useCBOW = false   // skip-gram: better quality on a small corpus

        let word2vec = Word2Vec(parameters: params)
        var lastLoggedDecile = -1
        let model = word2vec.train(sentences: sentences) { fraction in
            // Log every ~10% so progress is visible in the console without spamming it.
            let decile = Int(fraction * 10)
            if decile != lastLoggedDecile {
                lastLoggedDecile = decile
                appLog.info("Training progress: \(Int(fraction * 100), privacy: .public)%")
            }
            progress(fraction)
        }

        guard !model.vocabulary.isEmpty else {
            return .failure("Training produced an empty vocabulary. Try a larger corpus or a lower min-count.")
        }

        let duration = Date().timeIntervalSince(start)
        appLog.info("Training done in \(String(format: "%.1f", duration), privacy: .public)s. Vocab size = \(model.vocabulary.count, privacy: .public).")

        let info = TrainingInfo(
            bookTitles: bookTitles,
            sentenceCount: sentences.count,
            vocabularyCount: model.vocabulary.count,
            duration: duration
        )
        return .success(model, info)
    }
}
