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
///
/// `Codable` so it can be written to a small JSON sidecar beside the cached binary model and
/// restored on the next launch — that's what lets a cache-loaded model still show its corpus,
/// timing, and hyperparameters instead of a bare "loaded from cache".
struct TrainingInfo: Codable {
    /// Titles of the books that were trained on, in the order they were loaded.
    let bookTitles: [String]
    let sentenceCount: Int
    let vocabularyCount: Int
    let duration: TimeInterval
    /// The hyperparameters this model was actually trained with, so the summary can
    /// report the values the user chose rather than the hard-coded defaults.
    let parameters: Word2VecParameters

    /// A short human-readable summary of the corpus (e.g. "1 book" or "3 books").
    var scopeSummary: String {
        Self.bookCountSummary(bookTitles.count)
    }

    /// Formats a book count as "1 book" / "N books". Shared so every corpus-size label
    /// (this summary, the training footer, its VoiceOver label, the book-list sheet) pluralizes
    /// identically from one place.
    static func bookCountSummary(_ count: Int) -> String {
        count == 1 ? "1 book" : "\(count) books"
    }
}

@MainActor
final class ModelStore {

    /// Singleton shared by the three feature tabs so they all see the same model.
    static let shared = ModelStore()

    private(set) var state: ModelState = .idle {
        didSet { notifyObservers() }
    }

    /// Info about the model's training run. Populated from a live run, or restored from the
    /// cache sidecar on launch; `nil` only when a cached model has no readable sidecar.
    private(set) var lastTrainingInfo: TrainingInfo?

    /// `true` when `lastTrainingInfo` was restored from the on-disk cache rather than produced
    /// by a training run this session, so the UI can say the detail describes a cached model.
    private(set) var trainingInfoFromCache = false

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

    /// JSON sidecar holding the `TrainingInfo` for the cached model (corpus, timing, params).
    /// Kept beside `cacheURL`; a missing or stale sidecar just means "no cached detail".
    private var metadataURL: URL {
        cacheURL.deletingPathExtension().appendingPathExtension("json")
    }

    /// On-disk size, in bytes, of the cached model file — the size of the final saved model.
    /// `nil` when nothing is cached (e.g. the model didn't persist, or was cleared). Reflects
    /// the actual file the app wrote, not a re-computed estimate, so it matches what's on disk.
    var cachedModelByteSize: Int? {
        guard let values = try? cacheURL.resourceValues(forKeys: [.fileSizeKey]) else { return nil }
        return values.fileSize
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
        let metadataURL = self.metadataURL
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
                // Best-effort: recover the training detail from the sidecar so a cache-loaded
                // model can show its corpus, timing, and hyperparameters. Only trust it if it
                // matches the model actually loaded (a stale sidecar would show wrong detail).
                let info = ModelStore.loadMetadata(from: metadataURL, matching: model)
                await MainActor.run {
                    ModelStore.shared.lastTrainingInfo = info
                    ModelStore.shared.trainingInfoFromCache = (info != nil)
                    ModelStore.shared.state = .ready(model)
                }
            } catch {
                // Corrupt/incompatible cache — fall back to idle so the user can retrain.
                appLog.error("Failed to load cached model: \(String(describing: error), privacy: .public)")
                await MainActor.run { ModelStore.shared.state = .idle }
            }
        }
    }

    /// Decodes the `TrainingInfo` sidecar at `url`, returning it only if it's consistent with
    /// `model` (same vector size and vocabulary count). A missing, corrupt, or mismatched
    /// sidecar returns `nil` — the model still loads, just without the extra detail.
    nonisolated private static func loadMetadata(from url: URL, matching model: WordEmbeddings) -> TrainingInfo? {
        guard let data = try? Data(contentsOf: url),
              let info = try? JSONDecoder().decode(TrainingInfo.self, from: data)
        else { return nil }
        guard info.parameters.vectorSize == model.vectorSize,
              info.vocabularyCount == model.vocabulary.count
        else {
            appLog.info("Cached model metadata was stale; ignoring it.")
            return nil
        }
        return info
    }

    /// Clears any cached model (and its metadata sidecar) from disk and resets to idle.
    func clearCache() {
        try? FileManager.default.removeItem(at: cacheURL)
        try? FileManager.default.removeItem(at: metadataURL)
        lastTrainingInfo = nil
        trainingInfoFromCache = false
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
    ///
    /// `parameters` carries the (already clamped) hyperparameters to train with. The caller —
    /// the Train tab — supplies the user's edits; the Tier-2/3 fields it doesn't expose keep
    /// their `Word2VecParameters` defaults.
    func train(stems: [String], parameters: Word2VecParameters) {
        if case .training = state { return }
        if case .loading = state { return }

        let cacheURL = self.cacheURL
        let metadataURL = self.metadataURL
        appLog.info("Training requested for \(stems.count, privacy: .public) book(s): \(stems.joined(separator: ", "), privacy: .public).")
        state = .training(progress: 0)

        Task.detached(priority: .userInitiated) {
            let result = ModelStore.performTraining(stems: stems, parameters: parameters) { progress in
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
                    ModelStore.shared.trainingInfoFromCache = false
                    // Cache for instant subsequent launches (best-effort; failure is non-fatal).
                    // The model is persisted BEFORE state becomes `.ready` so the first render
                    // can report the model's on-disk size; a save failure just omits that detail.
                    do {
                        try model.save(to: cacheURL)
                        appLog.info("Cached trained model to disk.")
                        // Only write the metadata sidecar once the model itself is saved, so a
                        // sidecar never describes a model that isn't on disk. If this fails the
                        // model still loads next launch, just without the extra detail.
                        do {
                            let encoded = try JSONEncoder().encode(info)
                            try encoded.write(to: metadataURL, options: .atomic)
                        } catch {
                            appLog.error("Could not cache model metadata: \(String(describing: error), privacy: .public)")
                        }
                    } catch {
                        appLog.error("Could not cache model: \(String(describing: error), privacy: .public)")
                        // The model didn't persist — drop any stale sidecar so a future launch
                        // doesn't pair fresh metadata with an old cached model.
                        try? FileManager.default.removeItem(at: metadataURL)
                    }
                    // Publish the ready model only after the save attempt, so `readyDetail`'s
                    // on-disk size reflects the file just written (or is correctly absent).
                    ModelStore.shared.state = .ready(model)
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
    /// preprocesses them, and runs Word2Vec. `progress` is invoked on training worker threads.
    nonisolated private static func performTraining(
        stems: [String],
        parameters: Word2VecParameters,
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

        // 3) Train with the parameters the caller supplied. The Train tab lets the user
        // edit the Tier-1 knobs (vector size, window, min count, iterations); every other
        // field keeps its `Word2VecParameters` default (e.g. skip-gram, which gives better
        // quality on a small corpus).
        let params = parameters
        appLog.info("Params: dims=\(params.vectorSize, privacy: .public) window=\(params.window, privacy: .public) minCount=\(params.minCount, privacy: .public) iter=\(params.iterations, privacy: .public) cbow=\(params.useCBOW, privacy: .public)")

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
            duration: duration,
            parameters: params
        )
        return .success(model, info)
    }
}
