//
//  GraphView.swift
//  WordVectors
//
//  Explore combines a full-vocabulary PCA scatter plot with nearest-word lookup.
//  PCA is intentionally performed away from the main actor because projecting a
//  large vocabulary is CPU-intensive enough to interrupt UI interactions.
//

import Charts
import Combine
import os
import SwiftUI
import WordVectorKit

/// One plotted word, including its original frequency rank for color encoding.
nonisolated private struct ProjectedWord: Identifiable, Sendable {
    let word: String
    let x: Float
    let y: Float
    let rank: Int

    var id: String { word }
}

/// A display-friendly nearest-neighbor result. `score` is the selected metric's natural value
/// (similarity for cosine/dot, distance for Euclidean).
nonisolated private struct NeighborResult: Identifiable, Sendable {
    let word: String
    let score: Float

    var id: String { word }
}

/// How the scatter plot colors its points. Defaults to `.frequency` so nothing about the existing
/// experience changes until the user deliberately switches to `.clusters`.
private enum ColorMode: String, CaseIterable, Identifiable {
    case frequency
    case clusters

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .frequency: return "Frequency"
        case .clusters: return "Clusters"
        }
    }
}

/// One legend row in Clusters mode: the cluster's index, its representative words, and its size.
/// `isVisible` reflects the current top-N (and legend-isolation) filter so the row can render its
/// swatch vivid or muted to match the plotted points.
private struct ClusterLegendEntry: Identifiable, Sendable {
    let cluster: Int
    let representatives: [String]
    let size: Int
    let isVisible: Bool

    var id: Int { cluster }

    /// Representative words joined for a compact single-line label ("spaceship · planet · alien").
    var label: String { representatives.joined(separator: " · ") }
}

/// A sendable projection job. `WordEmbeddings` is immutable after initialization, so it is
/// safe for the detached task to read even though the package does not currently declare the
/// reference type `Sendable`. Keeping that unchecked boundary here makes the concurrency
/// assumption narrow and explicit.
nonisolated private struct ProjectionRequest: @unchecked Sendable {
    let embeddings: WordEmbeddings
    let wordCount: Int
}

/// A sendable clustering job. Shares `ProjectionRequest`'s reasoning: `WordEmbeddings` is
/// immutable after init, so the detached k-means run can read it across the unchecked boundary.
/// `k` and `wordCount` travel with the request so a stale generation can be recognized purely by
/// its generation number, without inspecting the model.
nonisolated private struct ClusteringRequest: @unchecked Sendable {
    let embeddings: WordEmbeddings
    let k: Int
    let wordCount: Int
}

/// Bridges `ModelStore`'s closure-based UIKit observation into SwiftUI-published state.
///
/// The selected word is the single source of truth for both halves of Explore. Graph taps,
/// submitted searches, and neighbor-row taps all pass through `select(word:)`, ensuring the
/// graph highlight, query text, message, and results cannot drift apart.
@MainActor
private final class GraphViewModel: ObservableObject {
    @Published private(set) var vocabularyCount = 0
    @Published private(set) var points: [ProjectedWord] = []
    @Published private(set) var isProjecting = false
    @Published private(set) var emptyMessage = "Train a model first (see the Train tab)."
    @Published private(set) var selectedWord: String?
    @Published private(set) var neighbors: [NeighborResult] = []
    @Published private(set) var searchMessage = "Select a point or search the vocabulary."
    @Published var query = "king"

    /// The distance metric used to rank neighbors. Changing it re-ranks the current selection so
    /// the list and its score column immediately reflect the new metric.
    @Published var metric: DistanceMetric = .cosine {
        didSet {
            guard metric != oldValue else { return }
            rerankCurrentSelection()
        }
    }

    /// How the plot colors its points. Frequency is the default and leaves the original gradient
    /// behavior untouched; switching to Clusters kicks off (or reuses) a k-means run.
    @Published var colorMode: ColorMode = .frequency {
        didSet {
            guard colorMode != oldValue else { return }
            // Only spend CPU on clustering once the user actually asks to see it. Switching back
            // to Frequency keeps the already-computed clusters cached so a later toggle is instant.
            if colorMode == .clusters {
                requestClusteringIfNeeded()
            }
        }
    }

    /// Requested number of clusters. Changing it re-runs k-means; `clampedTopN` re-derives from the
    /// new cluster count so the top-N focus can't point past the clusters that now exist.
    @Published var clusterK: Int = 8 {
        didSet {
            guard clusterK != oldValue, colorMode == .clusters else { return }
            requestClustering()
        }
    }

    /// How many of the largest clusters keep vivid categorical colors. Smaller clusters are muted
    /// (greyed out) so the user can focus on the dominant themes. Purely a display filter — it
    /// never re-runs clustering, so dragging it is instant; it only rebuilds the legend/visibility.
    @Published var topN: Int = 8 {
        didSet {
            guard topN != oldValue else { return }
            // Changing top-N supersedes any single-cluster isolation, then refreshes the legend.
            isolatedCluster = nil
            rebuildLegend()
        }
    }

    /// When set, the legend row the user tapped is shown in isolation: only that cluster stays
    /// vivid and all others mute, regardless of top-N. Tapping the same row again clears it.
    @Published private(set) var isolatedCluster: Int?

    @Published private(set) var isClustering = false

    /// Word → cluster index for the current clustering result, in vocabulary order. Looked up by
    /// word (not position) so plotted points and cluster labels stay aligned even if either list
    /// is ever reordered. Empty until the first clustering run completes.
    @Published private(set) var clusterByWord: [String: Int] = [:]

    /// Legend rows for the current clustering result, already sorted largest-cluster-first and
    /// tagged with their current visibility under the top-N / isolation filter.
    @Published private(set) var clusterLegend: [ClusterLegendEntry] = []

    /// The palette's categorical hues for the currently visible clusters, keyed by cluster index,
    /// ready to hand to `.chartForegroundStyleScale(mapping:)`. Muted clusters are absent here and
    /// fall back to a neutral grey in the chart.
    private(set) var clusterCount = 0

    private var embeddings: WordEmbeddings?
    private var projectionGeneration = 0

    /// The in-flight projection task, retained so a newer request can cancel a superseded one
    /// before it starts its (uninterruptible) SVD, rather than letting several large PCA runs
    /// stack up on the background cores. The generation guard still protects correctness; this
    /// only reclaims wasted CPU when projections are requested in quick succession.
    private var projectionTask: Task<Void, Never>?

    /// Clustering has its own generation counter and in-flight task, mirroring the projection
    /// pattern exactly: a model-state change or a newer k bumps the generation so a slower,
    /// superseded k-means run can never overwrite fresher clusters, and the task is cancelled so
    /// back-to-back k changes don't stack Lloyd's iterations on the background cores.
    private var clusteringGeneration = 0
    private var clusteringTask: Task<Void, Never>?

    /// The word count that the current `clusterResult` was computed for. Clustering re-runs when
    /// this no longer matches the plotted word count, so the assignments always cover exactly the
    /// points on screen.
    private var clusteredWordCount = 0

    /// The most recent completed clustering result, retained so the legend and per-point visibility
    /// can be recomputed when top-N or isolation changes without re-running k-means.
    private var clusterResult: ClusteringResult?

    init() {
        ModelStore.shared.addObserver(self) { [weak self] state in
            self?.render(state)
        }
    }

    var selectedPoint: ProjectedWord? {
        guard let selectedWord else { return nil }
        return points.first { $0.word == selectedWord }
    }

    /// Normalizes the text-field input to match the lowercased training corpus before lookup.
    func submitSearch() {
        let word = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !word.isEmpty else {
            selectedWord = nil
            neighbors = []
            searchMessage = "Type a word to search."
            return
        }

        query = word
        select(word: word)
    }

    /// Clears the current selection when the user taps empty space in the graph. The query text
    /// is left as-is so a mistaken tap doesn't wipe out what they typed, but the highlighted
    /// point, neighbor list, and message all reset to the neutral prompt.
    func deselect() {
        guard selectedWord != nil else { return }
        selectedWord = nil
        neighbors = []
        searchMessage = "Select a point or search the vocabulary."
    }

    /// Updates selection and its neighbor list together for graph, search, and list actions.
    func select(word: String) {
        guard let embeddings else {
            selectedWord = nil
            neighbors = []
            searchMessage = "Train a model first (see the Train tab)."
            return
        }

        guard embeddings.contains(word) else {
            appLog.info("Explore query '\(word, privacy: .public)': out of vocabulary.")
            selectedWord = nil
            neighbors = []
            searchMessage = "'\(word)' is not in the vocabulary. Try a more common word."
            return
        }

        query = word
        selectedWord = word
        neighbors = embeddings.nearest(to: word, count: 10, metric: metric).map {
            NeighborResult(word: $0.word, score: $0.score)
        }
        appLog.info("Explore query '\(word, privacy: .public)' [\(self.metric.rawValue, privacy: .public)]: \(self.neighbors.count, privacy: .public) results.")
        searchMessage = neighbors.isEmpty
            ? "No neighbours found for '\(word)'."
            : "Nearest words to '\(word)':"
    }

    /// Re-ranks the current selection under the newly-chosen metric without changing what's
    /// selected. A no-op when nothing is selected (there are no neighbors to re-rank).
    private func rerankCurrentSelection() {
        guard let selectedWord else { return }
        select(word: selectedWord)
    }

    private func render(_ state: ModelState) {
        switch state {
        case let .ready(embeddings):
            let vocabularyCount = embeddings.vocabulary.count
            guard vocabularyCount > 0 else {
                clearReadyModel(message: "The trained model has no words to explore. Try training again.")
                return
            }

            self.embeddings = embeddings
            self.vocabularyCount = vocabularyCount
            points = []
            selectedWord = nil
            neighbors = []
            searchMessage = "Select a point or search the vocabulary."
            requestProjection(embeddings: embeddings, wordCount: vocabularyCount)
            // A new model invalidates any clusters computed for the old one. Discard them and,
            // only if the user is currently viewing clusters, kick off a fresh run for this model.
            invalidateClustering()
            if colorMode == .clusters {
                requestClustering()
            }

        case .idle:
            clearReadyModel(message: "Train a model first (see the Train tab).")
        case .loading:
            clearReadyModel(message: "Loading your saved model…")
        case let .training(progress):
            clearReadyModel(message: "Training in progress on the Train tab… \(progress.formatted(.percent.precision(.fractionLength(0))))")
        case .failed:
            clearReadyModel(message: "The model couldn’t be prepared. Return to Train and try again.")
        }
    }

    private func clearReadyModel(message: String) {
        projectionGeneration += 1
        embeddings = nil
        vocabularyCount = 0
        points = []
        selectedWord = nil
        neighbors = []
        isProjecting = false
        emptyMessage = message
        searchMessage = message
        invalidateClustering()
    }

    private func requestProjection(embeddings: WordEmbeddings, wordCount: Int) {
        projectionGeneration += 1
        let generation = projectionGeneration
        let request = ProjectionRequest(embeddings: embeddings, wordCount: wordCount)
        isProjecting = true

        // Supersede any still-pending projection so back-to-back requests don't stack SVDs.
        projectionTask?.cancel()
        projectionTask = Task { [weak self] in
            let projected = await Task.detached(priority: .userInitiated) { () -> [ProjectedWord] in
                // If this request was superseded before the detached work started, skip the
                // uninterruptible SVD entirely — its result would be dropped by the guard below.
                if Task.isCancelled { return [] }
                return request.embeddings.projected2D(wordCount: request.wordCount).enumerated().map { rank, point in
                    ProjectedWord(word: point.word, x: point.x, y: point.y, rank: rank)
                }
            }.value

            // PCA itself cannot be interrupted, but a model-state change invalidates its
            // generation so an older completion can never overwrite the current model.
            guard let self, generation == self.projectionGeneration else { return }
            self.points = projected
            self.isProjecting = false

            // Cluster assignments cover exactly the plotted words. If the plotted word count just
            // changed (a new model, typically) and the user is on Clusters mode, re-cluster so the
            // colors match the points now on screen.
            if self.colorMode == .clusters, self.clusteredWordCount != self.points.count {
                self.requestClustering()
            }
        }
    }

    // MARK: - Clustering

    /// A fixed seed so the same k and model always produce the same clusters. Recomputing (after a
    /// toggle, a k bump and back, etc.) then lands on identical colors and labels, which keeps the
    /// legend stable and stops clusters from "shuffling" under the user.
    private static let clusteringSeed: UInt64 = 42

    /// How many representative words the engine returns per cluster. Six comfortably fills a single
    /// legend line ("spaceship · planet · alien · robot · rocket · orbit") without wrapping.
    private static let labelsPerCluster = 6

    /// The desired top-N clamped to the clusters that actually exist, so the focus control can
    /// never point past the current cluster count (which may be < requested k on a tiny vocab).
    private var clampedTopN: Int {
        guard clusterCount > 0 else { return 0 }
        return min(max(topN, 1), clusterCount)
    }

    /// Kicks off clustering only when we don't already have a usable result for the current k and
    /// plotted word count — used when the user first switches into Clusters mode so a cached result
    /// (from an earlier toggle) is reused instead of recomputed.
    func requestClusteringIfNeeded() {
        guard clusterResult == nil || clusteredWordCount != points.count else { return }
        requestClustering()
    }

    /// Clears every clustering-derived value and bumps the generation so any in-flight k-means run
    /// is disowned. Called whenever the model changes; the generation bump is what makes a slow,
    /// superseded run drop its result on completion.
    private func invalidateClustering() {
        clusteringGeneration += 1
        clusteringTask?.cancel()
        clusteringTask = nil
        clusterResult = nil
        clusterByWord = [:]
        clusterLegend = []
        clusterCount = 0
        clusteredWordCount = 0
        isClustering = false
        isolatedCluster = nil
    }

    /// Runs k-means off the main actor for the current k over the plotted words, mirroring
    /// `requestProjection` beat for beat: bump-and-capture a generation, wrap the model in a
    /// sendable request, supersede any pending run, and apply the result only if its generation is
    /// still current. The seed is fixed so the result is deterministic for a given k and model.
    private func requestClustering() {
        guard let embeddings, !points.isEmpty else { return }

        let wordCount = points.count
        let k = clusterK
        clusteringGeneration += 1
        let generation = clusteringGeneration
        let request = ClusteringRequest(embeddings: embeddings, k: k, wordCount: wordCount)
        isClustering = true

        // Supersede any still-pending clustering so back-to-back k changes don't stack Lloyd's runs.
        clusteringTask?.cancel()
        clusteringTask = Task { [weak self] in
            let result = await Task.detached(priority: .userInitiated) { () -> ClusteringResult? in
                // If a newer k (or model) already superseded this request before the detached work
                // began, skip the k-means entirely — the guard below would drop its result anyway.
                if Task.isCancelled { return nil }
                return request.embeddings.cluster(
                    k: request.k,
                    wordCount: request.wordCount,
                    labelsPerCluster: GraphViewModel.labelsPerCluster,
                    seed: GraphViewModel.clusteringSeed
                )
            }.value

            // k-means cannot be interrupted mid-run, but a stale generation means either the model
            // changed or a newer k landed first, so an older completion must never overwrite it.
            guard let self, generation == self.clusteringGeneration, let result else { return }
            self.applyClustering(result, wordCount: wordCount)
        }
    }

    /// Stores a completed clustering result and derives the word→cluster map. Top-N is re-clamped
    /// to the new cluster count and any stale isolation is cleared, then the legend and per-point
    /// visibility are rebuilt.
    private func applyClustering(_ result: ClusteringResult, wordCount: Int) {
        clusterResult = result
        clusterCount = result.clusterCount
        clusteredWordCount = wordCount
        clusterByWord = Dictionary(
            result.assignments.map { ($0.word, $0.cluster) },
            uniquingKeysWith: { first, _ in first }
        )
        // An isolated cluster from a previous k may no longer exist; drop it rather than isolate a
        // cluster index that's now out of range.
        if let isolated = isolatedCluster, isolated >= result.clusterCount {
            isolatedCluster = nil
        }
        isClustering = false
        rebuildLegend()
    }

    /// Toggles single-cluster isolation from a legend tap. Tapping the isolated row again clears it.
    /// Isolation is purely a display filter, so it only rebuilds the legend — no re-clustering.
    func toggleIsolation(of cluster: Int) {
        guard colorMode == .clusters else { return }
        isolatedCluster = (isolatedCluster == cluster) ? nil : cluster
        rebuildLegend()
    }

    /// Recomputes which clusters are visible and rebuilds the legend rows (largest cluster first).
    /// Called after clustering completes and after any top-N or isolation change. Kept separate
    /// from clustering so dragging the top-N stepper stays instant.
    func rebuildLegend() {
        guard let clusterResult else {
            clusterLegend = []
            return
        }

        let visible = visibleClusterSet(for: clusterResult)
        // Order clusters largest-first (ties break toward the lower index) so the most prominent
        // themes head the legend and align with what "top N" highlights.
        let order = (0..<clusterResult.clusterCount).sorted { lhs, rhs in
            let sizeLHS = clusterResult.sizes[lhs]
            let sizeRHS = clusterResult.sizes[rhs]
            if sizeLHS != sizeRHS { return sizeLHS > sizeRHS }
            return lhs < rhs
        }

        clusterLegend = order.map { cluster in
            ClusterLegendEntry(
                cluster: cluster,
                representatives: clusterResult.representatives[cluster],
                size: clusterResult.sizes[cluster],
                isVisible: visible.contains(cluster)
            )
        }
    }

    /// The set of clusters that get vivid colors. Isolation wins outright (exactly one cluster);
    /// otherwise it's the `clampedTopN` largest clusters by size (lower index breaks ties, matching
    /// the legend order). Everything else is muted in the plot.
    func visibleClusterSet(for result: ClusteringResult) -> Set<Int> {
        if let isolatedCluster, isolatedCluster < result.clusterCount {
            return [isolatedCluster]
        }
        let largestFirst = (0..<result.clusterCount).sorted { lhs, rhs in
            let sizeLHS = result.sizes[lhs]
            let sizeRHS = result.sizes[rhs]
            if sizeLHS != sizeRHS { return sizeLHS > sizeRHS }
            return lhs < rhs
        }
        return Set(largestFirst.prefix(clampedTopN))
    }

    /// The cluster index for a plotted word, or nil if the word isn't in the current result (or
    /// clustering hasn't finished yet). Used by the chart to color each point.
    func cluster(for word: String) -> Int? {
        clusterByWord[word]
    }

    /// Whether a word's cluster is currently in the vivid (visible) set. Words in muted clusters —
    /// or words with no cluster yet — render greyed out so the top-N focus reads clearly.
    func isWordVisible(_ word: String) -> Bool {
        guard let cluster = clusterByWord[word], let clusterResult else { return false }
        return visibleClusterSet(for: clusterResult).contains(cluster)
    }

    /// Whether a cluster index is in the vivid (visible) set. Drives the chart's per-cluster color
    /// (palette hue when visible, muted grey otherwise) and the legend swatch styling.
    func isClusterVisible(_ cluster: Int) -> Bool {
        guard let clusterResult else { return false }
        return visibleClusterSet(for: clusterResult).contains(cluster)
    }
}

/// SwiftUI presentation of full-vocabulary projection and nearest-neighbor exploration.
struct GraphView: View {
    @StateObject private var viewModel = GraphViewModel()

    /// A categorical palette of ~12 distinct hues for cluster coloring. Chosen to stay reasonably
    /// distinguishable for the common red–green color-vision deficiencies: it leans on the
    /// blue/orange/purple/brown axis and avoids placing a pure red next to a pure green. The chart
    /// cycles through it by cluster index, so cluster counts beyond the palette length simply reuse
    /// hues from the top — acceptable because at most `topN` clusters are ever vivid at once.
    static let clusterPalette: [Color] = [
        Color(red: 0.12, green: 0.47, blue: 0.71), // blue
        Color(red: 1.00, green: 0.50, blue: 0.05), // orange
        Color(red: 0.17, green: 0.63, blue: 0.17), // green
        Color(red: 0.58, green: 0.40, blue: 0.74), // purple
        Color(red: 0.55, green: 0.34, blue: 0.29), // brown
        Color(red: 0.89, green: 0.47, blue: 0.76), // pink
        Color(red: 0.09, green: 0.75, blue: 0.81), // cyan
        Color(red: 0.74, green: 0.74, blue: 0.13), // olive
        Color(red: 0.84, green: 0.15, blue: 0.16), // red
        Color(red: 0.40, green: 0.40, blue: 0.85), // indigo
        Color(red: 0.99, green: 0.75, blue: 0.18), // amber
        Color(red: 0.30, green: 0.69, blue: 0.55), // teal
    ]

    /// The muted color for points whose cluster isn't in the current vivid set. A single low-alpha
    /// grey (rather than a faded version of each hue) makes the top-N focus unmistakable: focused
    /// clusters pop in color, everything else recedes into a neutral haze.
    static let mutedClusterColor = Color.secondary.opacity(0.18)

    /// The palette hue for a cluster index, cycling when the count exceeds the palette length.
    static func clusterColor(_ cluster: Int) -> Color {
        clusterPalette[cluster % clusterPalette.count]
    }

    var body: some View {
        Group {
            if viewModel.vocabularyCount == 0 {
                ContentUnavailableView(
                    "No Model to Explore",
                    systemImage: "chart.dots.scatter",
                    description: Text(viewModel.emptyMessage)
                )
            } else {
                exploreContent
            }
        }
        .background(Color(uiColor: .systemBackground))
    }

    private var exploreContent: some View {
        VStack(spacing: 12) {
            chartHeader

            ZStack {
                scatterChart

                if viewModel.points.isEmpty, viewModel.isProjecting {
                    ProgressView("Projecting all vectors…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
            // The plot takes a little over half of the tab's height (with a comfortable floor so
            // it stays readable on short windows), leaving the rest for the neighbor list below.
            .frame(maxWidth: .infinity)
            .containerRelativeFrame(.vertical) { height, _ in max(360, height * 0.55) }

            Divider()

            // The color-mode toggle sits above the mode-specific controls so it reads as the
            // switch that governs everything below it.
            colorModePicker

            // Frequency mode keeps the original search + neighbor experience untouched; Clusters
            // mode swaps in the k / top-N controls and the cluster legend.
            switch viewModel.colorMode {
            case .frequency:
                frequencySection
            case .clusters:
                clustersSection
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    /// The original Frequency-mode lower half: search field, metric picker, and neighbor list.
    /// Extracted verbatim so switching modes swaps this whole block in and out without disturbing
    /// any of its behavior.
    private var frequencySection: some View {
        VStack(spacing: 12) {
            searchField

            metricPicker

            HStack {
                Text(viewModel.searchMessage)
                    .font(.callout)
                    .foregroundStyle(.secondary)

                Spacer()

                // Header for the score column, adapting to the metric ("Score" vs "Distance").
                Text(viewModel.metric.scoreColumnTitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // The neighbor list fills whatever height remains and scrolls when the results
            // don't fit, rather than being clipped or pushing the plot off-screen.
            neighborList
                .frame(maxHeight: .infinity)
        }
    }

    /// The Clusters-mode lower half: the k and top-N controls, plus the cluster legend that names
    /// each visible cluster by its representative words. Fills the remaining height like the
    /// neighbor list does in Frequency mode.
    private var clustersSection: some View {
        VStack(spacing: 12) {
            clusterControls
            clusterLegendList
                .frame(maxHeight: .infinity)
        }
    }

    private var chartHeader: some View {
        HStack(spacing: 10) {
            Text("\(viewModel.vocabularyCount.formatted()) words")
                .font(.headline)
                .contentTransition(.numericText())

            Spacer()

            if viewModel.isProjecting {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel("Projecting word vectors")
            }
        }
    }

    /// The scatter plot. Frequency and Clusters mode need different color scales — a continuous
    /// gradient keyed by rank vs. a discrete categorical scale keyed by cluster — and Swift Charts
    /// resolves `chartForegroundStyleScale` per chart, so the two are built as separate `Chart`s.
    /// Everything they share (axes, plot style, tap overlay, accessibility) is applied by
    /// `sharedChartStyle` so the two variants can't drift apart.
    @ViewBuilder
    private var scatterChart: some View {
        Group {
            switch viewModel.colorMode {
            case .frequency:
                frequencyChart.modifier(sharedChartStyle)
            case .clusters:
                clustersChart.modifier(sharedChartStyle)
            }
        }
        .accessibilityValue("\(viewModel.points.count) projected words")
    }

    /// Frequency mode — unchanged from the original: points colored by frequency rank along the
    /// orange→faded-accent gradient, with the accent-highlighted selection on top.
    private var frequencyChart: some View {
        Chart {
            ForEach(viewModel.points) { point in
                PointMark(
                    x: .value("PCA dimension 1", point.x),
                    y: .value("PCA dimension 2", point.y)
                )
                .foregroundStyle(by: .value("Frequency rank", point.rank))
                .symbolSize(18)
                .opacity(0.7)
            }

            selectionMark
        }
        .chartForegroundStyleScale(
            range: Gradient(colors: [
                .orange,
                .accentColor.opacity(0.48),
            ])
        )
    }

    /// Clusters mode — points colored by a discrete categorical scale keyed on the cluster label.
    /// Each point's `by:` value is a stable per-cluster string ("Cluster 3"); `chartForegroundStyleScale`
    /// then maps every such label to its palette hue, or to the muted grey when the cluster isn't in
    /// the current top-N / isolation focus. Points with no assignment yet (clustering still running)
    /// also map to the muted color. This is a true categorical scale, not the frequency gradient.
    private var clustersChart: some View {
        Chart {
            ForEach(viewModel.points) { point in
                PointMark(
                    x: .value("PCA dimension 1", point.x),
                    y: .value("PCA dimension 2", point.y)
                )
                .foregroundStyle(by: .value("Cluster", clusterLabel(for: point.word)))
                // Muted (out-of-focus) points shrink and fade so the focused clusters read clearly;
                // vivid points keep the same footprint as Frequency mode.
                .symbolSize(viewModel.isWordVisible(point.word) ? 18 : 10)
                .opacity(viewModel.isWordVisible(point.word) ? 0.75 : 0.35)
            }

            selectionMark
        }
        // A discrete scale: parallel domain (cluster labels) and range (palette hue or muted grey)
        // arrays, both rebuilt each render so the colors track the current top-N / isolation focus.
        .chartForegroundStyleScale(
            domain: clusterScaleDomain,
            range: clusterScaleRange
        )
    }

    /// The accent-highlighted selection marker, shared by both chart variants so tap-to-select
    /// looks identical regardless of color mode.
    @ChartContentBuilder
    private var selectionMark: some ChartContent {
        if let selectedPoint = viewModel.selectedPoint {
            PointMark(
                x: .value("Selected PCA dimension 1", selectedPoint.x),
                y: .value("Selected PCA dimension 2", selectedPoint.y)
            )
            .foregroundStyle(Color.accentColor)
            .symbolSize(120)
            .annotation(position: .top, spacing: 6) {
                // The label's colors are set explicitly rather than inherited: the mark's
                // accentColor foregroundStyle otherwise cascades into the annotation and
                // renders the word as an unreadable dark blob. A solid accent capsule with a
                // white-on-accent label stays legible in both light and dark mode.
                Text(selectedPoint.word)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.accentColor, in: Capsule())
            }
        }
    }

    /// The categorical label a point maps to in Clusters mode. A point with no assignment yet gets
    /// a distinct "unclustered" domain value so the mapping can render it muted rather than crash
    /// on a missing domain entry.
    private func clusterLabel(for word: String) -> String {
        guard let cluster = viewModel.cluster(for: word) else { return "—" }
        return "Cluster \(cluster)"
    }

    /// The ordered domain of the categorical color scale: one label per cluster, plus the "—"
    /// sentinel for as-yet-unassigned points. The order must stay in lock-step with
    /// `clusterScaleRange` (same index → matching color), so both derive from the same 0..<count
    /// sequence with the sentinel appended last.
    private var clusterScaleDomain: [String] {
        var domain = (0..<viewModel.clusterCount).map { "Cluster \($0)" }
        domain.append("—")
        return domain
    }

    /// The color for each domain entry, index-aligned with `clusterScaleDomain`: a cluster gets its
    /// palette hue when it's in the vivid (focused) set and the muted grey otherwise; the trailing
    /// entry (the "—" sentinel) is always muted. Rebuilt each render so it follows the current
    /// top-N / isolation focus.
    private var clusterScaleRange: [Color] {
        var range = (0..<viewModel.clusterCount).map { cluster -> Color in
            viewModel.isClusterVisible(cluster)
                ? Self.clusterColor(cluster)
                : Self.mutedClusterColor
        }
        range.append(Self.mutedClusterColor)
        return range
    }

    /// Everything the two chart variants share: hidden built-in legend (both modes present their
    /// own — the frequency gradient is self-evident, clusters get the swatch legend below), the
    /// subtle axes and plot background, the tap-to-select overlay, and accessibility. Factored into
    /// one modifier so the Frequency and Clusters charts can never drift apart on these.
    private var sharedChartStyle: some ViewModifier { SharedChartStyle(handleTap: handleTap) }

    /// Carries the shared chart chrome. Takes the tap handler as a closure so the same overlay
    /// gesture drives selection identically in both color modes.
    private struct SharedChartStyle: ViewModifier {
        let handleTap: (CGPoint, ChartProxy, GeometryProxy) -> Void

        func body(content: Content) -> some View {
            content
                .chartLegend(.hidden)
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 5)) { _ in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                            .foregroundStyle(Color.secondary.opacity(0.16))
                        AxisTick()
                            .foregroundStyle(Color.secondary.opacity(0.3))
                        AxisValueLabel()
                            .foregroundStyle(.secondary)
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .automatic(desiredCount: 5)) { _ in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                            .foregroundStyle(Color.secondary.opacity(0.16))
                        AxisTick()
                            .foregroundStyle(Color.secondary.opacity(0.3))
                        AxisValueLabel()
                            .foregroundStyle(.secondary)
                    }
                }
                .chartPlotStyle { plotArea in
                    plotArea
                        .background(Color.secondary.opacity(0.045))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .chartOverlay { proxy in
                    GeometryReader { geometry in
                        Rectangle()
                            .fill(.clear)
                            .contentShape(Rectangle())
                            .gesture(
                                SpatialTapGesture()
                                    .onEnded { value in
                                        handleTap(value.location, proxy, geometry)
                                    }
                            )
                    }
                }
                .accessibilityLabel("Word vector scatter plot")
        }
    }

    /// A tap must land within this many points of a plotted marker to select it. Tapping the
    /// empty space beyond that radius deselects instead, so the graph reads as "nothing here"
    /// rather than snapping to some far-off point the user never aimed at. Sized for a fingertip,
    /// not the ~5pt marker: at a tighter radius a tap that visibly lands on the point cloud can
    /// still miss the nearest actual dot by 20–30pt, so it must be forgiving enough to feel like
    /// "tap near a word" while the wide empty bands above/below the streak still read as deselect.
    private static let selectionHitRadius: CGFloat = 28

    /// Resolves a tap in the chart overlay to either the nearest nearby word or a deselection.
    ///
    /// The hit-test is done in SCREEN space, not data space: the x and y axes span very
    /// different data ranges (PC1 ≈ ±1, PC2 ≈ ±0.05), so a fixed data-space radius would be an
    /// ellipse on screen. Projecting each point back to plot pixels via the chart proxy makes
    /// the radius a true circle in points, matching what the user sees under their finger.
    private func handleTap(at location: CGPoint, proxy: ChartProxy, geometry: GeometryProxy) {
        guard let plotFrame = proxy.plotFrame else { return }
        let frame = geometry[plotFrame]
        guard frame.contains(location) else { return }

        // `proxy.position(forX:)` returns coordinates relative to the plot area's origin, so
        // convert the tap (which arrives in the overlay's space) to the same plot-relative
        // space by subtracting the plot origin. Comparing both in one space is what keeps the
        // hit-test honest — an earlier version offset only one side and every point read ~16pt
        // off, so taps on the visible streak found nothing within the radius.
        let tap = CGPoint(x: location.x - frame.minX, y: location.y - frame.minY)

        // Find the plotted point whose on-screen position is closest to the tap.
        var nearestWord: String?
        var nearestDistanceSquared = CGFloat.greatestFiniteMagnitude
        for point in viewModel.points {
            guard let px = proxy.position(forX: point.x),
                  let py = proxy.position(forY: point.y) else { continue }
            let dx = px - tap.x
            let dy = py - tap.y
            let distanceSquared = dx * dx + dy * dy
            if distanceSquared < nearestDistanceSquared {
                nearestDistanceSquared = distanceSquared
                nearestWord = point.word
            }
        }

        // Within the hit radius, select that word; otherwise the tap was on empty space.
        let radius = Self.selectionHitRadius
        if let nearestWord, nearestDistanceSquared <= radius * radius {
            viewModel.select(word: nearestWord)
        } else {
            viewModel.deselect()
        }
    }

    /// Chooses how the plot colors its points. Bound to the view model's `colorMode`, whose `didSet`
    /// kicks off clustering the first time the user switches to Clusters. Frequency is the default,
    /// so the tab looks and behaves exactly as before until the user opts in.
    private var colorModePicker: some View {
        Picker("Color mode", selection: $viewModel.colorMode) {
            ForEach(ColorMode.allCases) { mode in
                Text(mode.displayName).tag(mode)
            }
        }
        .pickerStyle(.segmented)
    }

    /// The Clusters-mode knobs: a Stepper for k (4…12) that re-runs clustering, and a Stepper for
    /// how many of the largest clusters stay vivid (the rest grey out). The top-N stepper is capped
    /// at the actual cluster count so it can't over-run a small result. A spinner shows while
    /// k-means runs off the main actor.
    private var clusterControls: some View {
        VStack(spacing: 10) {
            HStack {
                Stepper(value: $viewModel.clusterK, in: 4...12) {
                    HStack(spacing: 6) {
                        Text("Clusters")
                        Text("\(viewModel.clusterK)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .contentTransition(.numericText())
                    }
                    .font(.callout)
                }

                if viewModel.isClustering {
                    ProgressView()
                        .controlSize(.small)
                        .padding(.leading, 8)
                        .accessibilityLabel("Clustering word vectors")
                }
            }

            // Top-N only makes sense once there are clusters to rank; while the first run is still
            // in flight (clusterCount == 0) the control would have an empty range, so hide it.
            if viewModel.clusterCount > 0 {
                Stepper(value: $viewModel.topN, in: 1...max(1, viewModel.clusterCount)) {
                    HStack(spacing: 6) {
                        Text("Show top")
                        Text("\(min(viewModel.topN, viewModel.clusterCount))")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .contentTransition(.numericText())
                        Text(min(viewModel.topN, viewModel.clusterCount) == 1 ? "cluster" : "clusters")
                            .foregroundStyle(.secondary)
                    }
                    .font(.callout)
                }
            }
        }
    }

    /// The cluster legend: one tappable row per cluster (largest first) showing a color swatch, its
    /// representative words ("spaceship · planet · alien"), and its size. This is what makes the
    /// coloring readable — it names each cluster. Tapping a row isolates that single cluster in the
    /// plot (tap again to clear); rows outside the current top-N focus render muted to mirror the
    /// plot. Sits where the neighbor list sits in Frequency mode.
    private var clusterLegendList: some View {
        List {
            ForEach(viewModel.clusterLegend) { entry in
                Button {
                    viewModel.toggleIsolation(of: entry.cluster)
                } label: {
                    HStack(spacing: 12) {
                        // Swatch uses the same palette hue as the plot when the cluster is focused,
                        // and greys out in lock-step when it's muted, so the legend always matches
                        // what the eye sees on the chart.
                        RoundedRectangle(cornerRadius: 3)
                            .fill(entry.isVisible ? Self.clusterColor(entry.cluster) : Self.mutedClusterColor)
                            .frame(width: 14, height: 14)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.label)
                                .font(.callout)
                                .foregroundStyle(entry.isVisible ? .primary : .secondary)
                                .lineLimit(2)

                            Text("\(entry.size) \(entry.size == 1 ? "word" : "words")")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }

                        Spacer()

                        // A checkmark marks the currently-isolated cluster so the toggle's state is
                        // obvious even though the row's own coloring already hints at focus.
                        if viewModel.isolatedCluster == entry.cluster {
                            Image(systemName: "scope")
                                .foregroundStyle(.tint)
                                .accessibilityLabel("Isolated")
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .overlay {
            // Before the first clustering result lands, the legend is empty; show the same kind of
            // inline hint the rest of the tab uses rather than a blank list.
            if viewModel.clusterLegend.isEmpty {
                Text(viewModel.isClustering ? "Finding clusters…" : "Switch to Clusters to group the vocabulary.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    /// Lets the user rank neighbors by cosine, dot product, or Euclidean distance. Bound to the
    /// view model's `metric`, whose `didSet` re-ranks the current selection automatically.
    private var metricPicker: some View {
        Picker("Distance metric", selection: $viewModel.metric) {
            ForEach(DistanceMetric.allCases, id: \.self) { metric in
                Text(metric.displayName).tag(metric)
            }
        }
        .pickerStyle(.segmented)
    }

    private var searchField: some View {
        HStack(spacing: 10) {
            TextField("Search vocabulary", text: $viewModel.query)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .submitLabel(.search)
                .onSubmit(viewModel.submitSearch)

            Button(action: viewModel.submitSearch) {
                Label("Search", systemImage: "magnifyingglass")
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var neighborList: some View {
        List {
            ForEach(Array(viewModel.neighbors.enumerated()), id: \.element.id) { index, neighbor in
                Button {
                    viewModel.select(word: neighbor.word)
                } label: {
                    HStack(spacing: 12) {
                        Text("\(index + 1).")
                            .foregroundStyle(.secondary)
                            .frame(width: 24, alignment: .trailing)

                        Text(neighbor.word)
                            .foregroundStyle(.primary)

                        Spacer()

                        Text(neighbor.score.formatted(.number.precision(.fractionLength(3))))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
    }
}
