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

/// A display-friendly nearest-neighbor result.
nonisolated private struct NeighborResult: Identifiable, Sendable {
    let word: String
    let similarity: Float

    var id: String { word }
}

/// A sendable projection job. `WordEmbeddings` is immutable after initialization, so it is
/// safe for the detached task to read even though the package does not currently declare the
/// reference type `Sendable`. Keeping that unchecked boundary here makes the concurrency
/// assumption narrow and explicit.
nonisolated private struct ProjectionRequest: @unchecked Sendable {
    let embeddings: WordEmbeddings
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

    private var embeddings: WordEmbeddings?
    private var projectionGeneration = 0

    /// The in-flight projection task, retained so a newer request can cancel a superseded one
    /// before it starts its (uninterruptible) SVD, rather than letting several large PCA runs
    /// stack up on the background cores. The generation guard still protects correctness; this
    /// only reclaims wasted CPU when projections are requested in quick succession.
    private var projectionTask: Task<Void, Never>?

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

    /// Selects the plotted point closest to a tap expressed in the chart's data coordinates.
    func selectClosestPoint(x: Float, y: Float) {
        guard let nearestPoint = points.min(by: { lhs, rhs in
            let lhsDistance = squaredDistance(from: lhs, toX: x, y: y)
            let rhsDistance = squaredDistance(from: rhs, toX: x, y: y)
            return lhsDistance < rhsDistance
        }) else { return }

        select(word: nearestPoint.word)
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
        neighbors = embeddings.nearest(to: word, count: 10).map {
            NeighborResult(word: $0.word, similarity: $0.similarity)
        }
        appLog.info("Explore query '\(word, privacy: .public)': \(self.neighbors.count, privacy: .public) results.")
        searchMessage = neighbors.isEmpty
            ? "No neighbours found for '\(word)'."
            : "Nearest words to '\(word)':"
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
        }
    }

    private func squaredDistance(from point: ProjectedWord, toX x: Float, y: Float) -> Float {
        let deltaX = point.x - x
        let deltaY = point.y - y
        return deltaX * deltaX + deltaY * deltaY
    }
}

/// SwiftUI presentation of full-vocabulary projection and nearest-neighbor exploration.
struct GraphView: View {
    @StateObject private var viewModel = GraphViewModel()

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
            .frame(maxWidth: .infinity)
            .frame(height: 280)

            Divider()

            searchField

            Text(viewModel.searchMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)

            neighborList
        }
        .padding(.horizontal, 16)
        .padding(.top, 12)
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

    private var scatterChart: some View {
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

            if let selectedPoint = viewModel.selectedPoint {
                PointMark(
                    x: .value("Selected PCA dimension 1", selectedPoint.x),
                    y: .value("Selected PCA dimension 2", selectedPoint.y)
                )
                .foregroundStyle(Color.accentColor)
                .symbolSize(100)
                .annotation(position: .top, spacing: 6) {
                    Text(selectedPoint.word)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 7)
                        .padding(.vertical, 4)
                        .background(.regularMaterial, in: Capsule())
                }
            }
        }
        .chartForegroundStyleScale(
            range: Gradient(colors: [
                .orange,
                .accentColor.opacity(0.48),
            ])
        )
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
                                guard let plotFrame = proxy.plotFrame else { return }
                                let frame = geometry[plotFrame]
                                guard frame.contains(value.location) else { return }
                                let plotLocation = CGPoint(
                                    x: value.location.x - frame.minX,
                                    y: value.location.y - frame.minY
                                )
                                guard let x = proxy.value(atX: plotLocation.x, as: Float.self),
                                      let y = proxy.value(atY: plotLocation.y, as: Float.self)
                                else { return }
                                viewModel.selectClosestPoint(x: x, y: y)
                            }
                    )
            }
        }
        .accessibilityLabel("Word vector scatter plot")
        .accessibilityValue("\(viewModel.points.count) projected words")
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

                        Text(neighbor.similarity.formatted(.number.precision(.fractionLength(3))))
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
