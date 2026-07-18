//
//  GraphView.swift
//  WordVectors
//
//  Feature D — A frequency-ranked scatter plot of the learned vocabulary. PCA is
//  intentionally performed away from the main actor because projecting a large
//  vocabulary is CPU-intensive enough to interrupt animations and slider input.
//

import Charts
import Combine
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
/// The project defaults UI code to `MainActor`. Only the expensive PCA call crosses away from
/// it; model changes, projection publication, and stale-result rejection all happen here.
@MainActor
private final class GraphViewModel: ObservableObject {
    @Published private(set) var vocabularyCount = 0
    @Published private(set) var countOptions: [Int] = []
    @Published private(set) var points: [ProjectedWord] = []
    @Published private(set) var isProjecting = false
    @Published private(set) var emptyMessage = "Train a model first (see the Train tab)."
    @Published var sliderPosition = 0.0

    private var embeddings: WordEmbeddings?
    private var projectionGeneration = 0

    init() {
        ModelStore.shared.addObserver(self) { [weak self] state in
            self?.render(state)
        }
    }

    var selectedWordCount: Int {
        guard !countOptions.isEmpty else { return 0 }
        let index = min(max(Int(sliderPosition.rounded()), 0), countOptions.count - 1)
        return countOptions[index]
    }

    /// Starts work only when the user releases the slider, never for its intermediate frames.
    func sliderEditingChanged(_ isEditing: Bool) {
        guard !isEditing, let embeddings else { return }
        requestProjection(embeddings: embeddings, wordCount: selectedWordCount)
    }

    private func render(_ state: ModelState) {
        switch state {
        case let .ready(embeddings):
            let vocabularyCount = embeddings.vocabulary.count
            guard vocabularyCount > 0 else {
                clearReadyModel(message: "The trained model has no words to graph. Try training again.")
                return
            }

            self.embeddings = embeddings
            self.vocabularyCount = vocabularyCount
            countOptions = Self.makeCountOptions(vocabularyCount: vocabularyCount)
            sliderPosition = 0
            points = []
            requestProjection(embeddings: embeddings, wordCount: countOptions[0])

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
        countOptions = []
        sliderPosition = 0
        points = []
        isProjecting = false
        emptyMessage = message
    }

    private func requestProjection(embeddings: WordEmbeddings, wordCount: Int) {
        projectionGeneration += 1
        let generation = projectionGeneration
        let request = ProjectionRequest(embeddings: embeddings, wordCount: wordCount)
        isProjecting = true

        Task { [weak self] in
            let projected = await Task.detached(priority: .userInitiated) {
                request.embeddings.projected2D(wordCount: request.wordCount).enumerated().map { rank, point in
                    ProjectedWord(word: point.word, x: point.x, y: point.y, rank: rank)
                }
            }.value

            // PCA itself cannot be interrupted, but changing the model or moving the slider
            // invalidates its generation so an older completion is harmless.
            guard let self, generation == self.projectionGeneration else { return }
            self.points = projected
            self.isProjecting = false
        }
    }

    /// Slider stops are every 1,000 words plus an exact final stop, making all words reachable
    /// when the vocabulary size is not itself a multiple of 1,000.
    private static func makeCountOptions(vocabularyCount: Int) -> [Int] {
        guard vocabularyCount > 1_000 else { return [vocabularyCount] }

        var options = Array(stride(from: 1_000, through: vocabularyCount, by: 1_000))
        if options.last != vocabularyCount {
            options.append(vocabularyCount)
        }
        return options
    }
}

/// Swift Charts presentation of the model's two-dimensional PCA projection.
struct GraphView: View {
    @StateObject private var viewModel = GraphViewModel()

    private static let wordCountFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.usesGroupingSeparator = true
        return formatter
    }()

    var body: some View {
        Group {
            if viewModel.vocabularyCount == 0 {
                ContentUnavailableView(
                    "No Model to Graph",
                    systemImage: "chart.dots.scatter",
                    description: Text(viewModel.emptyMessage)
                )
            } else {
                graphContent
            }
        }
        .background(Color(uiColor: .systemBackground))
    }

    private var graphContent: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {
                Text("\(formattedWordCount) words")
                    .font(.headline)
                    .contentTransition(.numericText())

                Spacer()

                if viewModel.isProjecting {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Projecting word vectors")
                }
            }

            ZStack {
                scatterChart

                if viewModel.points.isEmpty, viewModel.isProjecting {
                    ProgressView("Projecting vectors…")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            if viewModel.vocabularyCount > 1_000 {
                Slider(
                    value: $viewModel.sliderPosition,
                    in: 0...Double(viewModel.countOptions.count - 1),
                    step: 1,
                    onEditingChanged: viewModel.sliderEditingChanged
                )
                .accessibilityLabel("Number of words")
                .accessibilityValue("\(viewModel.selectedWordCount) words")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var scatterChart: some View {
        Chart(viewModel.points) { point in
            PointMark(
                x: .value("PCA dimension 1", point.x),
                y: .value("PCA dimension 2", point.y)
            )
            .foregroundStyle(by: .value("Frequency rank", point.rank))
            .symbolSize(18)
            .opacity(0.7)
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
        .accessibilityLabel("Word vector scatter plot")
        .accessibilityValue("\(viewModel.points.count) projected words")
    }

    private var formattedWordCount: String {
        Self.wordCountFormatter.string(from: NSNumber(value: viewModel.selectedWordCount))
            ?? String(viewModel.selectedWordCount)
    }
}
