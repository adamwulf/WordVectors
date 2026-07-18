//
//  TrainViewController.swift
//  WordVectors
//
//  Feature A — Train word vectors from the bundled Gutenberg corpus.
//  Lets the user pick how much corpus to use, shows a progress bar driven by the
//  training callback, and reports the resulting model (vocabulary size, time).
//  A cached model loads automatically at launch; "Retrain" clears the cache.
//

import UIKit
import WordVectorKit

final class TrainViewController: UIViewController {

    private let scopeControl = UISegmentedControl(items: CorpusScope.allCases.map { $0.title })
    private let statusLabel = UILabel()
    private let progressView = UIProgressView(progressViewStyle: .default)
    private let progressLabel = UILabel()
    private let trainButton = UIButton(type: .system)
    private let retrainButton = UIButton(type: .system)
    private let detailLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Train"
        view.backgroundColor = .systemBackground
        buildUI()
        ModelStore.shared.addObserver(self) { [weak self] state in
            self?.render(state)
        }
    }

    // MARK: - UI

    private func buildUI() {
        scopeControl.selectedSegmentIndex = CorpusScope.single.rawValue

        statusLabel.font = .preferredFont(forTextStyle: .headline)
        statusLabel.numberOfLines = 0
        statusLabel.textAlignment = .center

        progressLabel.font = .preferredFont(forTextStyle: .footnote)
        progressLabel.textColor = .secondaryLabel
        progressLabel.textAlignment = .center

        progressView.progress = 0

        detailLabel.font = .preferredFont(forTextStyle: .footnote)
        detailLabel.textColor = .secondaryLabel
        detailLabel.numberOfLines = 0
        detailLabel.textAlignment = .center

        var trainConfig = UIButton.Configuration.filled()
        trainConfig.title = "Train Model"
        trainButton.configuration = trainConfig
        trainButton.addTarget(self, action: #selector(trainTapped), for: .touchUpInside)

        var retrainConfig = UIButton.Configuration.gray()
        retrainConfig.title = "Retrain (clear cache)"
        retrainButton.configuration = retrainConfig
        retrainButton.addTarget(self, action: #selector(retrainTapped), for: .touchUpInside)

        let corpusCaption = UILabel()
        corpusCaption.text = "Corpus size (larger = slower, better vectors)"
        corpusCaption.font = .preferredFont(forTextStyle: .footnote)
        corpusCaption.textColor = .secondaryLabel
        corpusCaption.numberOfLines = 0
        corpusCaption.textAlignment = .center

        let stack = UIStackView(arrangedSubviews: [
            statusLabel,
            corpusCaption,
            scopeControl,
            trainButton,
            progressView,
            progressLabel,
            retrainButton,
            detailLabel,
        ])
        stack.axis = .vertical
        stack.spacing = 16
        stack.alignment = .fill
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.setCustomSpacing(24, after: retrainButton)
        view.addSubview(stack)

        NSLayoutConstraint.activate([
            stack.centerYAnchor.constraint(equalTo: view.safeAreaLayoutGuide.centerYAnchor),
            stack.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -24),
        ])
    }

    // MARK: - Actions

    @objc private func trainTapped() {
        let scope = CorpusScope(rawValue: scopeControl.selectedSegmentIndex) ?? .single
        ModelStore.shared.train(scope: scope)
    }

    @objc private func retrainTapped() {
        ModelStore.shared.clearCache()
        let scope = CorpusScope(rawValue: scopeControl.selectedSegmentIndex) ?? .single
        ModelStore.shared.train(scope: scope)
    }

    // MARK: - State rendering

    private func render(_ state: ModelState) {
        switch state {
        case .idle:
            statusLabel.text = "No model yet"
            progressView.isHidden = true
            progressLabel.isHidden = true
            setControlsEnabled(idle: true)
            trainButton.isHidden = false
            retrainButton.isHidden = true
            detailLabel.text = "Pick a corpus size and tap Train. Skip-gram, 100 dims, 5 epochs."

        case .loading:
            statusLabel.text = "Loading cached model…"
            progressView.isHidden = true
            progressLabel.isHidden = true
            setControlsEnabled(idle: false)
            detailLabel.text = nil

        case let .training(progress):
            statusLabel.text = "Training…"
            progressView.isHidden = false
            progressLabel.isHidden = false
            progressView.setProgress(Float(progress), animated: true)
            progressLabel.text = String(format: "%.0f%%", progress * 100)
            setControlsEnabled(idle: false)
            detailLabel.text = "This runs on a background thread; the other tabs stay usable once a model is ready."

        case let .ready(model):
            statusLabel.text = "✓ Model ready"
            progressView.isHidden = true
            progressLabel.isHidden = true
            setControlsEnabled(idle: true)
            trainButton.isHidden = true
            retrainButton.isHidden = false
            detailLabel.text = readyDetail(model: model)

        case let .failed(message):
            statusLabel.text = "Training failed"
            progressView.isHidden = true
            progressLabel.isHidden = true
            setControlsEnabled(idle: true)
            trainButton.isHidden = false
            retrainButton.isHidden = true
            detailLabel.text = message
        }
    }

    private func readyDetail(model: WordEmbeddings) -> String {
        var lines = ["Vocabulary: \(model.vocabulary.count) words · \(model.vectorSize) dims"]
        if let info = ModelStore.shared.lastTrainingInfo {
            lines.append("Trained on \(info.scope.title) · \(info.sentenceCount) sentences")
            lines.append(String(format: "Training time: %.1fs", info.duration))
        } else {
            lines.append("Loaded from cache. Use the Nearest and Word Algebra tabs.")
        }
        return lines.joined(separator: "\n")
    }

    /// When `idle` is false the user is mid-run: disable inputs to prevent overlap.
    private func setControlsEnabled(idle: Bool) {
        scopeControl.isEnabled = idle
        trainButton.isEnabled = idle
        retrainButton.isEnabled = idle
    }
}
