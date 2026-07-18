//
//  TrainViewController.swift
//  WordVectors
//
//  Feature A — Train word vectors from the bundled Gutenberg corpus.
//  Lets the user pick which books to train on (checkboxes; at least one required),
//  shows a progress bar driven by the training callback, and reports the resulting
//  model (vocabulary size, time). A cached model loads automatically at launch;
//  "Retrain" clears the cache.
//

import UIKit
import WordVectorKit

final class TrainViewController: UIViewController {

    /// Every bundled book, resolved once at load. Empty only if the bundle has no corpus.
    private let books = CorpusLoader.allBooks()

    /// Stems the user has checked. Seeded with the default book so training is always
    /// possible on first launch, and never allowed to become empty via the UI.
    private var selectedStems: Set<String> = []

    /// One checkbox row per book, kept so we can refresh their checked state on toggle.
    private var bookRows: [BookRow] = []

    private let statusLabel = UILabel()
    private let progressView = UIProgressView(progressViewStyle: .default)
    private let progressLabel = UILabel()
    private let trainButton = UIButton(type: .system)
    private let retrainButton = UIButton(type: .system)
    private let detailLabel = UILabel()
    private let selectionHintLabel = UILabel()

    /// Scrolls the whole content so a long book list stays reachable on small screens.
    private let scrollView = UIScrollView()

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Train"
        view.backgroundColor = .systemBackground

        // Default selection: the designated default book if present, otherwise the first
        // bundled book, so the list is never empty when at least one book exists.
        if books.contains(where: { $0.stem == defaultBookStem }) {
            selectedStems = [defaultBookStem]
        } else if let first = books.first {
            selectedStems = [first.stem]
        }

        buildUI()
        ModelStore.shared.addObserver(self) { [weak self] state in
            self?.render(state)
        }
    }

    // MARK: - UI

    private func buildUI() {
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
        corpusCaption.text = "Choose books to train on (more books = slower, better vectors)"
        corpusCaption.font = .preferredFont(forTextStyle: .footnote)
        corpusCaption.textColor = .secondaryLabel
        corpusCaption.numberOfLines = 0
        corpusCaption.textAlignment = .center

        selectionHintLabel.font = .preferredFont(forTextStyle: .caption1)
        selectionHintLabel.textColor = .secondaryLabel
        selectionHintLabel.numberOfLines = 0
        selectionHintLabel.textAlignment = .center

        // One tappable checkbox row per book.
        bookRows = books.map { book in
            let row = BookRow(book: book)
            row.onToggle = { [weak self] in self?.toggle(book.stem) }
            return row
        }

        let booksStack = UIStackView(arrangedSubviews: bookRows)
        booksStack.axis = .vertical
        booksStack.spacing = 4
        booksStack.alignment = .fill

        // If somehow no books are bundled, say so instead of showing an empty list.
        if bookRows.isEmpty {
            let empty = UILabel()
            empty.text = "No corpus books found in the app bundle."
            empty.font = .preferredFont(forTextStyle: .footnote)
            empty.textColor = .secondaryLabel
            empty.numberOfLines = 0
            empty.textAlignment = .center
            booksStack.addArrangedSubview(empty)
        }

        let stack = UIStackView(arrangedSubviews: [
            statusLabel,
            corpusCaption,
            booksStack,
            selectionHintLabel,
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

        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.addSubview(stack)
        view.addSubview(scrollView)

        let content = scrollView.contentLayoutGuide
        let frame = scrollView.frameLayoutGuide
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),

            stack.topAnchor.constraint(greaterThanOrEqualTo: content.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -24),
            stack.centerYAnchor.constraint(equalTo: content.centerYAnchor).withPriority(.defaultLow),
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -24),
            stack.widthAnchor.constraint(equalTo: frame.widthAnchor, constant: -48),
        ])

        refreshBookRows()
        updateSelectionState()
    }

    // MARK: - Selection

    /// Toggles a book, but never lets the last checked book be unchecked — at least one
    /// book must always be selected so training has something to work with.
    private func toggle(_ stem: String) {
        if selectedStems.contains(stem) {
            // Refuse to remove the final selection.
            guard selectedStems.count > 1 else { return }
            selectedStems.remove(stem)
        } else {
            selectedStems.insert(stem)
        }
        refreshBookRows()
        updateSelectionState()
    }

    private func refreshBookRows() {
        for row in bookRows {
            row.setChecked(selectedStems.contains(row.book.stem))
        }
    }

    /// Enables/disables Train based on whether anything is selected and updates the hint.
    private func updateSelectionState() {
        let count = selectedStems.count
        selectionHintLabel.text = count == 1 ? "1 book selected" : "\(count) books selected"
        // Only meaningful when idle; render() has the final say on enabled state.
        if case .ready = ModelStore.shared.state {} else if case .training = ModelStore.shared.state {} else {
            trainButton.isEnabled = count > 0
        }
    }

    private var orderedSelectedStems: [String] {
        // Preserve the on-screen (title) order so training and the summary read naturally.
        books.filter { selectedStems.contains($0.stem) }.map { $0.stem }
    }

    // MARK: - Actions

    @objc private func trainTapped() {
        guard !orderedSelectedStems.isEmpty else { return }
        ModelStore.shared.train(stems: orderedSelectedStems)
    }

    @objc private func retrainTapped() {
        guard !orderedSelectedStems.isEmpty else { return }
        ModelStore.shared.clearCache()
        ModelStore.shared.train(stems: orderedSelectedStems)
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
            detailLabel.text = "Pick one or more books and tap Train. Skip-gram, 100 dims, 5 epochs."

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
            lines.append("Trained on \(info.scopeSummary) · \(info.sentenceCount) sentences")
            lines.append(info.bookTitles.joined(separator: ", "))
            lines.append(String(format: "Training time: %.1fs", info.duration))
        } else {
            lines.append("Loaded from cache. Use the Nearest and Word Algebra tabs.")
        }
        return lines.joined(separator: "\n")
    }

    /// When `idle` is false the user is mid-run: disable inputs to prevent overlap.
    /// When idle, Train also requires at least one selected book.
    private func setControlsEnabled(idle: Bool) {
        for row in bookRows { row.isEnabled = idle }
        trainButton.isEnabled = idle && !selectedStems.isEmpty
        retrainButton.isEnabled = idle && !selectedStems.isEmpty
    }
}

// MARK: - Book checkbox row

/// A single tappable row: a checkbox glyph plus the book title. Behaves like a checkbox —
/// the whole row is one big touch target that reports toggles via `onToggle`.
private final class BookRow: UIControl {

    let book: CorpusBook
    var onToggle: (() -> Void)?

    private let checkImageView = UIImageView()
    private let titleLabel = UILabel()

    init(book: CorpusBook) {
        self.book = book
        super.init(frame: .zero)
        buildUI()
        addTarget(self, action: #selector(tapped), for: .touchUpInside)
        accessibilityTraits = .button
        isAccessibilityElement = true
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    private func buildUI() {
        checkImageView.contentMode = .scaleAspectFit
        checkImageView.setContentHuggingPriority(.required, for: .horizontal)
        checkImageView.tintColor = .tintColor

        titleLabel.text = book.title
        titleLabel.font = .preferredFont(forTextStyle: .body)
        titleLabel.numberOfLines = 0

        let row = UIStackView(arrangedSubviews: [checkImageView, titleLabel])
        row.axis = .horizontal
        row.spacing = 12
        row.alignment = .center
        row.isUserInteractionEnabled = false // let the control receive the touch
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)

        NSLayoutConstraint.activate([
            checkImageView.widthAnchor.constraint(equalToConstant: 26),
            checkImageView.heightAnchor.constraint(equalToConstant: 26),
            row.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            row.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -8),
            row.leadingAnchor.constraint(equalTo: leadingAnchor),
            row.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])
    }

    func setChecked(_ checked: Bool) {
        let name = checked ? "checkmark.square.fill" : "square"
        checkImageView.image = UIImage(systemName: name)
        titleLabel.textColor = isEnabled ? .label : .secondaryLabel
        accessibilityLabel = book.title
        accessibilityValue = checked ? "Selected" : "Not selected"
    }

    override var isEnabled: Bool {
        didSet {
            alpha = isEnabled ? 1.0 : 0.5
            titleLabel.textColor = isEnabled ? .label : .secondaryLabel
        }
    }

    @objc private func tapped() {
        onToggle?()
    }
}

// MARK: - Constraint helper

private extension NSLayoutConstraint {
    /// Fluent priority setter so a constraint can be created and prioritized inline.
    func withPriority(_ priority: UILayoutPriority) -> NSLayoutConstraint {
        self.priority = priority
        return self
    }
}
