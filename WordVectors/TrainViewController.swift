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

    /// The "Select All" header row above the book list, kept so its checked state can be
    /// refreshed whenever the selection changes. `nil` when no books are bundled.
    private var selectAllRow: SelectAllRow?

    /// The hyperparameters training will use. Seeded from `Word2VecParameters` defaults; the
    /// four Tier-1 stepper rows write their edited values back into this struct, and every
    /// other (Tier-2/3) field is shown read-only and left at its default.
    private var parameters = Word2VecParameters()

    /// The editable Tier-1 stepper rows, kept so training can be disabled mid-run.
    private var paramRows: [ParameterStepperRow] = []

    private let statusLabel = UILabel()
    private let progressView = UIProgressView(progressViewStyle: .default)
    private let progressLabel = UILabel()
    private let trainButton = UIButton(type: .system)
    private let retrainButton = UIButton(type: .system)
    private let detailLabel = UILabel()
    private let selectionHintLabel = UILabel()

    /// The book titles behind the "N books" line in the ready-state footer. Non-empty only
    /// while a trained model is shown, which is exactly when tapping the footer lists them.
    private var footerBookTitles: [String] = []

    /// Scrolls only the book list, so the corpus/hyperparameter headers, the hyperparameter
    /// column, and the Train/Retrain controls stay fixed while a long book list scrolls.
    private let booksScrollView = UIScrollView()

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
        // The footer's "N books" line is tappable to reveal the full title list — see
        // `detailTapped`. The gesture is harmless in other states because it only acts when
        // `footerBookTitles` is populated, which happens solely in the ready state.
        detailLabel.isUserInteractionEnabled = true
        detailLabel.addGestureRecognizer(
            UITapGestureRecognizer(target: self, action: #selector(detailTapped)))

        var trainConfig = UIButton.Configuration.filled()
        trainConfig.title = "Train Model"
        trainButton.configuration = trainConfig
        trainButton.addTarget(self, action: #selector(trainTapped), for: .touchUpInside)

        var retrainConfig = UIButton.Configuration.gray()
        retrainConfig.title = "Retrain (clear cache)"
        retrainButton.configuration = retrainConfig
        retrainButton.addTarget(self, action: #selector(retrainTapped), for: .touchUpInside)

        // Title + subtitle for the leading column, mirroring the "Hyperparameters" headline in
        // the trailing column so both columns read the same way.
        let corpusTitle = UILabel()
        corpusTitle.text = "Training Corpus"
        corpusTitle.font = .preferredFont(forTextStyle: .headline)
        corpusTitle.textAlignment = .center

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

        // "Select All" header row above the book list, only when there's a list to select.
        // Toggling it on selects every book; off collapses to just the first book (which keeps
        // the "at least one book always selected" invariant). Only created when books exist.
        if !books.isEmpty {
            let row = SelectAllRow()
            row.onToggle = { [weak self] in self?.toggleSelectAll() }
            selectAllRow = row
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

        // Tier-1 editable hyperparameters: one stepper row each, clamped to a sane range.
        // The steppers write straight back into `parameters`.
        let paramsCaption = UILabel()
        paramsCaption.text = "Hyperparameters"
        paramsCaption.font = .preferredFont(forTextStyle: .headline)
        paramsCaption.textAlignment = .center

        paramRows = [
            ParameterStepperRow(
                title: "Vector length",
                subtitle: "dimensions per word",
                range: 25...1000, step: 25, value: parameters.vectorSize
            ) { [weak self] in self?.parameters.vectorSize = $0; self?.parametersChanged() },
            ParameterStepperRow(
                title: "Iterations",
                subtitle: "training epochs",
                range: 1...20, step: 1, value: parameters.iterations
            ) { [weak self] in self?.parameters.iterations = $0; self?.parametersChanged() },
            ParameterStepperRow(
                title: "Context window",
                subtitle: "words of context each side",
                range: 2...10, step: 1, value: parameters.window
            ) { [weak self] in self?.parameters.window = $0; self?.parametersChanged() },
            ParameterStepperRow(
                title: "Min count",
                subtitle: "drop words rarer than this",
                range: 1...20, step: 1, value: parameters.minCount
            ) { [weak self] in self?.parameters.minCount = $0; self?.parametersChanged() },
        ]

        let paramsStack = UIStackView(arrangedSubviews: paramRows)
        paramsStack.axis = .vertical
        paramsStack.spacing = 4
        paramsStack.alignment = .fill

        // Tier-2/3: shown read-only so the full config is visible, but not editable here.
        let advancedLabel = UILabel()
        advancedLabel.font = .preferredFont(forTextStyle: .footnote)
        advancedLabel.textColor = .secondaryLabel
        advancedLabel.numberOfLines = 0
        advancedLabel.text = readOnlyParametersText()

        // The book list and the hyperparameters sit side by side: books in the leading column,
        // hyperparameters in the trailing column, each with its own caption.
        // Title sits directly above its subtitle with tight spacing, so the pair reads as one
        // header while still lining up with the trailing column's "Hyperparameters" headline.
        let corpusHeader = UIStackView(arrangedSubviews: [corpusTitle, corpusCaption])
        corpusHeader.axis = .vertical
        corpusHeader.spacing = 4
        corpusHeader.alignment = .fill

        // Only the book list scrolls. The stack of rows is pinned to the scroll view's content
        // guide and matched to its frame width, so it scrolls vertically and never horizontally.
        booksStack.translatesAutoresizingMaskIntoConstraints = false
        booksScrollView.translatesAutoresizingMaskIntoConstraints = false
        booksScrollView.showsHorizontalScrollIndicator = false
        // The book list is the one element that stretches to absorb spare vertical space; give it
        // the lowest hugging/compression-resistance so it — and not the fixed rows — grows or
        // shrinks as the window resizes.
        booksScrollView.setContentHuggingPriority(.defaultLow, for: .vertical)
        booksScrollView.setContentCompressionResistancePriority(.defaultLow, for: .vertical)
        booksScrollView.addSubview(booksStack)
        let booksContent = booksScrollView.contentLayoutGuide
        let booksFrame = booksScrollView.frameLayoutGuide
        NSLayoutConstraint.activate([
            booksStack.topAnchor.constraint(equalTo: booksContent.topAnchor),
            booksStack.bottomAnchor.constraint(equalTo: booksContent.bottomAnchor),
            booksStack.leadingAnchor.constraint(equalTo: booksContent.leadingAnchor),
            booksStack.trailingAnchor.constraint(equalTo: booksContent.trailingAnchor),
            booksStack.widthAnchor.constraint(equalTo: booksFrame.widthAnchor),
        ])

        // The "Select All" row sits between the header and the scrolling book list, so it stays
        // pinned at the top while the list scrolls beneath it. Only present when books exist.
        var corpusColumnViews: [UIView] = [corpusHeader]
        if let selectAllRow {
            corpusColumnViews.append(selectAllRow)
        }
        corpusColumnViews.append(booksScrollView)
        corpusColumnViews.append(selectionHintLabel)

        let corpusColumn = UIStackView(arrangedSubviews: corpusColumnViews)
        corpusColumn.axis = .vertical
        corpusColumn.spacing = 16
        corpusColumn.alignment = .fill
        // Tighten the gap between the "Select All" row and the list it controls so they read as
        // one unit, distinct from the header above.
        if let selectAllRow {
            corpusColumn.setCustomSpacing(8, after: selectAllRow)
        }

        let paramsColumn = UIStackView(arrangedSubviews: [
            paramsCaption,
            paramsStack,
            advancedLabel,
        ])
        paramsColumn.axis = .vertical
        paramsColumn.spacing = 16
        paramsColumn.alignment = .fill
        // The hyperparameter column hugs its content and pins to the top so it doesn't stretch
        // to the corpus column's full (scrollable) height.
        paramsColumn.setContentHuggingPriority(.required, for: .vertical)

        // Wrap the fixed-height params column so the shorter of the two columns sits at the top
        // rather than being stretched to match the scrollable corpus column.
        let paramsContainer = UIStackView(arrangedSubviews: [paramsColumn, UIView()])
        paramsContainer.axis = .vertical
        paramsContainer.alignment = .fill

        let columns = UIStackView(arrangedSubviews: [corpusColumn, paramsContainer])
        columns.axis = .horizontal
        columns.spacing = 24
        columns.alignment = .fill
        columns.distribution = .fillEqually
        // The columns row absorbs the main stack's spare vertical space (which the book list then
        // fills), so the fixed labels and buttons keep their intrinsic heights.
        columns.setContentHuggingPriority(.defaultLow, for: .vertical)
        columns.setContentCompressionResistancePriority(.defaultLow, for: .vertical)

        // Fixed content: status header, the two columns (only the book list inside scrolls), and
        // the Train/Retrain controls. Pinned to the safe area — the whole window no longer scrolls.
        let stack = UIStackView(arrangedSubviews: [
            statusLabel,
            columns,
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
            stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 24),
            stack.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            stack.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -24),
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

    /// Handles the "Select All" header row. When every book is already selected, toggling it off
    /// collapses the selection to just the first book — never to nothing, preserving the
    /// "at least one book always selected" invariant. Otherwise it selects every book.
    private func toggleSelectAll() {
        if selectedStems.count == books.count {
            // All selected → collapse to just the first book on screen.
            if let first = books.first {
                selectedStems = [first.stem]
            }
        } else {
            // Some or none selected → select everything.
            selectedStems = Set(books.map { $0.stem })
        }
        refreshBookRows()
        updateSelectionState()
    }

    private func refreshBookRows() {
        for row in bookRows {
            row.setChecked(selectedStems.contains(row.book.stem))
        }
        // The header reflects reality: checked only when every book is selected.
        selectAllRow?.setChecked(!books.isEmpty && selectedStems.count == books.count)
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
        ModelStore.shared.train(stems: orderedSelectedStems, parameters: parameters)
    }

    @objc private func retrainTapped() {
        guard !orderedSelectedStems.isEmpty else { return }
        ModelStore.shared.clearCache()
        ModelStore.shared.train(stems: orderedSelectedStems, parameters: parameters)
    }

    // MARK: - State rendering

    private func render(_ state: ModelState) {
        // Clear the tappable book list up front; only the ready state repopulates it (via
        // `readyDetail`), so a tap can never surface titles from a prior model.
        footerBookTitles = []
        switch state {
        case .idle:
            statusLabel.text = "No model yet"
            progressView.isHidden = true
            progressLabel.isHidden = true
            setControlsEnabled(idle: true)
            trainButton.isHidden = false
            retrainButton.isHidden = true
            detailLabel.text = idleDetailText

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
        // Keep the footer's tap affordance in sync with reality: only the ready state populates
        // `footerBookTitles`, so only then should VoiceOver present the label as an activatable
        // button. All other states clear the button trait so it reads as plain text.
        updateFooterAccessibility()
    }

    /// Mirrors the tappable-footer state into VoiceOver. When a trained model is shown the
    /// footer lists its books on tap, so it advertises the `.button` trait, a descriptive label,
    /// and a hint; otherwise it is a plain, non-interactive text element. Matches the affordance
    /// the sibling `BookRow`/`SelectAllRow` controls expose.
    private func updateFooterAccessibility() {
        if footerBookTitles.isEmpty {
            detailLabel.accessibilityTraits.remove(.button)
            detailLabel.accessibilityLabel = nil
            detailLabel.accessibilityHint = nil
        } else {
            let scope = TrainingInfo.bookCountSummary(footerBookTitles.count)
            detailLabel.accessibilityTraits.insert(.button)
            detailLabel.accessibilityLabel = "List the \(scope) trained on"
            detailLabel.accessibilityHint = "Shows the full list of books in the training corpus"
        }
    }

    private func readyDetail(model: WordEmbeddings) -> String {
        var firstLine = "Vocabulary: \(model.vocabulary.count) words · \(model.vectorSize) dims"
        // Append the model's on-disk size when it's cached, so the footer reports how large the
        // final saved model actually is.
        if let bytes = ModelStore.shared.cachedModelByteSize {
            let size = ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
            firstLine += " · \(size) on disk"
        }
        var lines = [firstLine]
        if let info = ModelStore.shared.lastTrainingInfo {
            let p = info.parameters
            // Note provenance so cached detail isn't mistaken for a run that just happened.
            let scope = ModelStore.shared.trainingInfoFromCache
                ? "Cached · trained on \(info.scopeSummary) · \(info.sentenceCount) sentences"
                : "Trained on \(info.scopeSummary) · \(info.sentenceCount) sentences"
            lines.append(scope)
            // Show only the count here — listing every title balloons the footer for large
            // corpora. The full list is a tap away (see `detailTapped`); stash the titles so
            // that handler can present them.
            footerBookTitles = info.bookTitles
            lines.append("Tap to list the \(info.scopeSummary)")
            lines.append("\(p.iterations) epochs · window \(p.window) · min count \(p.minCount)")
            lines.append(String(format: "Training time: %.1fs", info.duration))
        } else {
            footerBookTitles = []
            lines.append("Loaded from cache. Use the Nearest and Word Algebra tabs.")
        }
        return lines.joined(separator: "\n")
    }

    /// Presents the full training-corpus book list in a scrollable sheet. No-op unless the
    /// footer is currently showing a trained model's "N books" line (the only time
    /// `footerBookTitles` is populated), so taps in other states do nothing.
    @objc private func detailTapped() {
        guard !footerBookTitles.isEmpty else { return }
        let listVC = BookListViewController(titles: footerBookTitles)
        let nav = UINavigationController(rootViewController: listVC)
        if let sheet = nav.sheetPresentationController {
            sheet.detents = [.medium(), .large()]
            sheet.prefersGrabberVisible = true
        }
        present(nav, animated: true)
    }

    /// The idle-state hint, reflecting the currently-selected editable hyperparameters so it
    /// never goes stale as the user adjusts the steppers.
    private var idleDetailText: String {
        let model = parameters.useCBOW ? "CBOW" : "Skip-gram"
        return "Pick one or more books, tune the hyperparameters, and tap Train. "
            + "\(model), \(parameters.vectorSize) dims, \(parameters.iterations) epochs."
    }

    /// Called after a stepper edits `parameters`. Editing a hyperparameter doesn't change the
    /// model state, so `render` won't fire — refresh the idle hint here so it stays accurate.
    private func parametersChanged() {
        if case .idle = ModelStore.shared.state {
            detailLabel.text = idleDetailText
        }
    }

    /// Describes the Tier-2/3 parameters the user can see but not edit here. Reads from a
    /// fresh `Word2VecParameters` so it always matches the defaults training actually uses.
    private func readOnlyParametersText() -> String {
        let p = Word2VecParameters()
        let model = p.useCBOW ? "CBOW" : "skip-gram"
        return [
            "Fixed: \(model) · \(p.negativeSamples) negative samples",
            String(format: "learning rate %.3f · subsample %.0e", p.initialAlpha, p.subsample),
        ].joined(separator: "\n")
    }

    /// When `idle` is false the user is mid-run: disable inputs to prevent overlap.
    /// When idle, Train also requires at least one selected book.
    private func setControlsEnabled(idle: Bool) {
        selectAllRow?.isEnabled = idle
        for row in bookRows { row.isEnabled = idle }
        for row in paramRows { row.isEnabled = idle }
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

// MARK: - Select-all header row

/// A tappable "Select All" row that sits above the book list. Behaves like the book checkboxes —
/// the whole row is one touch target reporting toggles via `onToggle` — but is styled as a header:
/// its label is emphasized and a hairline separator sits beneath it to set it off from the list.
/// It shows checked only when every book is selected; the controller drives its state.
private final class SelectAllRow: UIControl {

    var onToggle: (() -> Void)?

    private let checkImageView = UIImageView()
    private let titleLabel = UILabel()
    private var checked = false

    init() {
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

        titleLabel.text = "Select All"
        titleLabel.font = .preferredFont(forTextStyle: .subheadline).bold()
        titleLabel.numberOfLines = 0

        let row = UIStackView(arrangedSubviews: [checkImageView, titleLabel])
        row.axis = .horizontal
        row.spacing = 12
        row.alignment = .center
        row.isUserInteractionEnabled = false // let the control receive the touch
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)

        // A hairline separator beneath the row visually groups it with the list it controls
        // while still setting it apart as a header.
        let separator = UIView()
        separator.backgroundColor = .separator
        separator.translatesAutoresizingMaskIntoConstraints = false
        addSubview(separator)

        NSLayoutConstraint.activate([
            checkImageView.widthAnchor.constraint(equalToConstant: 26),
            checkImageView.heightAnchor.constraint(equalToConstant: 26),
            row.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            row.leadingAnchor.constraint(equalTo: leadingAnchor),
            row.trailingAnchor.constraint(equalTo: trailingAnchor),
            separator.topAnchor.constraint(equalTo: row.bottomAnchor, constant: 8),
            separator.leadingAnchor.constraint(equalTo: leadingAnchor),
            separator.trailingAnchor.constraint(equalTo: trailingAnchor),
            separator.bottomAnchor.constraint(equalTo: bottomAnchor),
            separator.heightAnchor.constraint(equalToConstant: 1.0 / UIScreen.main.scale),
        ])
    }

    func setChecked(_ checked: Bool) {
        self.checked = checked
        let name = checked ? "checkmark.square.fill" : "square"
        checkImageView.image = UIImage(systemName: name)
        accessibilityLabel = "Select All"
        accessibilityValue = checked ? "All books selected" : "Not all books selected"
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

// MARK: - Hyperparameter stepper row

/// A labeled row with a `UIStepper` for one integer hyperparameter. The stepper is clamped to
/// `range` and moves by `step`; the current value shows to the left of the +/− control, and
/// each change is reported through `onChange` as the clamped integer value.
private final class ParameterStepperRow: UIView {

    var onChange: ((Int) -> Void)?

    private let titleLabel = UILabel()
    private let subtitleLabel = UILabel()
    private let valueLabel = UILabel()
    private let stepper = UIStepper()

    init(title: String,
         subtitle: String,
         range: ClosedRange<Int>,
         step: Int,
         value: Int,
         onChange: @escaping (Int) -> Void) {
        self.onChange = onChange
        super.init(frame: .zero)

        titleLabel.text = title
        titleLabel.font = .preferredFont(forTextStyle: .body)
        titleLabel.numberOfLines = 0

        subtitleLabel.text = subtitle
        subtitleLabel.font = .preferredFont(forTextStyle: .caption1)
        subtitleLabel.textColor = .secondaryLabel
        subtitleLabel.numberOfLines = 0

        valueLabel.font = .preferredFont(forTextStyle: .body).monospaced()
        valueLabel.textColor = .label
        valueLabel.textAlignment = .right
        valueLabel.setContentHuggingPriority(.required, for: .horizontal)
        valueLabel.setContentCompressionResistancePriority(.required, for: .horizontal)

        stepper.minimumValue = Double(range.lowerBound)
        stepper.maximumValue = Double(range.upperBound)
        stepper.stepValue = Double(step)
        stepper.value = Double(value)
        stepper.setContentHuggingPriority(.required, for: .horizontal)
        stepper.setContentCompressionResistancePriority(.required, for: .horizontal)
        stepper.addTarget(self, action: #selector(stepperChanged), for: .valueChanged)

        let labels = UIStackView(arrangedSubviews: [titleLabel, subtitleLabel])
        labels.axis = .vertical
        labels.spacing = 2
        labels.alignment = .leading
        // Hug the label text so it doesn't stretch and shove the controls to the far edge —
        // the value + stepper should sit just to the right of the title, not across the row.
        labels.setContentHuggingPriority(.required, for: .horizontal)

        // A flexible spacer that absorbs the row's extra width, pushing the value and stepper to
        // the trailing edge so they line up across every row regardless of label width.
        let spacer = UIView()
        spacer.setContentHuggingPriority(.defaultLow, for: .horizontal)
        spacer.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        let row = UIStackView(arrangedSubviews: [labels, spacer, valueLabel, stepper])
        row.axis = .horizontal
        row.spacing = 12
        row.alignment = .center
        row.translatesAutoresizingMaskIntoConstraints = false
        addSubview(row)

        NSLayoutConstraint.activate([
            row.topAnchor.constraint(equalTo: topAnchor, constant: 8),
            row.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -8),
            row.leadingAnchor.constraint(equalTo: leadingAnchor),
            row.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])

        // Let VoiceOver reach the stepper itself (an adjustable control) rather than
        // collapsing the row into one static element — otherwise the value can't be changed
        // with the rotor. The stepper carries the parameter's name as its label.
        stepper.accessibilityLabel = title
        updateValueDisplay()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    /// Current stepper value as a clamped integer (the stepper already enforces range/step).
    private var currentValue: Int { Int(stepper.value.rounded()) }

    @objc private func stepperChanged() {
        updateValueDisplay()
        onChange?(currentValue)
    }

    private func updateValueDisplay() {
        valueLabel.text = "\(currentValue)"
        stepper.accessibilityValue = "\(currentValue)"
    }

    var isEnabled: Bool = true {
        didSet {
            stepper.isEnabled = isEnabled
            alpha = isEnabled ? 1.0 : 0.5
        }
    }
}

// MARK: - Font helper

private extension UIFont {
    /// A monospaced-digit variant so value labels don't jitter as digits change width.
    func monospaced() -> UIFont {
        let descriptor = fontDescriptor.addingAttributes([
            .featureSettings: [[
                UIFontDescriptor.FeatureKey.type: kNumberSpacingType,
                UIFontDescriptor.FeatureKey.selector: kMonospacedNumbersSelector,
            ]],
        ])
        return UIFont(descriptor: descriptor, size: pointSize)
    }

    /// A bold variant of the same size/style, used to emphasize the "Select All" header while
    /// keeping it a Dynamic Type text style. Falls back to the original font if the bold trait
    /// can't be applied.
    func bold() -> UIFont {
        guard let descriptor = fontDescriptor.withSymbolicTraits(.traitBold) else { return self }
        return UIFont(descriptor: descriptor, size: pointSize)
    }
}

// MARK: - Book list sheet

/// A scrollable, modally-presented list of every book in the training corpus. Backed by a
/// `UITableView` so it stays usable for large corpora (100+ books) where an alert would
/// truncate or grow unwieldy. Presented from the training footer's "N books" line.
private final class BookListViewController: UITableViewController {

    private let titles: [String]

    init(titles: [String]) {
        self.titles = titles
        super.init(style: .plain)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Title Case the shared "N book(s)" summary for the nav bar, so pluralization stays in
        // one place (TrainingInfo.bookCountSummary) while the title keeps iOS's Title Case.
        title = TrainingInfo.bookCountSummary(titles.count).capitalized
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            systemItem: .done, primaryAction: UIAction { [weak self] _ in
                self?.dismiss(animated: true)
            })
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        // Titles are display-only; no selection affordance or trailing chevron.
        tableView.allowsSelection = false
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        titles.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        var content = cell.defaultContentConfiguration()
        content.text = titles[indexPath.row]
        content.textProperties.numberOfLines = 0
        cell.contentConfiguration = content
        return cell
    }
}
