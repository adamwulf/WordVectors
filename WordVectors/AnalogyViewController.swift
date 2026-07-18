//
//  AnalogyViewController.swift
//  WordVectors
//
//  Feature C — Word algebra: base - minus + plus (e.g. king - man + woman ≈ queen).
//  Three fields prefilled with the classic example. Every input is lowercased and
//  checked for vocabulary membership; a missing word is named in a friendly message.
//

import UIKit
import OSLog
import WordVectorKit

final class AnalogyViewController: UIViewController, UITextFieldDelegate, UITableViewDataSource {

    private let baseField = UITextField()
    private let minusField = UITextField()
    private let plusField = UITextField()
    private let computeButton = UIButton(type: .system)
    private let equationLabel = UILabel()
    private let messageLabel = UILabel()
    private let tableView = UITableView(frame: .zero, style: .plain)

    private var results: [(word: String, similarity: Float)] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Word Algebra"
        view.backgroundColor = .systemBackground
        buildUI()
        ModelStore.shared.addObserver(self) { [weak self] state in
            self?.render(state)
        }
    }

    // MARK: - UI

    private func buildUI() {
        configureField(baseField, placeholder: "base", text: "king")
        configureField(minusField, placeholder: "minus", text: "man")
        configureField(plusField, placeholder: "plus", text: "woman")

        equationLabel.font = .preferredFont(forTextStyle: .headline)
        equationLabel.textAlignment = .center
        equationLabel.adjustsFontSizeToFitWidth = true
        updateEquationLabel()

        let minusSign = fixedLabel("−")
        let plusSign = fixedLabel("+")

        let equationRow = UIStackView(arrangedSubviews: [baseField, minusSign, minusField, plusSign, plusField])
        equationRow.axis = .horizontal
        equationRow.spacing = 8
        equationRow.alignment = .center
        equationRow.distribution = .fill

        var config = UIButton.Configuration.filled()
        config.title = "Compute"
        computeButton.configuration = config
        computeButton.addTarget(self, action: #selector(computeTapped), for: .touchUpInside)

        messageLabel.font = .preferredFont(forTextStyle: .callout)
        messageLabel.textColor = .secondaryLabel
        messageLabel.numberOfLines = 0

        tableView.dataSource = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        tableView.translatesAutoresizingMaskIntoConstraints = false

        let inputStack = UIStackView(arrangedSubviews: [equationLabel, equationRow, computeButton, messageLabel])
        inputStack.axis = .vertical
        inputStack.spacing = 12
        inputStack.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(inputStack)
        view.addSubview(tableView)

        NSLayoutConstraint.activate([
            inputStack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            inputStack.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
            inputStack.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),

            tableView.topAnchor.constraint(equalTo: inputStack.bottomAnchor, constant: 16),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    private func configureField(_ field: UITextField, placeholder: String, text: String) {
        field.placeholder = placeholder
        field.text = text
        field.borderStyle = .roundedRect
        field.autocapitalizationType = .none
        field.autocorrectionType = .no
        field.returnKeyType = .done
        field.textAlignment = .center
        field.delegate = self
        field.addTarget(self, action: #selector(fieldChanged), for: .editingChanged)
        field.setContentHuggingPriority(.defaultLow, for: .horizontal)
    }

    private func fixedLabel(_ text: String) -> UILabel {
        let label = UILabel()
        label.text = text
        label.font = .preferredFont(forTextStyle: .title3)
        label.setContentHuggingPriority(.required, for: .horizontal)
        label.setContentCompressionResistancePriority(.required, for: .horizontal)
        return label
    }

    private func updateEquationLabel() {
        let b = baseField.text?.lowercased() ?? "base"
        let m = minusField.text?.lowercased() ?? "minus"
        let p = plusField.text?.lowercased() ?? "plus"
        equationLabel.text = "\(b) − \(m) + \(p) = ?"
    }

    // MARK: - Actions

    @objc private func fieldChanged() {
        updateEquationLabel()
    }

    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        return true
    }

    @objc private func computeTapped() {
        view.endEditing(true)
        results = []

        guard let model = ModelStore.shared.embeddings else {
            setMessage("Train a model first (see the Train tab).")
            tableView.reloadData()
            return
        }

        let base = normalized(baseField.text)
        let minus = normalized(minusField.text)
        let plus = normalized(plusField.text)

        guard !base.isEmpty, !minus.isEmpty, !plus.isEmpty else {
            setMessage("Fill in all three words.")
            tableView.reloadData()
            return
        }

        // Name any out-of-vocabulary input rather than silently returning nothing.
        let missing = [base, minus, plus].filter { !model.contains($0) }
        if !missing.isEmpty {
            appLog.info("Analogy \(base, privacy: .public)−\(minus, privacy: .public)+\(plus, privacy: .public): OOV \(missing.joined(separator: ","), privacy: .public)")
            let list = missing.map { "'\($0)'" }.joined(separator: ", ")
            let plural = missing.count == 1 ? "is" : "are"
            setMessage("\(list) \(plural) not in the vocabulary. Try more common words.")
            tableView.reloadData()
            return
        }

        results = model.analogy(base: base, minus: minus, plus: plus, count: 10)
        appLog.info("Analogy \(base, privacy: .public)−\(minus, privacy: .public)+\(plus, privacy: .public): \(self.results.count, privacy: .public) results.")
        if results.isEmpty {
            setMessage("No result for \(base) − \(minus) + \(plus).")
        } else {
            setMessage("\(base) − \(minus) + \(plus) ≈")
        }
        tableView.reloadData()
    }

    private func normalized(_ text: String?) -> String {
        (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private func setMessage(_ text: String?) {
        messageLabel.text = text
    }

    // MARK: - State

    private func render(_ state: ModelState) {
        let ready: Bool
        if case .ready = state { ready = true } else { ready = false }
        baseField.isEnabled = ready
        minusField.isEnabled = ready
        plusField.isEnabled = ready
        computeButton.isEnabled = ready
        if !ready {
            results = []
            setMessage("No model yet — train one on the Train tab.")
            tableView.reloadData()
        } else if results.isEmpty {
            // Prompt the user only if they haven't already computed an analogy this session.
            setMessage("Model ready. Tap Compute to try king − man + woman.")
        }
    }

    // MARK: - UITableViewDataSource

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return results.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        let item = results[indexPath.row]
        var content = cell.defaultContentConfiguration()
        content.text = "\(indexPath.row + 1).  \(item.word)"
        content.secondaryText = String(format: "%.3f", item.similarity)
        content.prefersSideBySideTextAndSecondaryText = true
        cell.contentConfiguration = content
        cell.selectionStyle = .none
        return cell
    }
}
