//
//  NearestViewController.swift
//  WordVectors
//
//  Feature B — Nearest-word lookup. Type a word, get its 10 nearest neighbours by
//  cosine similarity. Input is lowercased (the corpus is lowercased during
//  preprocessing). Out-of-vocabulary words show a friendly message, never a crash.
//

import UIKit
import OSLog
import WordVectorKit

final class NearestViewController: UIViewController, UITextFieldDelegate, UITableViewDataSource {

    private let wordField = UITextField()
    private let searchButton = UIButton(type: .system)
    private let messageLabel = UILabel()
    private let tableView = UITableView(frame: .zero, style: .plain)

    private var results: [(word: String, similarity: Float)] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Nearest Words"
        view.backgroundColor = .systemBackground
        buildUI()
        ModelStore.shared.addObserver(self) { [weak self] state in
            self?.render(state)
        }
    }

    // MARK: - UI

    private func buildUI() {
        wordField.placeholder = "Enter a word (e.g. king)"
        // Prefilled with a common word that is in the default book's vocabulary, so tapping
        // "Find Nearest" immediately yields real results rather than an OOV message.
        wordField.text = "king"
        wordField.borderStyle = .roundedRect
        wordField.autocapitalizationType = .none
        wordField.autocorrectionType = .no
        wordField.returnKeyType = .search
        wordField.clearButtonMode = .whileEditing
        wordField.delegate = self

        var config = UIButton.Configuration.filled()
        config.title = "Find Nearest"
        searchButton.configuration = config
        searchButton.addTarget(self, action: #selector(searchTapped), for: .touchUpInside)

        messageLabel.font = .preferredFont(forTextStyle: .callout)
        messageLabel.textColor = .secondaryLabel
        messageLabel.numberOfLines = 0

        tableView.dataSource = self
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        tableView.translatesAutoresizingMaskIntoConstraints = false

        let inputStack = UIStackView(arrangedSubviews: [wordField, searchButton, messageLabel])
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

    // MARK: - Actions

    @objc private func searchTapped() {
        view.endEditing(true)
        performSearch()
    }

    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        performSearch()
        return true
    }

    private func performSearch() {
        results = []

        guard let model = ModelStore.shared.embeddings else {
            setMessage("Train a model first (see the Train tab).")
            tableView.reloadData()
            return
        }

        let raw = wordField.text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !raw.isEmpty else {
            setMessage("Type a word to search.")
            tableView.reloadData()
            return
        }

        // The corpus is lowercased during preprocessing, so lowercase the query too.
        let word = raw.lowercased()

        guard model.contains(word) else {
            appLog.info("Nearest query '\(word, privacy: .public)': out of vocabulary.")
            setMessage("'\(word)' is not in the vocabulary. Try a more common word.")
            tableView.reloadData()
            return
        }

        results = model.nearest(to: word, count: 10)
        appLog.info("Nearest query '\(word, privacy: .public)': \(self.results.count, privacy: .public) results.")
        if results.isEmpty {
            setMessage("No neighbours found for '\(word)'.")
        } else {
            setMessage("Nearest words to '\(word)':")
        }
        tableView.reloadData()
    }

    private func setMessage(_ text: String?) {
        messageLabel.text = text
    }

    // MARK: - State

    private func render(_ state: ModelState) {
        switch state {
        case .ready:
            wordField.isEnabled = true
            searchButton.isEnabled = true
            // Prompt the user only if they haven't already run a search this session.
            if results.isEmpty {
                setMessage("Model ready. Enter a word and tap Find Nearest.")
            }
        default:
            wordField.isEnabled = false
            searchButton.isEnabled = false
            results = []
            setMessage("No model yet — train one on the Train tab.")
            tableView.reloadData()
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
