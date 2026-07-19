import Foundation

/// A way of scoring how close two word vectors are. Chosen by the user in the Explore and
/// Word Algebra tabs to compare how different notions of "closeness" rank the vocabulary.
///
/// Each metric produces a `score` per candidate word. `nearest`/`analogy` always return their
/// results *best-first*, but "best" means different things per metric — `isHigherBetter`
/// captures that so callers (and the sort) don't hard-code the direction:
///   - `.cosine` and `.dotProduct` are *similarities*: larger is closer, so best-first is descending.
///   - `.euclidean` is a *distance*: smaller is closer, so best-first is ascending.
public enum DistanceMetric: String, CaseIterable, Sendable {
    /// Angle between the vectors, ignoring magnitude. Range −1…1; 1 is identical direction.
    case cosine
    /// Raw dot product, unnormalized. Rewards large-magnitude (often frequent) vectors.
    case dotProduct
    /// Straight-line (L2) distance. 0 is identical; larger is farther apart.
    case euclidean

    /// A short label for the metric, suitable for a picker segment.
    public var displayName: String {
        switch self {
        case .cosine: return "Cosine"
        case .dotProduct: return "Dot"
        case .euclidean: return "Euclidean"
        }
    }

    /// The word used for the score column's header under this metric. Cosine and dot product
    /// are similarities ("Score"); Euclidean is a distance ("Distance").
    public var scoreColumnTitle: String {
        switch self {
        case .cosine, .dotProduct: return "Score"
        case .euclidean: return "Distance"
        }
    }

    /// Whether a larger score means a closer match. True for similarities, false for distances.
    /// Used to sort results best-first without the ranking code knowing the specific metric.
    public var isHigherBetter: Bool {
        switch self {
        case .cosine, .dotProduct: return true
        case .euclidean: return false
        }
    }
}
