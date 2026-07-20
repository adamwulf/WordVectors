import Accelerate

/// One word assigned to a cluster.
public struct WordCluster: Sendable {
    public let word: String
    public let cluster: Int   // 0..<k, the index of the assigned cluster
}

/// The result of clustering the vocabulary's full-dimensional vectors.
public struct ClusteringResult: Sendable {
    /// Cluster assignment per word, in the SAME order as `vocabulary.prefix(wordCount)`.
    public let assignments: [WordCluster]
    /// Number of clusters actually produced (may be < requested k if the vocabulary is tiny).
    public let clusterCount: Int
    /// For each cluster index 0..<clusterCount, the words nearest that cluster's centroid,
    /// most-central first. Used by the UI to auto-label clusters. Length: clusterCount.
    /// Each inner array holds up to `labelsPerCluster` words.
    public let representatives: [[String]]
    /// Number of words in each cluster, index-aligned with cluster indices 0..<clusterCount.
    public let sizes: [Int]
}

extension WordEmbeddings {

    /// Clusters the most frequent `wordCount` words on their FULL-dimensional vectors using
    /// k-means with k-means++ initialization.
    ///
    /// ## Why unit-normalize (spherical k-means)
    /// word2vec captures meaning in the *direction* of a vector, not its magnitude, so semantic
    /// similarity is cosine similarity. Every selected vector is L2-normalized to unit length
    /// before clustering, and each centroid is re-normalized to unit length after every update.
    /// On the unit sphere squared Euclidean distance is a strictly increasing function of cosine
    /// distance — `‖a − b‖² = 2 − 2·cos(a, b)` for unit vectors — so assigning each word to its
    /// nearest centroid by squared Euclidean distance is exactly assigning it to the most
    /// cosine-similar centroid. This lets vDSP do the heavy lifting with plain Euclidean kernels
    /// while the clusters follow cosine geometry, which is far cleaner for word embeddings than
    /// clustering on raw (magnitude-carrying) Euclidean distance.
    ///
    /// ## Determinism
    /// The result depends only on the stored vectors and `seed`. A seeded SplitMix64 generator
    /// drives k-means++ seeding; there is no `Date`, `arc4random`, or system RNG anywhere, and ties
    /// are always broken toward lower indices. For a given build/target, this makes the result
    /// reproducible: the same seed over the same binary produces identical assignments, sizes, and
    /// representatives run after run. It is NOT guaranteed to be byte-for-byte identical across
    /// different platforms — float summation is non-associative, and the vDSP/BLAS reduction order
    /// and SIMD width can differ across CPU microarchitectures, so the accumulated floating-point
    /// sums (and thus the exact partition on near-tied points) may differ from one target to another.
    ///
    /// ## Empty-cluster strategy
    /// If a centroid loses every member during Lloyd's iteration (a real possibility with
    /// k-means++ on clumped data), it is *re-seeded* rather than dropped, which keeps
    /// `clusterCount` stable and predictable. The re-seed point is the single word that is
    /// currently farthest from its own assigned centroid — the point most poorly served by the
    /// current partition — chosen deterministically (lowest index on ties). This is the classic
    /// farthest-point re-seed and it strictly reduces the objective while guaranteeing every
    /// returned cluster is non-empty.
    ///
    /// - Parameters:
    ///   - k: desired number of clusters (clamped to [1, wordCount]).
    ///   - wordCount: how many of the most-frequent words to cluster (same convention as projected2D).
    ///   - labelsPerCluster: how many representative words to return per cluster (e.g. 6).
    ///   - seed: deterministic RNG seed so repeated runs give identical clusters.
    /// - Returns: assignments index-aligned with `vocabulary.prefix(wordCount)`, plus per-cluster
    ///   representative words and sizes. Returns empty assignments if wordCount<=0 or vocab empty.
    public func cluster(
        k: Int,
        wordCount: Int,
        labelsPerCluster: Int,
        seed: UInt64
    ) -> ClusteringResult {
        let empty = ClusteringResult(assignments: [], clusterCount: 0, representatives: [], sizes: [])

        // Degenerate: nothing to cluster. Matches projected2D's convention exactly.
        guard wordCount > 0, !vocabulary.isEmpty else { return empty }

        let selectedWords = Array(vocabulary.prefix(wordCount))
        let pointCount = selectedWords.count
        let dimension = vectorSize

        // A zero-dimensional embedding has no geometry to cluster on. Put every word in one
        // cluster so the public method stays total instead of dividing by an empty vector.
        guard dimension > 0 else {
            let assignments = selectedWords.map { WordCluster(word: $0, cluster: 0) }
            return ClusteringResult(
                assignments: assignments,
                clusterCount: 1,
                representatives: [Array(selectedWords.prefix(max(1, labelsPerCluster)))],
                sizes: [pointCount]
            )
        }

        // Clamp k into [1, pointCount]. k >= pointCount degenerates to "each word its own
        // cluster", which the loop below reaches naturally once every centroid is a distinct point.
        let clampedK = min(max(k, 1), pointCount)

        // Build a unit-normalized, row-major copy of just the selected vectors. Working on a
        // private contiguous buffer keeps the vDSP kernels tight and leaves the model untouched.
        // A word whose stored vector is missing or zero-length is treated as the origin; it
        // normalizes to the zero vector and is handled explicitly during assignment.
        var points = [Float](repeating: 0, count: pointCount * dimension)
        points.withUnsafeMutableBufferPointer { pointsBuffer in
            withUnsafeContiguousVectors { storageBase, storedRowCount in
                for row in 0..<pointCount {
                    let destination = pointsBuffer.baseAddress! + row * dimension
                    guard row < storedRowCount else { continue }
                    let source = storageBase + row * dimension
                    var sumOfSquares: Float = 0
                    vDSP_svesq(source, 1, &sumOfSquares, vDSP_Length(dimension))
                    let norm = sumOfSquares.squareRoot()
                    if norm > 0 {
                        var reciprocal = 1 / norm
                        vDSP_vsmul(source, 1, &reciprocal, destination, 1, vDSP_Length(dimension))
                    }
                    // norm == 0 leaves the row as the already-zeroed vector.
                }
            }
        }

        var generator = SplitMix64(seed: seed)

        // k-means++ seeding on the unit vectors, then Lloyd's iterations under the same metric.
        var centroids = kMeansPlusPlusCentroids(
            points: points,
            pointCount: pointCount,
            dimension: dimension,
            k: clampedK,
            generator: &generator
        )

        var assignmentByPoint = [Int](repeating: 0, count: pointCount)
        let maxIterations = 50

        points.withUnsafeBufferPointer { pointsBuffer in
            let pointsBase = pointsBuffer.baseAddress!

            for _ in 0..<maxIterations {
                var changed = false

                // Assignment step: nearest centroid by squared Euclidean distance (= cosine
                // distance on the sphere). Ties break toward the lower centroid index.
                centroids.withUnsafeBufferPointer { centroidsBuffer in
                    let centroidsBase = centroidsBuffer.baseAddress!
                    for point in 0..<pointCount {
                        let pointBase = pointsBase + point * dimension
                        var bestCluster = 0
                        var bestDistance = Float.infinity
                        for cluster in 0..<clampedK {
                            let centroidBase = centroidsBase + cluster * dimension
                            var distanceSquared: Float = 0
                            vDSP_distancesq(pointBase, 1, centroidBase, 1,
                                            &distanceSquared, vDSP_Length(dimension))
                            if distanceSquared < bestDistance {
                                bestDistance = distanceSquared
                                bestCluster = cluster
                            }
                        }
                        if assignmentByPoint[point] != bestCluster {
                            assignmentByPoint[point] = bestCluster
                            changed = true
                        }
                    }
                }

                // Update step: each centroid becomes the (re-normalized) mean of its members.
                var counts = [Int](repeating: 0, count: clampedK)
                var sums = [Float](repeating: 0, count: clampedK * dimension)
                sums.withUnsafeMutableBufferPointer { sumsBuffer in
                    let sumsBase = sumsBuffer.baseAddress!
                    for point in 0..<pointCount {
                        let cluster = assignmentByPoint[point]
                        counts[cluster] += 1
                        let sumBase = sumsBase + cluster * dimension
                        vDSP_vadd(sumBase, 1,
                                  pointsBase + point * dimension, 1,
                                  sumBase, 1, vDSP_Length(dimension))
                    }
                }

                // Re-seed any emptied centroid to the worst-served point (see doc comment).
                reseedEmptyClusters(
                    counts: &counts,
                    sums: &sums,
                    assignmentByPoint: &assignmentByPoint,
                    centroids: centroids,
                    points: pointsBase,
                    pointCount: pointCount,
                    dimension: dimension,
                    k: clampedK
                )

                // Divide the accumulated sums by the counts, then project back onto the unit
                // sphere so the next assignment step keeps measuring cosine distance.
                sums.withUnsafeMutableBufferPointer { sumsBuffer in
                    let sumsBase = sumsBuffer.baseAddress!
                    for cluster in 0..<clampedK {
                        let sumBase = sumsBase + cluster * dimension
                        let count = counts[cluster]
                        if count > 0 {
                            var reciprocal = 1 / Float(count)
                            vDSP_vsmul(sumBase, 1, &reciprocal, sumBase, 1, vDSP_Length(dimension))
                        }
                        var sumOfSquares: Float = 0
                        vDSP_svesq(sumBase, 1, &sumOfSquares, vDSP_Length(dimension))
                        let norm = sumOfSquares.squareRoot()
                        if norm > 0 {
                            var reciprocal = 1 / norm
                            vDSP_vsmul(sumBase, 1, &reciprocal, sumBase, 1, vDSP_Length(dimension))
                        }
                    }
                }
                centroids = sums

                // Early exit: a full pass with no reassignment means Lloyd's has converged.
                if !changed { break }
            }
        }

        return buildResult(
            selectedWords: selectedWords,
            assignmentByPoint: assignmentByPoint,
            centroids: centroids,
            points: points,
            pointCount: pointCount,
            dimension: dimension,
            k: clampedK,
            labelsPerCluster: labelsPerCluster
        )
    }

    // MARK: - k-means++ initialization

    /// Chooses `k` initial centroids from the unit vectors with D²-weighted k-means++ sampling.
    ///
    /// The first centroid is drawn uniformly at random (via the seeded generator); each further
    /// centroid is drawn with probability proportional to its squared distance from the nearest
    /// already-chosen centroid. This spreads the seeds out, which gives k-means far more stable,
    /// higher-quality minima than uniform seeding. All randomness flows through `generator`, so
    /// the seeding is fully deterministic.
    private func kMeansPlusPlusCentroids(
        points: [Float],
        pointCount: Int,
        dimension: Int,
        k: Int,
        generator: inout SplitMix64
    ) -> [Float] {
        var centroids = [Float](repeating: 0, count: k * dimension)

        points.withUnsafeBufferPointer { pointsBuffer in
            let pointsBase = pointsBuffer.baseAddress!

            // Nearest-centroid squared distance for every point, updated as centroids are added.
            var nearestDistanceSquared = [Float](repeating: Float.infinity, count: pointCount)

            func adoptCentroid(_ pointIndex: Int, at slot: Int) {
                centroids.withUnsafeMutableBufferPointer { centroidsBuffer in
                    let destination = centroidsBuffer.baseAddress! + slot * dimension
                    cblas_scopy(Int32(dimension), pointsBase + pointIndex * dimension, 1, destination, 1)
                }
                // Refresh each point's distance to the nearest centroid chosen so far.
                let centroidBase = pointsBase + pointIndex * dimension
                for point in 0..<pointCount {
                    var distanceSquared: Float = 0
                    vDSP_distancesq(pointsBase + point * dimension, 1, centroidBase, 1,
                                    &distanceSquared, vDSP_Length(dimension))
                    if distanceSquared < nearestDistanceSquared[point] {
                        nearestDistanceSquared[point] = distanceSquared
                    }
                }
            }

            // First centroid: uniform over the points.
            let firstIndex = Int(generator.next(upperBound: UInt64(pointCount)))
            adoptCentroid(firstIndex, at: 0)

            for slot in 1..<k {
                // Total weight is the sum of D². If every remaining point coincides with an
                // existing centroid (total == 0), the data has fewer than `slot` distinct
                // directions; fall back to the first strictly-unclaimed index so centroids stay
                // distinct and deterministic instead of stacking on one point.
                var total: Float = 0
                for point in 0..<pointCount where nearestDistanceSquared[point] > 0 {
                    total += nearestDistanceSquared[point]
                }

                var chosen = -1
                if total > 0 {
                    // Sample a target in [0, total) and walk the cumulative D² weights. Zero-weight
                    // points (those coinciding with an existing centroid) are skipped so the walk
                    // can never adopt a duplicate of an already-chosen centroid — even when
                    // `nextUnitFloat()` returns exactly 0.0, which would otherwise let a leading
                    // zero-weight point satisfy `cumulative >= target` on the first step.
                    let target = generator.nextUnitFloat() * total
                    var cumulative: Float = 0
                    for point in 0..<pointCount where nearestDistanceSquared[point] > 0 {
                        cumulative += nearestDistanceSquared[point]
                        if cumulative >= target {
                            chosen = point
                            break
                        }
                    }
                    // Rounding guard: fall back to the last positive-weight index (never a
                    // zero-weight point). The `total > 0` branch is only entered when at least one
                    // such point exists, so this always finds one; the -1 default is defensive.
                    if chosen < 0 {
                        for point in stride(from: pointCount - 1, through: 0, by: -1)
                        where nearestDistanceSquared[point] > 0 {
                            chosen = point
                            break
                        }
                    }
                } else {
                    for point in 0..<pointCount where nearestDistanceSquared[point] > 0 {
                        chosen = point
                        break
                    }
                    // All points already coincide with a centroid: reuse the first point. The
                    // clustering still converges; duplicate centroids simply merge in effect.
                    if chosen < 0 { chosen = 0 }
                }
                adoptCentroid(chosen, at: slot)
            }
        }

        return centroids
    }

    // MARK: - Empty-cluster re-seeding

    /// Re-seeds every centroid that lost all its members, keeping `clusterCount` fixed.
    ///
    /// The replacement point is the one currently farthest from its own centroid (largest
    /// intra-cluster squared distance), which is the point the partition serves worst; ties break
    /// toward the lower index for determinism. The moved point is removed from its old cluster's
    /// running sum/count and becomes the emptied cluster's sole member, so the caller's mean
    /// computation stays exact.
    private func reseedEmptyClusters(
        counts: inout [Int],
        sums: inout [Float],
        assignmentByPoint: inout [Int],
        centroids: [Float],
        points: UnsafePointer<Float>,
        pointCount: Int,
        dimension: Int,
        k: Int
    ) {
        for cluster in 0..<k where counts[cluster] == 0 {
            // Find the worst-served point that still has company (its cluster has >1 member),
            // so donating it never empties another cluster.
            var worstPoint = -1
            var worstDistance: Float = -1
            centroids.withUnsafeBufferPointer { centroidsBuffer in
                let centroidsBase = centroidsBuffer.baseAddress!
                for point in 0..<pointCount {
                    let owner = assignmentByPoint[point]
                    guard counts[owner] > 1 else { continue }
                    var distanceSquared: Float = 0
                    vDSP_distancesq(points + point * dimension, 1,
                                    centroidsBase + owner * dimension, 1,
                                    &distanceSquared, vDSP_Length(dimension))
                    if distanceSquared > worstDistance {
                        worstDistance = distanceSquared
                        worstPoint = point
                    }
                }
            }

            // With k <= pointCount there is always a donor cluster of size > 1 while some cluster
            // is empty, so worstPoint is found. Guard anyway to stay total.
            guard worstPoint >= 0 else { continue }

            let donor = assignmentByPoint[worstPoint]
            sums.withUnsafeMutableBufferPointer { sumsBuffer in
                let sumsBase = sumsBuffer.baseAddress!
                // Remove the point from its donor cluster's sum, then make it the new cluster's sum.
                vDSP_vsub(points + worstPoint * dimension, 1,
                          sumsBase + donor * dimension, 1,
                          sumsBase + donor * dimension, 1, vDSP_Length(dimension))
                cblas_scopy(Int32(dimension), points + worstPoint * dimension, 1,
                            sumsBase + cluster * dimension, 1)
            }
            counts[donor] -= 1
            counts[cluster] = 1
            assignmentByPoint[worstPoint] = cluster
        }
    }

    // MARK: - Result assembly

    /// Builds the public result: assignments in vocabulary order, per-cluster sizes, and the
    /// `labelsPerCluster` most-central words per cluster (nearest the final centroid, closest
    /// first; ties break toward the earlier vocabulary word for determinism).
    private func buildResult(
        selectedWords: [String],
        assignmentByPoint: [Int],
        centroids: [Float],
        points: [Float],
        pointCount: Int,
        dimension: Int,
        k: Int,
        labelsPerCluster: Int
    ) -> ClusteringResult {
        let assignments = zip(selectedWords, assignmentByPoint).map { word, cluster in
            WordCluster(word: word, cluster: cluster)
        }

        var sizes = [Int](repeating: 0, count: k)
        for cluster in assignmentByPoint {
            sizes[cluster] += 1
        }

        // For each cluster gather (point index, distance²-to-centroid), sort most-central first,
        // and keep up to labelsPerCluster words. A cluster is never empty here (re-seeding
        // guarantees it), so every representatives[c] is non-empty.
        let labelCap = max(0, labelsPerCluster)
        var representatives = [[String]](repeating: [], count: k)
        var membersByCluster = [[(index: Int, distanceSquared: Float)]](repeating: [], count: k)

        points.withUnsafeBufferPointer { pointsBuffer in
            centroids.withUnsafeBufferPointer { centroidsBuffer in
                let pointsBase = pointsBuffer.baseAddress!
                let centroidsBase = centroidsBuffer.baseAddress!
                for point in 0..<pointCount {
                    let cluster = assignmentByPoint[point]
                    var distanceSquared: Float = 0
                    vDSP_distancesq(pointsBase + point * dimension, 1,
                                    centroidsBase + cluster * dimension, 1,
                                    &distanceSquared, vDSP_Length(dimension))
                    membersByCluster[cluster].append((index: point, distanceSquared: distanceSquared))
                }
            }
        }

        for cluster in 0..<k {
            let ordered = membersByCluster[cluster].sorted { a, b in
                if a.distanceSquared != b.distanceSquared {
                    return a.distanceSquared < b.distanceSquared
                }
                return a.index < b.index // earlier vocabulary word wins ties
            }
            representatives[cluster] = ordered.prefix(labelCap).map { selectedWords[$0.index] }
        }

        return ClusteringResult(
            assignments: assignments,
            clusterCount: k,
            representatives: representatives,
            sizes: sizes
        )
    }
}

/// A tiny, fully deterministic pseudo-random generator (SplitMix64).
///
/// k-means++ needs a stream of random numbers, but the clustering must be reproducible from a
/// seed alone — no `Date`, no `arc4random`, no `SystemRandomNumberGenerator`. SplitMix64 is the
/// standard seed generator behind xoshiro/xoroshiro; it is a single 64-bit state advanced by a
/// fixed odd increment and finalized with a well-known avalanche mix, so the same seed always
/// yields the same sequence on every platform.
struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    /// Advances the state and returns the next 64-bit value (SplitMix64 finalizer).
    mutating func next() -> UInt64 {
        state = state &+ 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// A uniform integer in `0..<upperBound` using rejection sampling to avoid modulo bias.
    /// Returns 0 when `upperBound` is 0 so callers never trap on an empty range.
    mutating func next(upperBound: UInt64) -> UInt64 {
        guard upperBound > 0 else { return 0 }
        // Reject the top partial block so every value in range is equally likely.
        let limit = UInt64.max - (UInt64.max % upperBound)
        while true {
            let value = next()
            if value < limit {
                return value % upperBound
            }
        }
    }

    /// A uniform Float in the half-open interval [0, 1). Uses the top 24 bits so every
    /// representable single-precision fraction is reachable without bias.
    mutating func nextUnitFloat() -> Float {
        let bits = next() >> 40 // keep the high 24 bits
        return Float(bits) * (1.0 / Float(1 << 24))
    }
}
