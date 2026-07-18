import Accelerate

extension WordEmbeddings {

    /// Projects the most frequent words onto the first two principal components.
    ///
    /// PCA is useful for plotting embeddings because it preserves as much of the selected
    /// vectors' variation as possible in two axes. The mean is removed independently from
    /// every embedding dimension: subtracting one global mean would let dimensions with a
    /// large offset distort the directions of greatest variation.
    ///
    /// The decomposition is performed directly on the centered word-by-dimension matrix
    /// with LAPACK's single-precision SVD. Forming and diagonalizing a covariance matrix is
    /// intentionally avoided because doing so squares the matrix's condition number and is
    /// less numerically stable for the usual tall, narrow embedding matrix.
    ///
    /// - Parameter wordCount: Maximum number of words to project, taken from the beginning
    ///   of `vocabulary`. Non-positive values select no words.
    /// - Returns: Projected coordinates in the same order as the selected vocabulary words.
    ///   A missing or degenerate component is represented by zero coordinates on that axis.
    public func projected2D(wordCount: Int) -> [(word: String, x: Float, y: Float)] {
        guard wordCount > 0, !vocabulary.isEmpty else { return [] }

        let selectedWords = Array(vocabulary.prefix(wordCount))
        let rowCount = selectedWords.count
        let columnCount = vectorSize
        let zeroProjection = selectedWords.map { (word: $0, x: Float.zero, y: Float.zero) }

        // One centered sample has no variation, and a zero-dimensional embedding has no
        // direction to project onto. Avoid sending either empty shape through the Fortran API.
        guard rowCount >= 2, columnCount > 0 else { return zeroProjection }

        // LAPACK uses 32-bit dimensions on the supported Accelerate platforms. A model this
        // large is not practical in memory, but validate the conversion so this public method
        // remains total rather than trapping on an overflowing integer conversion.
        guard let convertedRows = __CLPK_integer(exactly: rowCount),
              let convertedColumns = __CLPK_integer(exactly: columnCount) else {
            return zeroProjection
        }
        var lapackRows = convertedRows
        var lapackColumns = convertedColumns

        // Store the mathematical word-by-dimension matrix in column-major order from the
        // outset. Each embedding dimension is therefore contiguous, which makes both the
        // per-column centering and the handoff to Fortran direct and unambiguous.
        var centered = [Float](repeating: 0, count: rowCount * columnCount)
        for (row, word) in selectedWords.enumerated() {
            guard let vector = vector(for: word), vector.count == columnCount else {
                return zeroProjection
            }
            for column in 0..<columnCount {
                centered[row + column * rowCount] = vector[column]
            }
        }

        centered.withUnsafeMutableBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            for column in 0..<columnCount {
                let columnAddress = baseAddress + column * rowCount
                var mean: Float = 0
                vDSP_meanv(columnAddress, 1, &mean, vDSP_Length(rowCount))
                var negativeMean = -mean
                vDSP_vsadd(columnAddress, 1,
                           &negativeMean,
                           columnAddress, 1,
                           vDSP_Length(rowCount))
            }
        }

        let componentCount = min(rowCount, columnCount)
        var singularValues = [Float](repeating: 0, count: componentCount)

        // Request no left singular vectors and the reduced V-transpose. For X = U S VT,
        // each row of VT is a right singular vector, i.e. one PCA direction in embedding
        // space. VT is itself column-major, so element (component, dimension) is stored at
        // component + dimension * leadingDimensionVT.
        var jobU: Int8 = 78       // ASCII "N"
        var jobVT: Int8 = 83      // ASCII "S"
        var leadingDimensionA = lapackRows
        var leadingDimensionU: __CLPK_integer = 1
        var leadingDimensionVT = __CLPK_integer(componentCount)
        var unusedU: Float = 0
        var vt = [Float](repeating: 0, count: componentCount * columnCount)
        var info: __CLPK_integer = 0

        // Ask LAPACK for its preferred workspace first. The query still requires valid
        // matrix/output pointers, but LWORK = -1 means no decomposition is performed.
        var matrix = centered
        var workspaceQuery: Float = 0
        var workspaceQuerySize: __CLPK_integer = -1
        sgesvd_(&jobU, &jobVT,
                &lapackRows, &lapackColumns,
                &matrix, &leadingDimensionA,
                &singularValues,
                &unusedU, &leadingDimensionU,
                &vt, &leadingDimensionVT,
                &workspaceQuery, &workspaceQuerySize,
                &info)

        guard info == 0,
              workspaceQuery.isFinite,
              workspaceQuery > 0,
              workspaceQuery <= 2_000_000_000 else {
            return zeroProjection
        }

        let workspaceCount = max(1, Int(workspaceQuery.rounded(.up)))
        guard let convertedWorkspaceCount = __CLPK_integer(exactly: workspaceCount) else {
            return zeroProjection
        }
        var lapackWorkspaceCount = convertedWorkspaceCount

        var workspace = [Float](repeating: 0, count: workspaceCount)
        matrix = centered
        info = 0
        sgesvd_(&jobU, &jobVT,
                &lapackRows, &lapackColumns,
                &matrix, &leadingDimensionA,
                &singularValues,
                &unusedU, &leadingDimensionU,
                &vt, &leadingDimensionVT,
                &workspace, &lapackWorkspaceCount,
                &info)

        guard info == 0 else { return zeroProjection }

        // Singular-vector signs are mathematically arbitrary. Make each sign deterministic
        // by requiring the loading with greatest magnitude (first on ties) to be positive.
        // This prevents an otherwise equivalent decomposition from mirroring a plot axis.
        let availableComponents = min(2, componentCount)
        let leadingSingularValue = singularValues[0]
        let rankTolerance = leadingSingularValue.isFinite
            ? leadingSingularValue * Float(max(rowCount, columnCount)) * Float.ulpOfOne
            : Float.infinity
        var components = [[Float]](repeating: [], count: availableComponents)
        for component in 0..<availableComponents {
            // LAPACK still returns a basis vector for a zero singular value, but that vector
            // does not represent variation in the data. Skip numerically absent components
            // so degenerate axes are exactly zero instead of exposing rounding noise.
            guard singularValues[component].isFinite,
                  singularValues[component] > rankTolerance else {
                continue
            }

            var direction = [Float](repeating: 0, count: columnCount)
            for column in 0..<columnCount {
                direction[column] = vt[component + column * componentCount]
            }

            var anchor = 0
            for column in 1..<columnCount where abs(direction[column]) > abs(direction[anchor]) {
                anchor = column
            }
            if direction[anchor] < 0 {
                var scale: Float = -1
                direction.withUnsafeMutableBufferPointer { directionBuffer in
                    vDSP_vsmul(directionBuffer.baseAddress!, 1,
                               &scale,
                               directionBuffer.baseAddress!, 1,
                               vDSP_Length(columnCount))
                }
            }
            components[component] = direction
        }

        // Multiplying X by V gives the PCA scores. Read each centered word with a stride of
        // rowCount because X remains column-major; each component vector is contiguous.
        var result = zeroProjection
        centered.withUnsafeBufferPointer { centeredBuffer in
            guard let centeredBase = centeredBuffer.baseAddress else { return }
            for row in 0..<rowCount {
                for component in 0..<availableComponents {
                    guard !components[component].isEmpty else { continue }
                    var coordinate: Float = 0
                    components[component].withUnsafeBufferPointer { directionBuffer in
                        vDSP_dotpr(centeredBase + row, vDSP_Stride(rowCount),
                                   directionBuffer.baseAddress!, 1,
                                   &coordinate, vDSP_Length(columnCount))
                    }
                    if component == 0 {
                        result[row].x = coordinate
                    } else {
                        result[row].y = coordinate
                    }
                }
            }
        }
        return result
    }
}
