// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "WordVectorKit",
    platforms: [
        .iOS(.v15),
        .macOS(.v12)
    ],
    products: [
        .library(
            name: "WordVectorKit",
            targets: ["WordVectorKit"]
        )
    ],
    targets: [
        .target(
            name: "WordVectorKit",
            dependencies: [],
            linkerSettings: [
                // Accelerate is a system framework used for vDSP dot products / AXPY updates.
                .linkedFramework("Accelerate")
            ]
        ),
        .testTarget(
            name: "WordVectorKitTests",
            dependencies: ["WordVectorKit"]
        )
    ]
)
