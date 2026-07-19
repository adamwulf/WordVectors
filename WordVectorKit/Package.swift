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
        ),
        .executable(
            name: "w2v-bench",
            targets: ["w2v-bench"]
        )
    ],
    targets: [
        .target(
            name: "WordVectorKit",
            dependencies: [],
            swiftSettings: [
                .unsafeFlags(["-Ounchecked"], .when(configuration: .release))
            ],
            linkerSettings: [
                // Accelerate is a system framework used for vDSP dot products / AXPY updates.
                .linkedFramework("Accelerate")
            ]
        ),
        .executableTarget(
            name: "w2v-bench",
            dependencies: ["WordVectorKit"]
        ),
        .testTarget(
            name: "WordVectorKitTests",
            dependencies: ["WordVectorKit"]
        )
    ]
)
