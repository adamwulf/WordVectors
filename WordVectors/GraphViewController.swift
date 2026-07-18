//
//  GraphViewController.swift
//  WordVectors
//
//  UIKit bridge for placing the SwiftUI graph beside the app's existing view controllers.
//

import SwiftUI

final class GraphViewController: UIHostingController<GraphView> {
    init() {
        super.init(rootView: GraphView())
        title = "Graph"
    }

    @MainActor required dynamic init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder, rootView: GraphView())
        title = "Graph"
    }
}
