//
//  SceneDelegate.swift
//  WordVectors
//
//  Created by Adam Wulf on 7/18/26.
//

import UIKit

class SceneDelegate: UIResponder, UIWindowSceneDelegate {

    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = (scene as? UIWindowScene) else { return }

        let window = UIWindow(windowScene: windowScene)
        window.rootViewController = SceneDelegate.makeRootViewController()
        self.window = window
        window.makeKeyAndVisible()

        // Load a cached model if one exists so 2nd+ launches are instant.
        ModelStore.shared.bootstrap()
    }

    /// Builds the three-tab UI: Train, Nearest words, and Word algebra.
    private static func makeRootViewController() -> UIViewController {
        let tabBar = UITabBarController()

        let train = TrainViewController()
        train.tabBarItem = UITabBarItem(
            title: "Train",
            image: UIImage(systemName: "brain.head.profile"),
            selectedImage: nil
        )

        let nearest = NearestViewController()
        nearest.tabBarItem = UITabBarItem(
            title: "Nearest",
            image: UIImage(systemName: "magnifyingglass"),
            selectedImage: nil
        )

        let analogy = AnalogyViewController()
        analogy.tabBarItem = UITabBarItem(
            title: "Word Algebra",
            image: UIImage(systemName: "plus.forwardslash.minus"),
            selectedImage: nil
        )

        tabBar.viewControllers = [
            UINavigationController(rootViewController: train),
            UINavigationController(rootViewController: nearest),
            UINavigationController(rootViewController: analogy),
        ]
        return tabBar
    }

    func sceneDidDisconnect(_ scene: UIScene) {}
    func sceneDidBecomeActive(_ scene: UIScene) {}
    func sceneWillResignActive(_ scene: UIScene) {}
    func sceneWillEnterForeground(_ scene: UIScene) {}
    func sceneDidEnterBackground(_ scene: UIScene) {}
}
