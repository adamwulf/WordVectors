#!/bin/sh

set -e

# Build a tokenized push URL for origin from input username and token
github_authed_url() {
    # Get the original URL
    original_url=$(git remote get-url origin)
    # Extract the rest of the URL after the protocol
    rest_url=${original_url#*://}
    # Extract everything after github.com/
    github_path=${rest_url#*github.com/}
    # Construct the new URL with the token
    echo "https://${1}:${2}@github.com/${github_path}"
}

# Next, automatically tag the build in Github

if [ "$CI_XCODEBUILD_EXIT_CODE" -eq 0 ]; then
    echo "Build succeeded"
    tag="build/$CI_BUILD_NUMBER"

    # GITHUB_TOKEN can be configured in github -> account settings -> developer settings -> personal access tokens -> fine grained token -> read/write access to code
    remote_url=$(github_authed_url $GITHUB_USERNAME $GITHUB_TOKEN)

    # --- Generate TestFlight "What to Test" notes -------------------------
    # Xcode Cloud auto-uploads TestFlight/WhatToTest.<locale>.txt as the
    # tester-facing "What to Test" notes. Apple requires the TestFlight folder
    # to sit next to the Xcode project/workspace; WordVectors.xcodeproj is at the
    # repository root, so the folder lives at the repo root (not under the
    # WordVectors/ source subfolder). We generate it here from the commit
    # range since the last build, so nothing is committed to the repo — the
    # file lives only in this ephemeral clone. Gated on the signed-app path so
    # it runs only for distribution builds that actually upload to TestFlight.
    if [ -n "$CI_APP_STORE_SIGNED_APP_PATH" ] && [ -d "$CI_APP_STORE_SIGNED_APP_PATH" ]; then
        # Xcode Cloud clones shallow and without tags; deepen so the tag range
        # and commit history are reachable for git describe / git log.
        git fetch --unshallow 2>/dev/null || git fetch --deepen 100 2>/dev/null || true
        git fetch "$remote_url" 'refs/tags/*:refs/tags/*' 2>/dev/null || true

        # Previous build's tag = lower bound. build/* numbers can skip (failed
        # or manual runs), so use the most recent reachable tag, not N-1. This
        # runs before the new build/$CI_BUILD_NUMBER tag is created below, so
        # HEAD's nearest build/* tag is genuinely the previous build.
        prev_tag=$(git describe --tags --match 'build/*' --abbrev=0 HEAD 2>/dev/null || echo "")
        if [ -n "$prev_tag" ]; then
            notes=$(git log --no-merges --pretty=format:'- %s' "${prev_tag}..HEAD")
        else
            # First build ever: no prior tag. Cap to the last 20 commits.
            notes=$(git log --no-merges -20 --pretty=format:'- %s' HEAD)
        fi

        # Empty range (e.g. rebuild of an already-tagged commit) → fallback.
        if [ -z "$notes" ]; then
            notes="- Maintenance build (no source changes since last build)"
        fi

        notes_dir="$CI_PRIMARY_REPOSITORY_PATH/TestFlight"
        mkdir -p "$notes_dir"
        printf '%s\n' "$notes" > "$notes_dir/WhatToTest.en-US.txt"
        echo "Wrote What-to-Test notes (previous tag: ${prev_tag:-none})"
    fi
    # ---------------------------------------------------------------------

    # This script runs once per xcodebuild action, so the same build can reach
    # here more than once. Skip if the tag is already on the remote — never
    # force-push, so an existing build/N tag can't be moved to a new commit.
    if git ls-remote --tags "$remote_url" "refs/tags/$tag" | grep -q "refs/tags/$tag"; then
        echo "Tag $tag already exists on remote, skipping."
    else
        echo "Tagging $tag"
        git tag -a -m "Build $CI_BUILD_NUMBER" $tag
        git push "$remote_url" --tags
        echo "Successfully pushed tag to remote repo."
    fi
else
    echo "Build failed"
    # Build failed
fi
