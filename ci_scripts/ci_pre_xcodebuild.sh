#!/bin/sh

# Stamp the build number from Xcode Cloud's monotonic counter.
# CI_BUILD_NUMBER is set by Xcode Cloud and increments per workflow run,
# guaranteeing App Store Connect sees a new value every upload.
#
# WordVectors has no shared version xcconfig; CURRENT_PROJECT_VERSION is
# defined directly in the Xcode project build settings (the app target uses
# GENERATE_INFOPLIST_FILE = YES and its committed Info.plist does not declare
# CFBundleVersion, so there is no CFBundleVersion in Info.plist to stamp). We
# therefore rewrite every CURRENT_PROJECT_VERSION assignment in project.pbxproj
# so all build configurations pick up the CI build number.

set -e

if [ -z "$CI_BUILD_NUMBER" ]; then
    echo "CI_BUILD_NUMBER is not set; skipping build number stamp."
    exit 0
fi

PBXPROJ="$CI_PRIMARY_REPOSITORY_PATH/WordVectors.xcodeproj/project.pbxproj"

if [ ! -f "$PBXPROJ" ]; then
    echo "Error: project file not found at $PBXPROJ"
    exit 1
fi

echo "Setting CURRENT_PROJECT_VERSION to $CI_BUILD_NUMBER in $PBXPROJ"
sed -i '' -E "s/CURRENT_PROJECT_VERSION = [^;]*;/CURRENT_PROJECT_VERSION = $CI_BUILD_NUMBER;/g" "$PBXPROJ"
