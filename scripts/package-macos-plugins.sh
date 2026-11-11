#!/bin/bash
set -euo pipefail
build=${1:?build directory required}
out=${2:?output directory required}
mkdir -p "$out"
temp=$(mktemp -d)
staging="$temp/EarshotBridge"
mkdir -p "$staging"
trap 'rm -rf "$temp"' EXIT
for kind in vst3 component; do
  bundle=$(find "$build" -type d -name "EarshotBridge.$kind" -prune)
  test -n "$bundle"
  test "$(printf '%s\n' "$bundle" | wc -l | tr -d ' ')" = 1
  test -f "$bundle/Contents/Info.plist"
  for key in CFBundleVersion CFBundleShortVersionString; do
    version=$(/usr/libexec/PlistBuddy -c "Print :$key" "$bundle/Contents/Info.plist")
    if ! [[ "$version" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]]; then
      echo "Invalid $key in $bundle" >&2
      exit 1
    fi
  done
  executable=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleExecutable' "$bundle/Contents/Info.plist")
  test -n "$executable"
  binary="$bundle/Contents/MacOS/$executable"
  test -f "$binary"
  test "$(lipo -archs "$binary")" = arm64
  otool -L "$binary"
  if otool -L "$binary" | grep -E '/Users/|/opt/homebrew/|/usr/local/'; then
    echo 'Non-portable dependency in plugin' >&2
    exit 1
  fi
  if [ "$kind" = vst3 ]; then
    nm -gU "$binary" | grep ' _GetPluginFactory$'
  else
    /usr/libexec/PlistBuddy -c 'Print :AudioComponents:0' "$bundle/Contents/Info.plist"
  fi
  # Development signature only. Do not describe this as Developer ID signing.
  codesign --force --sign - "$bundle"
  codesign --verify --deep --strict "$bundle"
  ditto "$bundle" "$staging/EarshotBridge.$kind"
done
cat > "$staging/BUILD.txt" <<INFO
Earshot Bridge experimental arm64 VST3 and AUv2
Source: ${GITHUB_SHA:-local}
Build: ${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-local}/actions/runs/${GITHUB_RUN_ID:-local}
Development/ad-hoc signature only. Not notarized. Never loaded in a DAW.
A separately installed earshot command is required for restoration.
No AAX (Avid SDK required), ARA2 host integration or model weights included.
INFO
name="EarshotBridge-arm64-${GITHUB_RUN_ID:-local}.${GITHUB_RUN_ATTEMPT:-1}.zip"
ditto -c -k --sequesterRsrc "$staging" "$out/$name"
(cd "$out" && shasum -a 256 "$name" > "$name.sha256" && shasum -a 256 -c "$name.sha256")
find "$staging" -type f -maxdepth 5
