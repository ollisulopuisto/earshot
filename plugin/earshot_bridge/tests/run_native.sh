#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLAP_INCLUDE_DIR:?Point CLAP_INCLUDE_DIR at the CLAP SDK include directory}"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
flags=(-std=c++17 -g -pthread -fno-omit-frame-pointer)
if [[ ${SANITIZER:-address,undefined} != none ]]; then
  flags+=("-fsanitize=${SANITIZER:-address,undefined}")
fi
"${CXX:-g++}" "${flags[@]}" -I src tests/wav_io_test.cpp -o "$out/wav"
"$out/wav"
"${CXX:-g++}" "${flags[@]}" -I "$CLAP_INCLUDE_DIR" tests/bridge_state_test.cpp -o "$out/state"
"$out/state"
