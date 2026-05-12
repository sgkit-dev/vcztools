#!/usr/bin/env bash
# Install REGENIE into validation/tools/regenie/.
# Pre-built statically-linked Linux binary from the GitHub release.
set -euo pipefail

REGENIE_VERSION="4.1"
REGENIE_ASSET="regenie_v${REGENIE_VERSION}.gz_x86_64_Linux.zip"
REGENIE_URL="https://github.com/rgcgithub/regenie/releases/download/v${REGENIE_VERSION}/${REGENIE_ASSET}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HERE}/tools/regenie"
MARKER="${PREFIX}/.installed"

if [[ -f "${MARKER}" ]]; then
    echo "regenie already installed at ${PREFIX}"
    exit 0
fi

mkdir -p "${PREFIX}/bin"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Downloading REGENIE v${REGENIE_VERSION}..."
curl -fL --retry 3 -o "${TMP}/regenie.zip" "${REGENIE_URL}"
unzip -q "${TMP}/regenie.zip" -d "${TMP}/regenie"
BIN="$(find "${TMP}/regenie" -maxdepth 2 -name 'regenie_*' -type f -executable | head -1)"
if [[ -z "${BIN}" ]]; then
    BIN="$(find "${TMP}/regenie" -maxdepth 2 -name 'regenie*' -type f | head -1)"
fi
if [[ -z "${BIN}" ]]; then
    echo "regenie binary not found in zip" >&2
    exit 1
fi
install -m 0755 "${BIN}" "${PREFIX}/bin/regenie"
"${PREFIX}/bin/regenie" --version 2>&1 | head -3 || true
echo "${REGENIE_VERSION}" > "${MARKER}"
echo "regenie installed at ${PREFIX}"
