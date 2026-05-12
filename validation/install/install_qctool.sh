#!/usr/bin/env bash
# Install qctool v2 into validation/tools/qctool/.
# Official static x86_64 build from well.ox.ac.uk.
set -euo pipefail

QCTOOL_VERSION="2.2.5"
QCTOOL_TARBALL="qctool_v${QCTOOL_VERSION}-CentOS_Linux7.9-x86_64.tgz"
QCTOOL_URL="https://www.well.ox.ac.uk/~gav/resources/${QCTOOL_TARBALL}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HERE}/tools/qctool"
MARKER="${PREFIX}/.installed"

if [[ -f "${MARKER}" ]]; then
    echo "qctool already installed at ${PREFIX}"
    exit 0
fi

mkdir -p "${PREFIX}/bin"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Downloading qctool v${QCTOOL_VERSION}..."
curl -fL --retry 3 -o "${TMP}/qctool.tgz" "${QCTOOL_URL}"
tar -xzf "${TMP}/qctool.tgz" -C "${TMP}"
# The tarball expands to qctool_v<version>-CentOS_Linux*/qctool
EXTRACTED="$(find "${TMP}" -maxdepth 2 -name qctool -type f -executable | head -1)"
if [[ -z "${EXTRACTED}" ]]; then
    echo "qctool binary not found in tarball" >&2
    exit 1
fi
install -m 0755 "${EXTRACTED}" "${PREFIX}/bin/qctool"
"${PREFIX}/bin/qctool" -help 2>&1 | head -1 || true
echo "${QCTOOL_VERSION}" > "${MARKER}"
echo "qctool installed at ${PREFIX}"
