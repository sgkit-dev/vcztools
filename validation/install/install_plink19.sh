#!/usr/bin/env bash
# Install PLINK 1.9 into validation/tools/plink19/.
# Static x86_64 Linux binary from the official cog-genomics distribution.
set -euo pipefail

PLINK19_VERSION="20231211"
PLINK19_URL="https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_${PLINK19_VERSION}.zip"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HERE}/tools/plink19"
MARKER="${PREFIX}/.installed"

if [[ -f "${MARKER}" ]]; then
    echo "plink19 already installed at ${PREFIX}"
    exit 0
fi

mkdir -p "${PREFIX}/bin"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "Downloading PLINK 1.9 ${PLINK19_VERSION}..."
curl -fL --retry 3 -o "${TMP}/plink.zip" "${PLINK19_URL}"
unzip -q "${TMP}/plink.zip" -d "${TMP}/plink"
install -m 0755 "${TMP}/plink/plink" "${PREFIX}/bin/plink"
"${PREFIX}/bin/plink" --version | head -1
echo "${PLINK19_VERSION}" > "${MARKER}"
echo "plink19 installed at ${PREFIX}"
