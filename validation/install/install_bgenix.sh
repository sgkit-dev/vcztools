#!/usr/bin/env bash
# Build bgenix from source into validation/tools/bgenix/.
# The reference BGEN library distributes only source; we build via waf.
set -euo pipefail

BGEN_URL="https://enkre.net/cgi-bin/code/bgen/tarball/release/bgen.tgz"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${HERE}/tools/bgenix"
MARKER="${PREFIX}/.installed"

if [[ -f "${MARKER}" ]]; then
    echo "bgenix already installed at ${PREFIX}"
    exit 0
fi

mkdir -p "${PREFIX}/bin" "${PREFIX}/src"
TMP="${PREFIX}/src"

echo "Downloading bgen reference library..."
curl -fL --retry 3 -o "${TMP}/bgen.tgz" "${BGEN_URL}"
tar -xzf "${TMP}/bgen.tgz" -C "${TMP}"
SRCDIR="$(find "${TMP}" -maxdepth 1 -mindepth 1 -type d -name 'bgen*' | head -1)"
if [[ -z "${SRCDIR}" ]]; then
    echo "bgen source tree not found after extract" >&2
    exit 1
fi

cd "${SRCDIR}"
# Patch View.cpp for GCC 13+: `std::ios::streampos` in a block-scoped
# variable triggers a spurious "not declared" error there. Using
# `auto` lets the compiler deduce the same streampos type.
sed -i 's/std::ios::streampos origin =/auto origin =/' src/View.cpp
python3 ./waf configure --prefix="${PREFIX}"
python3 ./waf
# bgenix + cat-bgen land in build/release/apps after a plain `./waf` build.
for bin in bgenix cat-bgen; do
    BUILT="$(find build -type f -name "${bin}" | head -1)"
    if [[ -z "${BUILT}" ]]; then
        echo "${bin} not built" >&2
        exit 1
    fi
    install -m 0755 "${BUILT}" "${PREFIX}/bin/${bin}"
done

"${PREFIX}/bin/bgenix" -help 2>&1 | head -1 || true
echo "$(date -u +%Y-%m-%d)" > "${MARKER}"
echo "bgenix installed at ${PREFIX}"
