#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # presentation/coding
PRES="$(cd "${HERE}/.." && pwd)"                         # presentation/
eog --fullscreen "${PRES}/images/lattices_merged.gif"
