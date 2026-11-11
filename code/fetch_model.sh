#!/bin/bash
# fetch_model.sh — DEPRECATED. Use pget.sh instead.
#
# This script had two bugs that corrupted a 3 GB download:
#   1. it shared one parts directory across runs, so changing N reused stale
#      part files with different chunk boundaries;
#   2. it hardcoded a cat of part0..part15 regardless of N.
# pget.sh fixes both: dedicated "<out>.parts" dir and a dynamic assembly loop.
#
# Kept only as a warning. Do not use.
echo "fetch_model.sh is deprecated and produced a corrupt file; use:" >&2
echo "  code/pget.sh <url> <outfile> <parts>" >&2
exit 1
