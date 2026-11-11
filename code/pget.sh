#!/bin/bash
# pget.sh URL OUTFILE [PARTS] — parallel range download with resume.
# Needed because this network throttles each connection to ~90 KB/s.
set -u
URL="$1"; OUT="$2"; N="${3:-8}"
TMP="${OUT}.parts"; mkdir -p "$TMP"

SIZE=$(curl -sIL --max-time 40 "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
if [ -z "${SIZE:-}" ] || [ "$SIZE" -lt 1000 ]; then echo "FATAL: no size"; exit 1; fi
echo "size=$SIZE parts=$N"

CHUNK=$(( (SIZE + N - 1) / N ))
pids=()
for i in $(seq 0 $((N-1))); do
  S=$((i*CHUNK)); E=$((S+CHUNK-1)); [ $E -ge $SIZE ] && E=$((SIZE-1))
  WANT=$((E-S+1)); P="$TMP/part$i"
  (
    for a in $(seq 1 300); do
      HAVE=$(stat -c%s "$P" 2>/dev/null || echo 0)
      [ "$HAVE" -ge "$WANT" ] && break
      curl -sL --max-time 1800 -r $((S+HAVE))-$E "$URL" >> "$P"
      sleep 1
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

: > "$OUT"
for i in $(seq 0 $((N-1))); do cat "$TMP/part$i" >> "$OUT"; done
GOT=$(stat -c%s "$OUT")
echo "got=$GOT expected=$SIZE"
[ "$GOT" -eq "$SIZE" ] && { rm -rf "$TMP"; echo OK; } || { echo MISMATCH; exit 1; }
