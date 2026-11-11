#!/usr/bin/env python3
"""Probe what an API key actually exposes. Never prints the key.

Usage:
  set -a; . ./.env; set +a
  python tools/api_probe.py [--provider google]

Writes proof/api_surface.md (overwrites the measured table).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

CANDIDATES = [
    "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash",
    "gemini-2.5-flash-lite", "gemma-4-31b-it", "gemma-4-26b-a4b-it",
    "gemini-flash-latest", "gemini-pro-latest", "gemini-2.0-flash-001",
]


def post(url: str, body: dict, timeout: int = 40) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        try:
            return json.load(e)
        except Exception:
            return {"error": {"message": str(e)}}


def google(key: str) -> list[dict]:
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    rows = []
    for m in CANDIDATES:
        d = post(f"{base}/{m}:generateContent?key={key}", {
            "contents": [{"parts": [{"text": "The capital of France is"}]}],
            "generationConfig": {"maxOutputTokens": 1, "temperature": 0,
                                 "responseLogprobs": True, "logprobs": 5},
        })
        if "error" in d:
            msg = str(d["error"].get("message", ""))[:70]
            state = "disabled" if "not enabled" in msg.lower() else "error"
        else:
            c = d.get("candidates", [{}])[0]
            state = "yes" if c.get("logprobsResult") else "no-field"
        rows.append({"model": m, "logprobs": state})
        print(f"{m:28} {state}", file=sys.stderr, flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="google")
    args = ap.parse_args()
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GOOGLE_API_KEY not set (put it in .env, never in the repo)")
    rows = google(key)
    print(json.dumps({"provider": "google", "results": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
