"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fmt, pct, showToken } from "@/lib/format";
import type { TokenTrace } from "@/lib/types";

interface Props {
  trace: TokenTrace | null;
}

async function copyText(text: string): Promise<boolean> {
  try {
    if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  } catch {
    return false;
  }
}

export function LayerTrace({ trace }: Props) {
  const [status, setStatus] = useState<"idle" | "ok" | "fail">("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const onCopy = useCallback(async () => {
    if (!trace) return;
    const ok = await copyText(JSON.stringify(trace, null, 2));
    setStatus(ok ? "ok" : "fail");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setStatus("idle"), 1600);
  }, [trace]);

  const rows = trace ? [...trace.layers].sort((a, b) => b.l - a.l) : [];
  const maxNorm = rows.reduce((m, r) => Math.max(m, r.norm), 0) || 1;

  return (
    <div className="section">
      <div className="section-head">
        <span className="label">Layer trace</span>
        <div className="row" style={{ gap: 6 }}>
          {trace ? (
            <span className="hint">
              T{trace.step} · {rows.length}L
            </span>
          ) : null}
          <button
            type="button"
            className="btn ghost"
            onClick={onCopy}
            disabled={!trace}
            title="Copy this token's full trace as JSON"
          >
            {status === "ok" ? "copied" : status === "fail" ? "copy failed" : "copy"}
          </button>
        </div>
      </div>

      {trace ? (
        <>
          <div className="trace-table">
            {rows.map((layer) => (
              <div className="trace-row" key={layer.l}>
                <span className="ln">L{layer.l}</span>
                <span className="bar-wrap">
                  <span
                    className="bar"
                    style={{ width: `${Math.max(2, (100 * layer.norm) / maxNorm)}%`, display: "block" }}
                  />
                </span>
                <span className="lens">
                  {layer.lens.length === 0
                    ? "not sampled"
                    : layer.lens.map(([tok, p], i) => (
                        <span key={`${layer.l}-${i}`}>
                          {i > 0 ? " · " : ""}
                          {showToken(tok, 10)} <span className="p">{pct(p)}</span>
                        </span>
                      ))}
                </span>
                {layer.neurons.length > 0 ? (
                  <span className="neur">
                    {layer.neurons
                      .map(([idx, val]) => `n${idx}:${fmt(val, 1)}`)
                      .join("  ")}
                  </span>
                ) : null}
              </div>
            ))}
          </div>
          <div className="hint">
            lens is a readout of the residual stream, not a decision the model made.
          </div>
        </>
      ) : (
        <div className="trace-empty">no token selected</div>
      )}
    </div>
  );
}
