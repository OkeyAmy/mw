"use client";

/**
 * The brain, in the reference's visual grammar but driven only by real data.
 *
 *   columns   layer 0 … L-1 from left to right (x = layer index, no invention)
 *   nodes     one per neuron the wire carries (top-3 in the stream, up to 12)
 *             radius  = residual norm at that layer (scaled within the token)
 *             y       = the neuron's activation value within that layer
 *   edges     layer l → l+1 only, plus faint l → l+2 skips; these encode
 *             adjacency, not measured weights, and the caption says so
 *   particles one per transition per generated token, duration from that
 *             token's measured ms (slower token → slower wave)
 *
 * `xy` (shared PCA) is an optional geometry mode, not the default. If a layer
 * carries no neurons it is drawn hollow and labelled.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from "react";
import { fmt } from "@/lib/format";
import { DEFAULT_LAYERS, type TokenTrace } from "@/lib/types";

interface Props {
  traces: TokenTrace[];
  selectedIndex: number;
  selectedLayer: number | null;
  layers: number;
  mode: "layers" | "pca";
  onPick: (layer: number, tokenIndex: number) => void;
  onExample?: (text: string) => void;
}

interface NodeView {
  layer: number;
  neuron: number;
  value: number;
  x: number;
  y: number;
  r: number;
  hot: number;
  hasData: boolean;
}

interface Particle {
  layer: number;
  born: number;
  dur: number;
}

const EXAMPLES = [
  "def add(a, b):",
  "The capital of France is",
  "Explain recursion in one sentence",
];

const ACCENT = "159,232,200";
const MUTED = "138,138,138";

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

export function BrainGraph({
  traces,
  selectedIndex,
  selectedLayer,
  layers,
  mode,
  onPick,
  onExample,
}: Props) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<Particle[]>([]);
  const lastLenRef = useRef(0);
  const hoverRef = useRef<NodeView | null>(null);
  const dirtyRef = useRef(true);

  const [size, setSize] = useState({ w: 640, h: 420 });
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);

  const layerCount = Math.max(1, layers || DEFAULT_LAYERS);
  const activeIndex = selectedIndex >= 0 ? selectedIndex : traces.length - 1;
  const trace = activeIndex >= 0 ? traces[activeIndex] : null;

  const stats = useMemo(() => {
    if (!trace) return { nodes: 0, active: 0, maxAbs: 0, maxNorm: 0 };
    let maxAbs = 0;
    let maxNorm = 0;
    for (const layer of trace.layers) {
      maxNorm = Math.max(maxNorm, layer.norm);
      for (const [, v] of layer.neurons) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
    const threshold = Math.max(0, maxAbs * 0.25);
    let count = 0;
    for (const layer of trace.layers) {
      for (const [, v] of layer.neurons) if (Math.abs(v) >= threshold) count += 1;
    }
    return { nodes: trace.layers.length, active: count, maxAbs, maxNorm };
  }, [trace]);

  const tokenSpeed = useMemo(() => {
    const last = traces.length > 0 ? traces[traces.length - 1] : null;
    const prev = traces.length > 1 ? traces[traces.length - 2] : null;
    if (!last || !prev) return 0;
    const dt = last.ms - prev.ms;
    return dt > 0 ? 1000 / dt : 0;
  }, [traces]);

  // ---------------------------------------------------------------- layout ---
  const layout = useMemo(() => {
    const padL = 30;
    const padR = 26;
    const padT = 30;
    const padB = 96;
    const usableW = Math.max(40, size.w - padL - padR);
    const usableH = Math.max(80, size.h - padT - padB);
    const colX = (l: number) =>
      layerCount <= 1 ? padL + usableW / 2 : padL + (l / (layerCount - 1)) * usableW;
    return { padL, padR, padT, padB, usableW, usableH, colX, midY: padT + usableH / 2 };
  }, [size, layerCount]);

  const buildNodes = useCallback((): NodeView[] => {
    if (!trace) return [];
    const out: NodeView[] = [];
    const maxNorm = stats.maxNorm || 1;
    const maxAbs = stats.maxAbs || 1;
    for (let l = 0; l < layerCount; l += 1) {
      const layer = trace.layers.find((x) => x.l === l);
      const x = layout.colX(l);
      if (!layer || layer.neurons.length === 0) {
        out.push({
          layer: l,
          neuron: -1,
          value: 0,
          x,
          y: layout.midY,
          r: 2.5,
          hot: 0,
          hasData: false,
        });
        continue;
      }
      const neurons = layer.neurons.slice(0, 12);
      const band = layout.usableH * 0.55;
      // y is the measured activation on a log scale shared by the whole token,
      // so columns differ from each other instead of being rank-normalised
      // into the same three rows.
      const logMax = Math.log1p(Math.max(1e-9, maxAbs));
      neurons.forEach(([index, value]) => {
        const logNorm = clamp(Math.log1p(Math.abs(value)) / logMax, 0, 1);
        const y = layout.midY + (0.5 - logNorm) * band + ((index % 5) - 2) * 0.7;
        const jitter = ((index % 7) - 3) * 1.1;
        out.push({
          layer: l,
          neuron: index,
          value,
          x: x + jitter,
          y,
          r: 1.8 + 5.4 * clamp(layer.norm / maxNorm, 0, 1),
          hot: clamp(Math.abs(value) / maxAbs, 0, 1),
          hasData: true,
        });
      });
    }
    return out;
  }, [trace, layerCount, layout, stats.maxAbs, stats.maxNorm]);

  // ------------------------------------------------------------------ draw ---
  const draw = useCallback(
    (now: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = size.w;
      const h = size.h;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, w, h);

      const muted = `rgba(${MUTED},`;
      const accent = `rgba(${ACCENT},`;
      const { colX, midY } = layout;

      // layer axes + labels
      ctx.font = "10px 'JetBrains Mono', ui-monospace, monospace";
      ctx.textAlign = "center";
      for (let l = 0; l < layerCount; l += 1) {
        const x = colX(l);
        ctx.strokeStyle = `${muted}0.13)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, layout.padT - 8);
        ctx.lineTo(Math.round(x) + 0.5, size.h - layout.padB + 6);
        ctx.stroke();
        if (l % 5 === 0 || l === layerCount - 1) {
          ctx.fillStyle = `${muted}0.55)`;
          ctx.fillText(`L${l}`, x, size.h - layout.padB + 18);
        }
      }

      if (!trace) {
        for (let l = 0; l < layerCount; l += 1) {
          ctx.beginPath();
          ctx.arc(colX(l), midY, 2.5, 0, Math.PI * 2);
          ctx.strokeStyle = `${muted}0.35)`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.fillStyle = `${muted}0.5)`;
        ctx.font = "10px 'JetBrains Mono', ui-monospace, monospace";
        ctx.fillText("no data", colX(Math.floor(layerCount / 2)), midY + 22);
        return;
      }

      if (mode === "pca") {
        const pts = trace.layers
          .filter((l) => Boolean(l.xy))
          .map((l) => ({ l: l.l, xy: l.xy as [number, number], norm: l.norm }));
        if (pts.length === 0) {
          ctx.fillStyle = `${muted}0.65)`;
          ctx.textAlign = "center";
          ctx.fillText("no xy on this frame — geometry unavailable", w / 2, midY);
          return;
        }
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const p of pts) {
          minX = Math.min(minX, p.xy[0]);
          maxX = Math.max(maxX, p.xy[0]);
          minY = Math.min(minY, p.xy[1]);
          maxY = Math.max(maxY, p.xy[1]);
        }
        const spanX = Math.max(1e-9, maxX - minX);
        const spanY = Math.max(1e-9, maxY - minY);
        const sx = layout.usableW / spanX;
        const sy = layout.usableH / spanY;
        const s = Math.min(sx, sy) * 0.82;
        const px = (v: number) => w / 2 + (v - (minX + maxX) / 2) * s;
        const py = (v: number) => midY - (v - (minY + maxY) / 2) * s;

        ctx.beginPath();
        pts.forEach((p, i) => {
          const x = px(p.xy[0]);
          const y = py(p.xy[1]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = `${accent}0.45)`;
        ctx.lineWidth = 1;
        ctx.stroke();

        let hoverPoint: { x: number; y: number; text: string } | null = null;
        void hoverPoint;
        return;
      }

      // ---- columns mode ----------------------------------------------------
      const nodes = buildNodes();
      const byLayer = new Map<number, NodeView[]>();
      for (const n of nodes) {
        const list = byLayer.get(n.layer) ?? [];
        list.push(n);
        byLayer.set(n.layer, list);
      }

      // adjacency edges (structure, not weights)
      for (let l = 0; l < layerCount - 1; l += 1) {
        const from = byLayer.get(l) ?? [];
        const to = byLayer.get(l + 1) ?? [];
        for (const a of from) {
          for (const b of to) {
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `${accent}0.09)`;
            ctx.lineWidth = 0.6;
            ctx.stroke();
          }
        }
        const skipTo = byLayer.get(l + 2);
        if (from.length > 0 && skipTo && skipTo.length > 0) {
          ctx.beginPath();
          ctx.moveTo(from[0].x, from[0].y);
          ctx.lineTo(skipTo[0].x, skipTo[0].y);
          ctx.strokeStyle = `${accent}0.03)`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      // selected column band
      if (selectedLayer !== null) {
        const x = colX(selectedLayer);
        ctx.fillStyle = `${accent}0.05)`;
        ctx.fillRect(x - 9, layout.padT - 10, 18, size.h - layout.padB - layout.padT + 18);
        ctx.strokeStyle = `${accent}0.35)`;
        ctx.lineWidth = 1;
        ctx.strokeRect(
          Math.round(x) - 8.5,
          layout.padT - 9.5,
          17,
          size.h - layout.padB - layout.padT + 17,
        );
      }

      // particles
      const nowMs = now;
      particlesRef.current = particlesRef.current.filter((p) => {
        const start = p.born;
        if (nowMs < start) return true;
        const t = (nowMs - start) / 1000 / p.dur;
        if (t >= 1) return false;
        const from = (byLayer.get(p.layer) ?? [])[0];
        const to = (byLayer.get(p.layer + 1) ?? [])[0];
        if (!from || !to) return false;
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t;
        const fade = t < 0.8 ? 1 : (1 - t) / 0.2;
        // trail: a short hairline, no glow
        const tx = from.x + (to.x - from.x) * Math.max(0, t - 0.08);
        const ty = from.y + (to.y - from.y) * Math.max(0, t - 0.08);
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(x, y);
        ctx.strokeStyle = `rgba(159,232,200,${fade * 0.45})`;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 1.6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232,255,246,${fade * 0.9})`;
        ctx.fill();
        return true;
      });

      // nodes
      for (const n of nodes) {
        if (!n.hasData) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
          ctx.strokeStyle = `${muted}0.5)`;
          ctx.lineWidth = 1;
          ctx.stroke();
          continue;
        }
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(159,232,200,${0.18 + n.hot * 0.72})`;
        ctx.fill();
        ctx.strokeStyle = `rgba(159,232,200,${0.3 + n.hot * 0.6})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();

        if (
          hoverRef.current &&
          hoverRef.current.layer === n.layer &&
          hoverRef.current.neuron === n.neuron
        ) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.r + 4, 0, Math.PI * 2);
          ctx.strokeStyle = `${accent}0.9)`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

    },
    [buildNodes, layout, layerCount, mode, selectedLayer, size, stats.maxNorm, trace],
  );

  // redraw on data/layout changes; loop only while particles are alive
  useEffect(() => {
    dirtyRef.current = true;
    draw(performance.now());
  }, [draw]);

  useEffect(() => {
    let raf = 0;
    let last = 0;
    const loop = (ts: number) => {
      raf = requestAnimationFrame(loop);
      const alive = particlesRef.current.length > 0;
      if (alive || dirtyRef.current) {
        if (ts - last > 16 || dirtyRef.current) {
          draw(ts);
          last = ts;
          dirtyRef.current = false;
        }
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [draw]);

  // new token → one particle per layer transition, timed by the token's ms
  useEffect(() => {
    if (traces.length > lastLenRef.current) {
      const token = traces[traces.length - 1];
      const dur = clamp((token.ms / 1000) * 0.85, 0.22, 2.6);
      const now = performance.now();
      for (let l = 0; l < layerCount - 1; l += 1) {
        particlesRef.current.push({ layer: l, born: now + l * 7, dur });
      }
      if (particlesRef.current.length > 240) {
        particlesRef.current.splice(0, particlesRef.current.length - 240);
      }
      dirtyRef.current = true;
    }
    lastLenRef.current = traces.length;
  }, [traces, layerCount]);

  // resize
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => {
      setSize({ w: el.clientWidth || 640, h: el.clientHeight || 420 });
      dirtyRef.current = true;
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // --------------------------------------------------------------- pointer ---
  const hitTest = useCallback(
    (clientX: number, clientY: number): NodeView | null => {
      const canvas = canvasRef.current;
      if (!canvas || !trace || mode === "pca") return null;
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      const nodes = buildNodes();
      let best: NodeView | null = null;
      let bestD = 14;
      for (const n of nodes) {
        if (!n.hasData) continue;
        const d = Math.hypot(n.x - x, n.y - y);
        if (d < Math.max(bestD, n.r + 5)) {
          best = n;
          bestD = d;
        }
      }
      return best;
    },
    [buildNodes, mode, trace],
  );

  const onMove = (e: ReactPointerEvent) => {
    const node = hitTest(e.clientX, e.clientY);
    hoverRef.current = node;
    const rect = wrapRef.current?.getBoundingClientRect();
    if (node && rect) {
      const layer = trace?.layers.find((l) => l.l === node.layer);
      setHover({
        x: e.clientX - rect.left + 14,
        y: e.clientY - rect.top + 12,
        text: `L${node.layer} · norm ${fmt(layer?.norm ?? 0, 1)} · n${node.neuron} · ${fmt(
          node.value,
          3,
        )}${node.r ? "" : ""}`,
      });
      if (wrapRef.current) wrapRef.current.style.cursor = "pointer";
    } else {
      setHover(null);
      if (wrapRef.current) wrapRef.current.style.cursor = "crosshair";
    }
    dirtyRef.current = true;
  };

  const onLeave = () => {
    hoverRef.current = null;
    setHover(null);
  };

  const onClick = (e: ReactMouseEvent) => {
    if (!trace) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (mode === "pca") {
      // nearest trajectory point by layer order
      const pts = trace.layers.filter((l) => Boolean(l.xy));
      if (pts.length === 0) return;
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const p of pts) {
        const xy = p.xy as [number, number];
        minX = Math.min(minX, xy[0]);
        maxX = Math.max(maxX, xy[0]);
        minY = Math.min(minY, xy[1]);
        maxY = Math.max(maxY, xy[1]);
      }
      const s =
        Math.min(
          layout.usableW / Math.max(1e-9, maxX - minX),
          layout.usableH / Math.max(1e-9, maxY - minY),
        ) * 0.82;
      let best = pts[0];
      let bestD = Infinity;
      for (const p of pts) {
        const xy = p.xy as [number, number];
        const px = size.w / 2 + (xy[0] - (minX + maxX) / 2) * s;
        const py = layout.midY - (xy[1] - (minY + maxY) / 2) * s;
        const d = Math.hypot(px - x, py - y);
        if (d < bestD) {
          bestD = d;
          best = p;
        }
      }
      onPick(best.l, activeIndex);
      return;
    }

    // nearest column by x
    let layer = 0;
    let bestD = Infinity;
    for (let l = 0; l < layerCount; l += 1) {
      const d = Math.abs(layout.colX(l) - x);
      if (d < bestD) {
        bestD = d;
        layer = l;
      }
    }
    onPick(layer, activeIndex);
  };

  const steered = trace?.steered === true;
  const noTokens = traces.length === 0;

  return (
    <div
      className="graph-wrap"
      ref={wrapRef}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
      onClick={onClick}
    >
      <canvas ref={canvasRef} className="graph-canvas" aria-label="Layer columns of measured activations" />

      <div className={`graph-caption${steered ? " causal" : ""}`}>
        {steered ? "STEERING ACTIVE · causal" : "measured activations · correlation, not meaning"}
      </div>

      <div className="graph-hud">
        <span className="hud-item">
          <span className="hud-label">tokens</span>
          <span className="hud-value">{traces.length}</span>
        </span>
        <span className="hud-item">
          <span className="hud-label">active</span>
          <span className="hud-value" title="neurons at ≥25% of this token's peak activation">
            {stats.active}
          </span>
        </span>
        <span className="hud-item">
          <span className="hud-label">layer</span>
          <span className="hud-value">
            {trace ? `L${selectedLayer ?? layerCount - 1}` : "—"}
          </span>
        </span>
        <span className="hud-item">
          <span className="hud-label">tok/s</span>
          <span className="hud-value">{tokenSpeed > 0 ? tokenSpeed.toFixed(1) : "—"}</span>
        </span>
      </div>

      <div className="graph-mode">
        {mode === "pca"
          ? "geometry: x/y = shared PCA of residual · z order = layer"
          : "x = layer · y = neuron activation · radius = norm · edges = adjacency"}
      </div>

      {noTokens ? (
        <div className="brain-empty">
          <b>Send a prompt to open the brain</b>
          <p>
            each generated token lights one column per layer ({layerCount} layers). click a column
            to open it. try one of these:
          </p>
          <div className="brain-examples">
            {EXAMPLES.map((example) => (
              <button
                type="button"
                key={example}
                className="brain-example"
                onClick={() => onExample?.(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {hover ? (
        <div
          className="graph-tooltip"
          style={{ left: hover.x, top: hover.y }}
          role="status"
          aria-live="off"
        >
          {hover.text}
        </div>
      ) : null}
    </div>
  );
}
