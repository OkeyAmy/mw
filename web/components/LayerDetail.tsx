"use client";

/**
 * Right panel — tabbed so it can never overflow: Trace | Neurons | Steer.
 *
 * Top-32 neurons come from GET /api/detail/{session}/{step}; the stream
 * carries top-3. Until the endpoint answers, the Neurons tab says which it is
 * showing and never pads the list.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { fmt, pct, showToken } from "@/lib/format";
import { DEFAULT_LAYERS, DEFAULT_STEER_LAYER } from "@/lib/types";
import type { DetailResponse, LayerTrace, SteerAck, Steering, TokenTrace } from "@/lib/types";
import { LayerTrace as LayerTraceTable } from "./LayerTrace";

interface Props {
  trace: TokenTrace | null;
  sessionId?: string;
  selectedLayer: number | null;
  steering: Steering;
  steerAck: SteerAck | null;
  layers: number;
  onSelectLayer: (layer: number | null) => void;
  onSteering: (next: Steering, flush?: boolean) => void;
  onNeuronSteer: (layer: number, neuron: number, value: number) => void;
  onNeuronClear: (layer: number | null) => void;
}

type Tab = "trace" | "neurons" | "steer";
type DetailState = "idle" | "loading" | "ok" | "error";

export function LayerDetail({
  trace,
  sessionId,
  selectedLayer,
  steering,
  steerAck,
  layers,
  onSelectLayer,
  onSteering,
  onNeuronSteer,
  onNeuronClear,
}: Props) {
  const layerCount = Math.max(1, layers || DEFAULT_LAYERS);
  const maxLayer = layerCount - 1;
  const [tab, setTab] = useState<Tab>("trace");
  const [detail, setDetail] = useState<DetailResponse | null>(null);
  const [detailState, setDetailState] = useState<DetailState>("idle");
  const [neuron, setNeuron] = useState<number | null>(null);
  const [value, setValue] = useState("0");
  const cache = useRef<Map<string, DetailResponse>>(new Map());

  useEffect(() => {
    if (!trace || !sessionId) {
      setDetail(null);
      setDetailState("idle");
      return;
    }
    const key = `${sessionId}:${trace.step}`;
    const cached = cache.current.get(key);
    if (cached) {
      setDetail(cached);
      setDetailState("ok");
      return;
    }
    let alive = true;
    setDetailState("loading");
    fetch(`/api/detail/${encodeURIComponent(sessionId)}/${trace.step}`, { cache: "no-store" })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()) as DetailResponse;
      })
      .then((json) => {
        if (!alive) return;
        if (json && Array.isArray(json.layers)) {
          cache.current.set(key, json);
          setDetail(json);
          setDetailState("ok");
        } else {
          setDetailState("error");
        }
      })
      .catch(() => {
        if (!alive) return;
        setDetail(null);
        setDetailState("error");
      });
    return () => {
      alive = false;
    };
  }, [sessionId, trace?.step]);

  useEffect(() => {
    setNeuron(null);
    setValue("0");
  }, [selectedLayer, trace?.step]);

  // Opening a layer is a drill-down: show its neurons immediately.
  useEffect(() => {
    if (selectedLayer !== null) setTab("neurons");
  }, [selectedLayer]);

  const streamLayer: LayerTrace | null =
    trace && selectedLayer !== null
      ? (trace.layers.find((l) => l.l === selectedLayer) ?? null)
      : null;
  const detailLayer: LayerTrace | null =
    detail && selectedLayer !== null
      ? (detail.layers.find((l) => l.l === selectedLayer) ?? null)
      : null;
  const layerData = detailLayer ?? streamLayer;

  const neurons = useMemo(() => {
    const list = layerData?.neurons ?? [];
    return [...list].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 32);
  }, [layerData]);
  const maxNeuron = neurons.reduce((m, [, v]) => Math.max(m, Math.abs(v)), 0) || 1;

  const steered = trace?.steered === true;
  const incoherent =
    steered && (trace?.coef === undefined || trace?.steer_layer === undefined);

  const layerCoef = selectedLayer !== null && steering.layer === selectedLayer ? steering.coef : 0;
  const MAX_COEF = 4;

  const commitCoef = (coef: number, flush = false) => {
    if (selectedLayer === null) return;
    onSteering(
      { layer: selectedLayer, coef: Math.max(-MAX_COEF, Math.min(MAX_COEF, coef)) },
      flush,
    );
  };

  const applyNeuron = (target: number) => {
    if (selectedLayer === null || neuron === null) return;
    const v = Math.max(-1000, Math.min(1000, target));
    setValue(String(v));
    onNeuronSteer(selectedLayer, neuron, v);
  };

  return (
    <aside className="detail" aria-label="Layer detail">
      <div className="detail-head">
        <div className="detail-title">
          <b>{selectedLayer !== null ? `L${selectedLayer}` : "Layer detail"}</b>
          <span className="detail-sub">
            {selectedLayer !== null
              ? `${layerData ? fmt(layerData.norm, 1) : "—"} norm · ${
                  detailState === "ok" ? `${neurons.length} neurons` : "top-3"
                }`
              : "click a column"}
          </span>
        </div>
        {selectedLayer !== null ? (
          <button type="button" className="btn ghost icon" onClick={() => onSelectLayer(null)}>
            close
          </button>
        ) : null}
      </div>

      <div className="detail-tabs" role="tablist" aria-label="Layer detail sections">
        {(["trace", "neurons", "steer"] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            className={`detail-tab${tab === t ? " active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="detail-body" role="tabpanel">
        {tab === "trace" ? (
          <>
            <div className="section">
              <div className="section-head">
                <span className="label">Token frame</span>
                <span className={`chip ${steered ? "causal" : ""}`}>
                  {steered ? "causal" : "correlation"}
                </span>
              </div>
              {trace ? (
                <div className="kv-list">
                  <div className="kv">
                    <span>step</span>
                    <span>
                      T{trace.step} · {JSON.stringify(trace.token)}
                    </span>
                  </div>
                  <div className="kv">
                    <span>steered</span>
                    <span>{trace.steered === true ? "true" : trace.steered === false ? "false" : "absent"}</span>
                  </div>
                  <div className="kv">
                    <span>coef</span>
                    <span>{trace.coef === undefined ? "absent" : fmt(trace.coef, 2)}</span>
                  </div>
                  <div className="kv">
                    <span>steer_layer</span>
                    <span>{trace.steer_layer === undefined ? "absent" : (trace.steer_layer ?? "null")}</span>
                  </div>
                  <div className="kv">
                    <span>layers</span>
                    <span>{trace.layers.length}</span>
                  </div>
                </div>
              ) : (
                <div className="empty-note">no token selected</div>
              )}
              {incoherent ? (
                <div className="detail-warning" role="alert">
                  frame says <code>steered: true</code> but{" "}
                  {trace?.coef === undefined ? <code>coef</code> : null}
                  {trace?.coef === undefined && trace?.steer_layer === undefined ? " and " : null}
                  {trace?.steer_layer === undefined ? <code>steer_layer</code> : null} is missing —
                  not claiming an effect.
                </div>
              ) : null}
            </div>

            <div className="section">
              <div className="section-head">
                <span className="label">
                  {selectedLayer !== null ? `Lens · L${selectedLayer}` : "Lens"}
                </span>
                <span className="hint">top-3</span>
              </div>
              {layerData && layerData.lens.length > 0 ? (
                <div className="lens-list">
                  {layerData.lens.slice(0, 3).map(([tok, p], i) => (
                    <div className="lens-row" key={`${tok}-${i}`}>
                      <span className="tok">{showToken(tok, 14)}</span>
                      <span className="pbar">
                        <span style={{ width: `${Math.max(1, p * 100)}%` }} />
                      </span>
                      <span className="pnum">{pct(p)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-note">
                  {selectedLayer !== null ? "no lens data for this layer" : "pick a layer"}
                </div>
              )}
            </div>

            <details className="fold">
              <summary>all layers</summary>
              <LayerTraceTable trace={trace} />
            </details>
          </>
        ) : null}

        {tab === "neurons" ? (
          selectedLayer === null ? (
            <div className="empty-note">click a column in the brain to open its neurons</div>
          ) : (
            <>
              <div className="section">
                <div className="section-head">
                  <span className="label">Top neurons</span>
                  <span className="hint">
                    {detailState === "loading"
                      ? "loading top-32…"
                      : detailState === "ok"
                        ? "top-32 · /api/detail"
                        : "top-3 · stream"}
                  </span>
                </div>
                {detailState === "error" ? (
                  <div className="detail-warning">
                    /api/detail did not answer — showing the top-3 the stream carried. The other 29
                    are not drawn.
                  </div>
                ) : null}
                {neurons.length === 0 ? (
                  <div className="empty-note">no neuron data for L{selectedLayer}</div>
                ) : (
                  <div className="neuron-list">
                    {neurons.map(([idx, act]) => (
                      <button
                        type="button"
                        key={idx}
                        className={`neuron-row${neuron === idx ? " selected" : ""}`}
                        onClick={() => {
                          setNeuron(idx);
                          setValue("0");
                        }}
                      >
                        <span className="nid">n{idx}</span>
                        <span className="nbar">
                          <span style={{ width: `${Math.max(1, (Math.abs(act) / maxNeuron) * 100)}%` }} />
                        </span>
                        <span className="nval">{fmt(act, 2)}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="section">
                <div className="section-head">
                  <span className="label">Steer neuron</span>
                  <span className="hint">{neuron === null ? "pick one" : `n${neuron}`}</span>
                </div>
                <div className="field-row">
                  <input
                    type="number"
                    step={0.5}
                    value={value}
                    disabled={neuron === null}
                    onChange={(e) => setValue(e.target.value)}
                    aria-label="Target activation value"
                  />
                  <button
                    type="button"
                    className="btn ghost"
                    disabled={neuron === null || !Number.isFinite(Number(value))}
                    onClick={() => applyNeuron(Number(value))}
                  >
                    apply
                  </button>
                  <button
                    type="button"
                    className="btn ghost"
                    disabled={neuron === null}
                    onClick={() => neuron !== null && applyNeuron(0)}
                  >
                    ablate
                  </button>
                  <button
                    type="button"
                    className="btn ghost"
                    onClick={() => {
                      onNeuronClear(selectedLayer);
                      setNeuron(null);
                    }}
                  >
                    clear
                  </button>
                </div>
                <p className="note">
                  0 = ablation, a large value amplifies. Written into the MLP intermediate before
                  the down projection.
                </p>
                <div className="ack">
                  {neuron !== null &&
                  steerAck?.neuron === neuron &&
                  steerAck.layer === selectedLayer
                    ? `ack: n${neuron} = ${steerAck.value !== undefined ? fmt(steerAck.value, 2) : "?"}`
                    : "no ack yet"}
                </div>
              </div>
            </>
          )
        ) : null}

        {tab === "steer" ? (
          <>
            {selectedLayer !== null ? (
              <div className="section">
                <div className="section-head">
                  <span className="label">Layer vector · L{selectedLayer}</span>
                  <span className={`hint${layerCoef === 0 ? "" : " causal"}`}>
                    {layerCoef === 0 ? "0 = off" : layerCoef.toFixed(1)}
                  </span>
                </div>
                <div className="slider-row">
                  <input
                    type="range"
                    min={-MAX_COEF}
                    max={MAX_COEF}
                    step={0.1}
                    value={layerCoef}
                    onChange={(e) => commitCoef(Number(e.target.value))}
                    onPointerUp={() => commitCoef(layerCoef, true)}
                    onKeyUp={() => commitCoef(layerCoef, true)}
                    aria-label={`Steering coefficient at layer ${selectedLayer}`}
                  />
                  <span className="num">{layerCoef === 0 ? "0.0" : layerCoef.toFixed(1)}</span>
                </div>
                <p className="note">
                  coef × the layer&rsquo;s mean residual norm. Causal only for this layer, direction
                  and coefficient — each frame reports whether it was steered.
                </p>
                <button
                  type="button"
                  className="btn ghost"
                  disabled={layerCoef === 0}
                  onClick={() => commitCoef(0, true)}
                >
                  zero
                </button>
              </div>
            ) : null}

            <div className="section">
              <div className="section-head">
                <span className="label">Global steering</span>
                <span className="hint">
                  {steering.coef === 0 ? "0 = off" : `L${steering.layer} · ${steering.coef.toFixed(1)}`}
                </span>
              </div>
              <div className="slider-row">
                <span className="num">{Math.min(steering.layer, maxLayer)}</span>
                <input
                  type="range"
                  min={0}
                  max={maxLayer}
                  step={1}
                  value={Math.min(steering.layer, maxLayer)}
                  onChange={(e) =>
                    onSteering({ layer: Number(e.target.value), coef: steering.coef })
                  }
                  onPointerUp={() => onSteering(steering, true)}
                  onKeyUp={() => onSteering(steering, true)}
                  aria-label="Steering layer"
                />
                <span className="hint">layer</span>
              </div>
              <div className="slider-row">
                <span className="num">
                  {steering.coef === 0 ? "0.0" : steering.coef.toFixed(1)}
                </span>
                <input
                  type="range"
                  min={-4}
                  max={4}
                  step={0.1}
                  value={Math.max(-4, Math.min(4, steering.coef))}
                  onChange={(e) => onSteering({ layer: steering.layer, coef: Number(e.target.value) })}
                  onPointerUp={() => onSteering(steering, true)}
                  onKeyUp={() => onSteering(steering, true)}
                  aria-label="Steering coefficient"
                />
                <span className="hint">coef</span>
              </div>
              <button
                type="button"
                className="btn ghost"
                onClick={() => onSteering({ layer: DEFAULT_STEER_LAYER, coef: 0 }, true)}
              >
                reset
              </button>
            </div>

            <div className="ack">
              {steerAck
                ? `last ack: L${steerAck.layer}${
                    steerAck.coef !== undefined ? ` · coef ${steerAck.coef.toFixed(1)}` : ""
                  }${steerAck.neuron !== undefined ? ` · n${steerAck.neuron}` : ""}`
                : "no ack yet"}
            </div>
          </>
        ) : null}
      </div>
    </aside>
  );
}
