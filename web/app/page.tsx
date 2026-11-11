"use client";

import { useCallback, useEffect, useState } from "react";
import { BrainGraph } from "@/components/BrainGraph";
import { Composer } from "@/components/Composer";
import { LayerDetail } from "@/components/LayerDetail";
import { MessageList } from "@/components/MessageList";
import { Onboarding } from "@/components/Onboarding";
import { shortPath } from "@/lib/format";
import { useGeneration } from "@/lib/useGeneration";
import { DEFAULT_LAYERS, type ChatMessage, type ModelInfo, type Steering } from "@/lib/types";

const ONBOARD_KEY = "mod_wei.onboarded.v1";

function buildExport(messages: ChatMessage[], info: ModelInfo | null, steering: Steering) {
  return {
    format: "mw.session.v1",
    exported_at: new Date().toISOString(),
    model: info,
    steering,
    messages: messages.map((m) =>
      m.role === "user"
        ? { role: "user", text: m.text }
        : {
            role: "assistant",
            session: m.sessionId ?? null,
            text: m.text,
            steering: m.steering,
            injections: m.injections,
            done_reason: m.doneReason ?? null,
            error: m.error ?? null,
            n_tokens: m.traces.length,
            tokens: m.traces,
          },
    ),
  };
}

export default function Page() {
  const gen = useGeneration();
  const {
    phase,
    info,
    messages,
    activeId,
    activeMessage,
    traces,
    selectedIndex,
    selectedTrace,
    steering,
    steerAck,
    notice,
    lastError,
    prefilling,
    canSend,
    send,
    cancel,
    inject,
    neuronSteer,
    neuronClear,
    reconnect,
    selectToken,
    focusMessage,
    newChat,
    setSteering,
    dismissNotice,
  } = gen;

  const [railOpen, setRailOpen] = useState(true);
  const [mode, setMode] = useState<"layers" | "pca">("layers");
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [onboarding, setOnboarding] = useState(false);

  const modelReady = info?.ready === true;
  const layers = info?.layers && info.layers > 0 ? info.layers : DEFAULT_LAYERS;

  const currentFrame = selectedTrace ?? (traces.length > 0 ? traces[traces.length - 1] : null);
  const steeredNow = currentFrame?.steered === true;

  useEffect(() => {
    try {
      if (!localStorage.getItem(ONBOARD_KEY)) setOnboarding(true);
    } catch {
      /* private mode */
    }
  }, []);

  const closeOnboarding = useCallback(() => {
    try {
      localStorage.setItem(ONBOARD_KEY, "1");
    } catch {
      /* ignore */
    }
    setOnboarding(false);
  }, []);

  const useExample = useCallback(
    (text: string) => {
      setDraft(text);
      setRailOpen(true);
      // A chip is copy that says "try one of these" — clicking runs it.
      if (canSend) {
        if (send(text, steering.layer, steering.coef, 48)) setDraft("");
      }
      requestAnimationFrame(() => {
        document.querySelector<HTMLTextAreaElement>(".composer textarea")?.focus();
      });
    },
    [canSend, send, steering.coef, steering.layer],
  );

  const startNewChat = useCallback(() => {
    newChat();
    setSelectedLayer(null);
    setDraft("");
  }, [newChat]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (onboarding) return;
      const target = e.target as HTMLElement | null;
      if (target && target.tagName === "INPUT") return;
      e.preventDefault();
      cancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cancel, onboarding]);

  const onSend = useCallback(
    (text: string, maxNew: number) => send(text, steering.layer, steering.coef, maxNew),
    [send, steering.coef, steering.layer],
  );

  const onPick = useCallback(
    (layer: number, tokenIndex: number) => {
      selectToken(tokenIndex);
      setSelectedLayer(layer);
    },
    [selectToken],
  );

  const exportSession = useCallback(() => {
    const payload = buildExport(messages, info, steering);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    a.href = url;
    a.download = `mw_session_${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, [info, messages, steering]);

  const phaseLabel =
    prefilling !== null
      ? `prefilling ${prefilling} tokens…`
      : !modelReady
        ? "model loading"
        : phase;
  const pending = phase === "streaming" || prefilling !== null;
  const dotTone =
    prefilling !== null
      ? "warn"
      : phase === "error"
        ? "err"
        : phase === "streaming"
          ? "live"
          : modelReady
            ? "ok"
            : "warn";

  return (
    <main className={`shell${railOpen ? "" : " rail-collapsed"}`}>
      {/* ------------------------------------------------------- chat rail --- */}
      <aside className="rail" aria-label="Chat">
        <div className="rail-head">
          {railOpen ? (
            <>
              <span className="brand">mw</span>
              <span className={`dot ${dotTone}`} />
              <button
                type="button"
                className="btn ghost rail-new"
                onClick={startNewChat}
                disabled={messages.length === 0 || phase === "streaming"}
                title="New chat (clears the transcript and traces)"
              >
                new chat
              </button>
              <button
                type="button"
                className="btn ghost icon rail-toggle"
                onClick={() => setRailOpen(false)}
                title="Collapse chat"
                aria-label="Collapse chat"
              >
                ‹
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn ghost icon rail-toggle"
              onClick={() => setRailOpen(true)}
              title="Expand chat"
              aria-label="Expand chat"
            >
              ›
            </button>
          )}
        </div>

        {railOpen ? (
          <>
            <div className="rail-model">
              {modelReady ? (
                <>
                  <div className="rail-model-arch">{info?.arch ?? "unknown architecture"}</div>
                  <div className="rail-model-grid">
                    <span>{info?.layers ?? "?"}L</span>
                    <span>d{info?.d_model ?? "?"}</span>
                    <span>{info?.dtype ?? "?"}</span>
                    <span title={info?.model}>{shortPath(info?.model)}</span>
                    <span>{info?.steer_vectors ?? 0} vectors</span>
                    <span>{traces.length} tok</span>
                  </div>
                </>
              ) : (
                <div className="rail-loading">
                  <b>model loading…</b>
                  <span>GET /api/info · not ready · prompts blocked</span>
                </div>
              )}
            </div>

            {lastError ? (
              <div className="rail-banner error" role="alert">
                <span>{lastError}</span>
                <button type="button" className="btn ghost" onClick={reconnect}>
                  retry
                </button>
              </div>
            ) : notice ? (
              <div className="rail-banner" role="status">
                <span>{notice}</span>
                <button type="button" className="btn ghost" onClick={dismissNotice}>
                  ok
                </button>
              </div>
            ) : !modelReady ? (
              <div className="rail-banner" role="status">
                <span>model loading — prompts are blocked until weights are resident</span>
              </div>
            ) : null}

            <MessageList
              messages={messages}
              activeId={activeId}
              phase={phase}
              layers={layers}
              onFocus={focusMessage}
            />

            <Composer
              canSend={canSend}
              streaming={phase === "streaming"}
              modelReady={modelReady}
              steering={steering}
              value={draft}
              onChange={setDraft}
              onSend={onSend}
              onCancel={cancel}
              onInject={inject}
            />
          </>
        ) : (
          <div className="rail-mark" aria-hidden="true">
            chat
          </div>
        )}
      </aside>

      {/* ------------------------------------------------------------ brain --- */}
      <section className="stage" aria-label="Brain">
        <div className="stage-head">
          <div className="stage-title">
            <b>brain</b>
            <span className="stage-sub">
              {mode === "layers"
                ? `${layers} layer columns · real traces`
                : "shared PCA geometry · real xy"}
            </span>
          </div>
          <div className="stage-actions">
            <span className={`chip ${steeredNow ? "causal" : ""}`}>
              {steeredNow ? "causal" : "correlational"}
            </span>
            <span className={`chip ${phase === "error" ? "err" : ""}`}>
              <span className={`dot ${dotTone}`} />
              {phaseLabel}
            </span>
            <button
              type="button"
              className={`btn ghost${mode === "pca" ? " on" : ""}`}
              onClick={() => setMode(mode === "layers" ? "pca" : "layers")}
              title="Toggle the shared-PCA geometry view"
            >
              {mode === "layers" ? "geometry" : "columns"}
            </button>
            <button
              type="button"
              className="btn ghost icon"
              onClick={() => setOnboarding(true)}
              title="Quick tour"
              aria-label="Quick tour"
            >
              ?
            </button>
            <button
              type="button"
              className="btn ghost"
              onClick={exportSession}
              disabled={messages.length === 0}
              title="Download transcript and every token trace as JSON"
            >
              export
            </button>
          </div>
        </div>

        {lastError ? (
          <div className="banner error" role="alert">
            <strong>error</strong>
            <span>{lastError}</span>
            <span className="spacer" />
            <button type="button" className="btn ghost" onClick={reconnect}>
              retry
            </button>
          </div>
        ) : notice ? (
          <div className="banner" role="status">
            <span>{notice}</span>
            <span className="spacer" />
            <button type="button" className="btn ghost" onClick={dismissNotice}>
              dismiss
            </button>
          </div>
        ) : null}

        <div className="stage-body">
          <BrainGraph
            traces={traces}
            selectedIndex={selectedIndex}
            selectedLayer={selectedLayer}
            layers={layers}
            mode={mode}
            pending={pending}
            prefilling={prefilling}
            onPick={onPick}
            onExample={useExample}
          />
        </div>
      </section>

      {/* ----------------------------------------------------- layer detail --- */}
      <LayerDetail
        trace={selectedTrace}
        sessionId={activeMessage?.sessionId}
        selectedLayer={selectedLayer}
        steering={steering}
        steerAck={steerAck}
        layers={layers}
        onSelectLayer={setSelectedLayer}
        onSteering={setSteering}
        onNeuronSteer={neuronSteer}
        onNeuronClear={neuronClear}
      />

      <Onboarding open={onboarding} onClose={closeOnboarding} />
    </main>
  );
}
