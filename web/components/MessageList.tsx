"use client";

import { Fragment, useEffect, useMemo, useRef } from "react";
import { fmt } from "@/lib/format";
import type { AssistantMessage, ChatMessage, Injection, Phase } from "@/lib/types";

interface Props {
  messages: ChatMessage[];
  activeId: string | null;
  phase: Phase;
  layers: number;
  onFocus: (id: string) => void;
}

function steeringLabel(m: AssistantMessage): string {
  return m.steering.coef === 0
    ? "no steering"
    : `steer L${m.steering.layer} · coef ${m.steering.coef.toFixed(1)}`;
}

export function MessageList({ messages, activeId, phase, layers, onFocus }: Props) {
  const listRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const el = listRef.current;
    if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  return (
    <div
      className="messages"
      ref={listRef}
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      onScroll={() => {
        const el = listRef.current;
        if (!el) return;
        pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
      }}
    >
      {messages.length === 0 ? (
        <div className="empty-chat">
          <p className="empty-line">Every token streams with its internal trace.</p>
          <details className="how">
            <summary>How this works ▸</summary>
            <p>
              Each generated token arrives with its full trace: residual norm per layer, the
              logit-lens readout at every layer, and its top MLP neurons.
            </p>
            <p>
              The brain draws one ribbon per token through all {layers} layer planes. Click a node
              to open it.
            </p>
            <p>
              <code>Enter</code> send · <code>Shift+Enter</code> newline · <code>Esc</code> cancel.
              While streaming you can inject text mid-generation.
            </p>
          </details>
        </div>
      ) : null}

      {messages.map((m) => {
        if (m.role === "user") {
          return (
            <div className="msg user" key={m.id}>
              <div className="who">you</div>
              <div className="msg-body">
                <div className="bubble">{m.text}</div>
              </div>
            </div>
          );
        }

        const active = m.id === activeId;
        const live = !m.finished && phase === "streaming";
        const lastToken = m.traces.length > 0 ? m.traces[m.traces.length - 1] : null;
        const prevToken = m.traces.length > 1 ? m.traces[m.traces.length - 2] : null;
        const dt = lastToken && prevToken ? lastToken.ms - prevToken.ms : 0;

        return (
          <div className={`msg assistant${active ? " active" : ""}`} key={m.id}>
            <div className="who">mw</div>
            <div className="msg-body">
              <div className="bubble">
                {m.traces.length === 0 && live ? (
                  <span className="bubble pending">generating</span>
                ) : (
                  <AssistantText message={m} />
                )}
                {live ? <span className="cursor" aria-hidden="true" /> : null}
                {m.error ? <span style={{ color: "var(--err)" }}> {m.error}</span> : null}
              </div>
              <div className="msg-meta">
                <button type="button" className="clickable" onClick={() => onFocus(m.id)}>
                  {active ? "inspecting" : "inspect"}
                </button>
                <span className="sep">·</span>
                <span>{m.traces.length} tokens</span>
                <span className="sep">·</span>
                <span>{steeringLabel(m)}</span>
                {dt > 0 ? (
                  <>
                    <span className="sep">·</span>
                    <span>{fmt(dt, 0)} ms/tok</span>
                  </>
                ) : null}
                {m.injections.length > 0 ? (
                  <>
                    <span className="sep">·</span>
                    <span>{m.injections.length} injected</span>
                  </>
                ) : null}
                {m.finished && m.doneReason ? (
                  <>
                    <span className="sep">·</span>
                    <span>{m.doneReason}</span>
                  </>
                ) : null}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Renders the assistant text token by token so an injected system note can sit
 * at the exact step the server says it landed — not appended at the end, which
 * would misrepresent when the model conditioned on it.
 */
function AssistantText({ message }: { message: AssistantMessage }) {
  const { byStep, leftover } = useMemo(() => {
    const map = new Map<number, Injection[]>();
    for (const inj of message.injections) {
      const list = map.get(inj.atStep) ?? [];
      list.push(inj);
      map.set(inj.atStep, list);
    }
    const steps = new Set(message.traces.map((t) => t.step));
    const orphans = message.injections.filter((inj) => !steps.has(inj.atStep));
    return { byStep: map, leftover: orphans };
  }, [message.injections, message.traces]);

  if (message.traces.length === 0) {
    return (
      <>
        {message.text}
        {leftover.map((inj, j) => (
          <InjectionChip key={`orphan-${j}`} injection={inj} />
        ))}
      </>
    );
  }

  return (
    <>
      {message.traces.map((trace) => (
        <Fragment key={`${message.id}-${trace.step}`}>
          <span>{trace.token}</span>
          {(byStep.get(trace.step) ?? []).map((inj, j) => (
            <InjectionChip key={`${message.id}-inj-${trace.step}-${j}`} injection={inj} />
          ))}
        </Fragment>
      ))}
      {leftover.map((inj, j) => (
        <InjectionChip key={`${message.id}-tail-${j}`} injection={inj} />
      ))}
    </>
  );
}

function InjectionChip({ injection }: { injection: Injection }) {
  return (
    <span
      className="inject-chip"
      title={`server: injected at step ${injection.atStep} (${injection.tokens} tokens)`}
    >
      <span className="inject-tag">injected @ step {injection.atStep}</span>
      {injection.text}
    </span>
  );
}
