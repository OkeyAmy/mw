"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Steering } from "@/lib/types";

interface Props {
  canSend: boolean;
  streaming: boolean;
  modelReady: boolean;
  steering: Steering;
  value: string;
  onChange: (text: string) => void;
  onSend: (text: string, maxNew: number) => boolean;
  onCancel: () => void;
  onInject: (text: string) => boolean;
}

export function Composer({
  canSend,
  streaming,
  modelReady,
  steering,
  value,
  onChange,
  onSend,
  onCancel,
  onInject,
}: Props) {
  const [maxNew, setMaxNew] = useState(48);
  const [injectText, setInjectText] = useState("");
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const injectRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 130)}px`;
  }, [value]);

  // Autofocus when the composer becomes usable, without stealing focus from a
  // field the user is already typing in.
  useEffect(() => {
    if (!canSend) return;
    const active = document.activeElement;
    if (active && active !== document.body) return;
    areaRef.current?.focus();
  }, [canSend]);

  const submit = useCallback(() => {
    if (!canSend || streaming) return;
    if (onSend(value, maxNew)) onChange("");
  }, [canSend, maxNew, onChange, onSend, streaming, value]);

  const submitInject = useCallback(() => {
    if (!injectText.trim()) return;
    if (onInject(injectText)) {
      setInjectText("");
      injectRef.current?.focus();
    }
  }, [injectText, onInject]);

  const placeholder = streaming
    ? "generating — Esc cancels"
    : !modelReady
      ? "model loading — the composer unlocks when /api/info reports ready"
      : !canSend
        ? "waiting for the backend socket…"
        : "Send a prompt…  (Enter to send)";

  return (
    <div className="composer">
      <div className="composer-box">
        <textarea
          ref={areaRef}
          value={value}
          rows={1}
          disabled={streaming || !canSend}
          placeholder={placeholder}
          aria-label="Prompt"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        {streaming ? (
          <button type="button" className="btn danger" onClick={onCancel}>
            Cancel
          </button>
        ) : (
          <button
            type="button"
            className="btn"
            disabled={!canSend || value.trim().length === 0}
            onClick={submit}
          >
            Send
          </button>
        )}
      </div>

      {streaming ? (
        <div className="inject-row">
          <input
            ref={injectRef}
            value={injectText}
            placeholder="inject context mid-generation…"
            aria-label="Text to inject mid-generation"
            onChange={(e) => setInjectText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submitInject();
              }
            }}
          />
          <button
            type="button"
            className="btn ghost"
            disabled={injectText.trim().length === 0}
            onClick={submitInject}
          >
            inject
          </button>
        </div>
      ) : null}

      <div className="composer-foot">
        <span>
          <kbd>Enter</kbd> send · <kbd>Esc</kbd> cancel
        </span>
        <span className="grow" />
        <span>
          steer L{steering.layer} · coef {steering.coef.toFixed(1)}
          {steering.coef === 0 ? " (off)" : ""}
        </span>
        <label className="row" style={{ gap: 5 }}>
          max
          <input
            type="number"
            min={1}
            max={256}
            value={maxNew}
            disabled={streaming}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) setMaxNew(Math.max(1, Math.min(256, Math.round(n))));
            }}
            aria-label="Maximum new tokens"
          />
        </label>
      </div>
    </div>
  );
}
