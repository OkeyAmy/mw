"use client";

import { useEffect, useState } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
}

const STEPS = [
  {
    title: "Send a prompt",
    body: "Type in the composer and press Enter. Every generated token streams in with its internal trace.",
  },
  {
    title: "Open a layer",
    body: "Hover a node to read its numbers; click it to open the top-32 neurons, lens top-3 and steering controls for that layer.",
  },
  {
    title: "Change something",
    body: "Steer a layer with the coefficient slider, or inject new text mid-generation. The token frame reports whether it was steered.",
  },
];

export function Onboarding({ open, onClose }: Props) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (open) setStep(0);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  const current = STEPS[step];

  return (
    <div className="onboard-backdrop" role="dialog" aria-modal="true" aria-label="Quick tour">
      <div className="onboard">
        <div className="onboard-top">
          <span className="label">
            step {step + 1} / {STEPS.length}
          </span>
          <button type="button" className="btn ghost icon" onClick={onClose}>
            skip
          </button>
        </div>

        <h2>{current.title}</h2>
        <p>{current.body}</p>

        <div className="onboard-dots" aria-hidden="true">
          {STEPS.map((s, i) => (
            <span key={s.title} className={`onboard-dot${i === step ? " on" : ""}`} />
          ))}
        </div>

        <div className="onboard-actions">
          <button
            type="button"
            className="btn ghost icon"
            disabled={step === 0}
            onClick={() => setStep((s) => Math.max(0, s - 1))}
          >
            back
          </button>
          {step < STEPS.length - 1 ? (
            <button type="button" className="btn" onClick={() => setStep((s) => s + 1)}>
              next
            </button>
          ) : (
            <button type="button" className="btn" onClick={onClose}>
              start
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
