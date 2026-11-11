"use client";

/**
 * WebSocket + REST state machine for the microscope (app/PROTOCOL.md).
 *
 *   connecting ──open──► ready ──chat──► streaming ──done──► done
 *        ▲                  ▲                 │
 *        └── backoff ────────┴── close/error ─┴──► error
 *
 * Every socket handler reads refs, never the state captured at mount: the
 * component re-renders once per token, but the socket is created once.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_STEER_LAYER,
  type AssistantMessage,
  type ChatMessage,
  type Health,
  type Injection,
  type LayerTrace,
  type ModelInfo,
  type Phase,
  type Pair,
  type ServerMessage,
  type SteerAck,
  type Steering,
  type TokenTrace,
} from "./types";

const RECONNECT_CAP_MS = 8000;
const STEER_DEBOUNCE_MS = 140;

function uid(): string {
  return Math.random().toString(36).slice(2, 10);
}

function backendWsUrl(): string {
  // Next rejects a ws:// rewrite destination and does not reliably proxy
  // upgrades, so the browser talks to the FastAPI port directly.
  const explicit = process.env.NEXT_PUBLIC_WS_URL;
  if (explicit) return explicit;
  if (typeof window === "undefined") return "ws://127.0.0.1:8000/ws";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.hostname || "127.0.0.1";
  const port = process.env.NEXT_PUBLIC_BACKEND_PORT || "8000";
  return `${proto}//${host}:${port}/ws`;
}

function num(v: unknown, fallback = 0): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function toPairs(v: unknown): Pair[] {
  if (!Array.isArray(v)) return [];
  const out: Pair[] = [];
  for (const item of v) {
    if (Array.isArray(item) && item.length >= 2) out.push([String(item[0] ?? ""), num(item[1])]);
  }
  return out.slice(0, 5);
}

function toNeurons(v: unknown): [number, number][] {
  if (!Array.isArray(v)) return [];
  const out: [number, number][] = [];
  for (const item of v) {
    if (Array.isArray(item) && item.length >= 2) out.push([num(item[0]), num(item[1])]);
  }
  return out.slice(0, 5);
}

export function parseLayers(v: unknown): LayerTrace[] {
  if (!Array.isArray(v)) return [];
  return v.map((entry, i) => {
    const layer = (entry ?? {}) as Record<string, unknown>;
    const xyRaw = layer.xy;
    const xy =
      Array.isArray(xyRaw) &&
      xyRaw.length >= 2 &&
      Number.isFinite(Number(xyRaw[0])) &&
      Number.isFinite(Number(xyRaw[1]))
        ? ([Number(xyRaw[0]), Number(xyRaw[1])] as [number, number])
        : null;
    return {
      l: num(layer.l, i),
      norm: num(layer.norm),
      lens: toPairs(layer.lens),
      neurons: toNeurons(layer.neurons),
      xy,
    };
  });
}

function parseToken(raw: Record<string, unknown>): TokenTrace | null {
  if (typeof raw.step !== "number") return null;
  const pcaRaw = raw.pca;
  const pca =
    pcaRaw && typeof pcaRaw === "object"
      ? {
          var_explained:
            typeof (pcaRaw as Record<string, unknown>).var_explained === "number"
              ? ((pcaRaw as Record<string, unknown>).var_explained as number)
              : undefined,
        }
      : undefined;
  return {
    type: "token",
    session: typeof raw.session === "string" ? raw.session : undefined,
    step: raw.step,
    token: typeof raw.token === "string" ? raw.token : "",
    token_id: typeof raw.token_id === "number" ? raw.token_id : undefined,
    ms: num(raw.ms),
    topk: toPairs(raw.topk),
    layers: parseLayers(raw.layers),
    steered: raw.steered === true,
    coef: typeof raw.coef === "number" ? raw.coef : undefined,
    steer_layer:
      typeof raw.steer_layer === "number"
        ? raw.steer_layer
        : raw.steer_layer === null
          ? null
          : undefined,
    pca,
  };
}

function patchAssistant(
  prev: ChatMessage[],
  id: string,
  patch: (m: AssistantMessage) => AssistantMessage,
): ChatMessage[] {
  const idx = prev.findIndex((m) => m.id === id);
  if (idx === -1) return prev;
  const next = prev.slice();
  next[idx] = patch(prev[idx] as AssistantMessage);
  return next;
}

/** The wire conversation: every prior turn, nothing synthetic. */
function toWireHistory(messages: ChatMessage[]): { role: "user" | "assistant"; content: string }[] {
  const out: { role: "user" | "assistant"; content: string }[] = [];
  for (const m of messages) {
    if (m.role === "user") {
      out.push({ role: "user", content: m.text });
    } else if (m.text.trim().length > 0) {
      out.push({ role: "assistant", content: m.text });
    }
  }
  return out;
}

export interface GenerationApi {
  phase: Phase;
  info: ModelInfo | null;
  health: Health | null;
  socketOpen: boolean;
  messages: ChatMessage[];
  activeId: string | null;
  activeMessage: AssistantMessage | null;
  traces: TokenTrace[];
  selectedIndex: number;
  selectedTrace: TokenTrace | null;
  steering: Steering;
  steerAck: SteerAck | null;
  notice: string | null;
  lastError: string | null;
  prefilling: number | null;
  canSend: boolean;
  send: (prompt: string, layer: number, coef: number, maxNew?: number) => boolean;
  cancel: () => void;
  inject: (text: string) => boolean;
  neuronSteer: (layer: number, neuron: number, value: number) => void;
  neuronClear: (layer: number | null) => void;
  reconnect: () => void;
  selectToken: (index: number) => void;
  focusMessage: (id: string) => void;
  newChat: () => void;
  setSteering: (next: Steering, flush?: boolean) => void;
  dismissNotice: () => void;
}

export function useGeneration(): GenerationApi {
  const [phase, setPhase] = useState<Phase>("connecting");
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [socketOpen, setSocketOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number>(-1);
  const [steering, setSteeringState] = useState<Steering>({
    layer: DEFAULT_STEER_LAYER,
    coef: 0,
  });
  const [steerAck, setSteerAck] = useState<SteerAck | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [prefilling, setPrefilling] = useState<number | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const phaseRef = useRef<Phase>("connecting");
  const streamingRef = useRef(false);
  const infoRef = useRef<ModelInfo | null>(null);
  const steerRef = useRef<Steering>({ layer: DEFAULT_STEER_LAYER, coef: 0 });
  const currentIdRef = useRef<string | null>(null);
  const pendingActivateRef = useRef<string | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const steerTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  const closedRef = useRef(false);
  const connectRef = useRef<() => void>(() => {});
  const handleRef = useRef<(msg: ServerMessage) => void>(() => {});

  const applyPhase = useCallback((next: Phase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const sendSteer = useCallback((next: Steering) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "steer", layer: next.layer, coef: next.coef }));
    }
  }, []);

  const setSteering = useCallback(
    (next: Steering, flush = false) => {
      steerRef.current = next;
      setSteeringState(next);
      if (steerTimer.current) clearTimeout(steerTimer.current);
      if (flush) {
        sendSteer(next);
        return;
      }
      steerTimer.current = setTimeout(() => sendSteer(steerRef.current), STEER_DEBOUNCE_MS);
    },
    [sendSteer],
  );

  // --------------------------------------------------------------- inbound ---
  const handleMessage = useCallback(
    (msg: ServerMessage) => {
      if (msg.type === "prefill") {
        setPrefilling(Math.max(0, num(msg.tokens)));
        return;
      }

      if (msg.type === "token") {
        const token = parseToken(msg as unknown as Record<string, unknown>);
        if (!token) return;
        setPrefilling(null);
        streamingRef.current = true;
        if (phaseRef.current !== "streaming") applyPhase("streaming");

        const id = currentIdRef.current;
        if (!id) {
          // Only reachable after a reconnect mid-stream. Keep the token.
          const fresh = uid();
          currentIdRef.current = fresh;
          setActiveId(fresh);
          setSelectedIndex(-1);
          setMessages((prev) => [
            ...prev,
            {
              id: fresh,
              role: "assistant",
              text: token.token,
              traces: [token],
              steering: { ...steerRef.current },
              injections: [],
              sessionId: token.session,
              finished: false,
            },
          ]);
          return;
        }
        // The panel keeps the previous turn visible until the new turn has at
        // least one token — sending must not blank the brain.
        if (pendingActivateRef.current === id) {
          pendingActivateRef.current = null;
          setActiveId(id);
          setSelectedIndex(-1);
        }
        setMessages((prev) =>
          patchAssistant(prev, id, (m) => ({
            ...m,
            text: m.text + token.token,
            traces: [...m.traces, token],
            sessionId: m.sessionId ?? token.session,
          })),
        );
        return;
      }

      if (msg.type === "done") {
        streamingRef.current = false;
        setPrefilling(null);
        const id = currentIdRef.current;
        if (id) {
          setMessages((prev) =>
            patchAssistant(prev, id, (m) => ({
              ...m,
              finished: true,
              doneReason:
                typeof msg.reason === "string"
                  ? msg.reason
                  : typeof msg.n_tokens === "number"
                    ? "max_new"
                    : "done",
            })),
          );
        }
        if (phaseRef.current !== "error") applyPhase("done");
        return;
      }

      if (msg.type === "error") {
        streamingRef.current = false;
        setPrefilling(null);
        const text = typeof msg.message === "string" ? msg.message : "backend error";
        setLastError(text);
        setNotice(text);
        const id = currentIdRef.current;
        if (id) {
          setMessages((prev) =>
            patchAssistant(prev, id, (m) => ({ ...m, finished: true, error: text })),
          );
        }
        applyPhase("error");
        return;
      }

      if (msg.type === "steer_ack") {
        setSteerAck({
          layer: num(msg.layer),
          coef: typeof msg.coef === "number" ? msg.coef : undefined,
          neuron: typeof msg.neuron === "number" ? msg.neuron : undefined,
          value: typeof msg.value === "number" ? msg.value : undefined,
        });
        return;
      }

      if (msg.type === "injected") {
        const injection: Injection = {
          atStep: num(msg.at_step),
          tokens: num(msg.tokens),
          text: typeof msg.text === "string" ? msg.text : "",
        };
        const id = currentIdRef.current;
        if (id) {
          setMessages((prev) =>
            patchAssistant(prev, id, (m) => ({
              ...m,
              injections: [...m.injections, injection],
            })),
          );
        }
        setNotice(
          `injected ${injection.tokens} token(s) at step ${injection.atStep} — later traces condition on this text`,
        );
        return;
      }

      if (msg.type === "restarted") {
        // The server discarded the cache and restarted from an edited prompt.
        // Close the current turn and open a fresh one so step numbers do not mix.
        const previous = currentIdRef.current;
        const fresh = uid();
        const prompt = typeof msg.prompt === "string" ? msg.prompt : "(edited prompt)";
        currentIdRef.current = fresh;
        setActiveId(fresh);
        setSelectedIndex(-1);
        setMessages((prev) => {
          const closed = previous
            ? patchAssistant(prev, previous, (m) => ({ ...m, finished: true }))
            : prev;
          return [
            ...closed,
            { id: uid(), role: "user", text: prompt },
            {
              id: fresh,
              role: "assistant",
              text: "",
              traces: [],
              steering: { ...steerRef.current },
              injections: [],
              finished: false,
            },
          ];
        });
        setNotice("server restarted generation from an edited prompt");
      }
    },
    [applyPhase],
  );

  useEffect(() => {
    handleRef.current = handleMessage;
  }, [handleMessage]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // -------------------------------------------------------------- outbound ---
  const connect = useCallback(() => {
    if (closedRef.current) return;
    if (retryTimer.current) {
      clearTimeout(retryTimer.current);
      retryTimer.current = null;
    }
    const previous = socketRef.current;
    if (previous) {
      previous.onopen = null;
      previous.onclose = null;
      previous.onerror = null;
      previous.onmessage = null;
      try {
        previous.close();
      } catch {
        /* already gone */
      }
      socketRef.current = null;
    }

    if (phaseRef.current !== "streaming") applyPhase("connecting");

    let ws: WebSocket;
    try {
      ws = new WebSocket(backendWsUrl());
    } catch {
      setLastError("could not build a WebSocket URL");
      applyPhase("error");
      const delay = Math.min(500 * 2 ** attemptRef.current++, RECONNECT_CAP_MS);
      retryTimer.current = setTimeout(() => connectRef.current(), delay);
      return;
    }
    socketRef.current = ws;

    ws.onopen = () => {
      if (closedRef.current) {
        ws.close();
        return;
      }
      attemptRef.current = 0;
      setSocketOpen(true);
      setLastError(null);
      if (!streamingRef.current) applyPhase("ready");
      // Deterministic steering after a reconnect: the backend stores it
      // globally, so push what the UI currently shows.
      sendSteer(steerRef.current);
    };

    ws.onmessage = (ev) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data as string);
      } catch {
        setLastError("unparseable message from the backend");
        return;
      }
      if (parsed && typeof parsed === "object" && "type" in parsed) {
        handleRef.current(parsed as ServerMessage);
      }
    };

    ws.onerror = () => {
      setLastError(`no answer from ${backendWsUrl()}`);
    };

    ws.onclose = () => {
      setSocketOpen(false);
      if (socketRef.current === ws) socketRef.current = null;
      if (closedRef.current) return;

      streamingRef.current = false;
      const id = currentIdRef.current;
      if (id) {
        setMessages((prev) =>
          patchAssistant(prev, id, (m) =>
            m.finished ? m : { ...m, finished: true, error: "connection lost mid-generation" },
          ),
        );
      }
      applyPhase("error");
      setLastError("connection to the backend lost — reconnecting");

      const delay = Math.min(500 * 2 ** attemptRef.current++, RECONNECT_CAP_MS);
      retryTimer.current = setTimeout(() => {
        retryTimer.current = null;
        connectRef.current();
      }, delay);
    };
  }, [applyPhase, sendSteer]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const reconnect = useCallback(() => {
    attemptRef.current = 0;
    setLastError(null);
    connectRef.current();
  }, []);

  const emit = useCallback((payload: Record<string, unknown>): boolean => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setNotice("not connected — waiting for the backend");
      return false;
    }
    ws.send(JSON.stringify(payload));
    return true;
  }, []);

  const send = useCallback(
    (prompt: string, layer: number, coef: number, maxNew = 48): boolean => {
      const text = prompt.trim();
      if (!text) return false;
      if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
        setNotice("not connected — waiting for the backend");
        return false;
      }
      if (streamingRef.current || phaseRef.current === "streaming") {
        setNotice("a generation is already running — press Esc to cancel first");
        return false;
      }
      if (!infoRef.current?.ready) {
        setNotice("the model is still loading — the prompt was not sent");
        return false;
      }

      steerRef.current = { layer, coef };
      setSteeringState({ layer, coef });

      // The server prefills from this array. Sending only `prompt` made every
      // turn a cold single-shot generation (the "it reloads" feeling).
      const history = toWireHistory(messagesRef.current);

      const assistantId = uid();
      currentIdRef.current = assistantId;
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "user", text },
        {
          id: assistantId,
          role: "assistant",
          text: "",
          traces: [],
          steering: { layer, coef },
          injections: [],
          finished: false,
        },
      ]);
      pendingActivateRef.current = assistantId;
      setNotice(null);
      setLastError(null);
      streamingRef.current = true;
      setPrefilling(null);
      applyPhase("streaming");

      socketRef.current.send(
        JSON.stringify({
          type: "chat",
          messages: [...history, { role: "user", content: text }],
          max_new: Math.max(1, Math.min(256, Math.round(maxNew))),
          layer,
          coef,
        }),
      );
      return true;
    },
    [applyPhase],
  );

  const cancel = useCallback(() => {
    if (!streamingRef.current) return;
    if (emit({ type: "cancel" })) {
      setNotice("cancel requested — waiting for the server to stop");
    }
  }, [emit]);

  const newChat = useCallback(() => {
    if (streamingRef.current) return;
    setMessages([]);
    messagesRef.current = [];
    setActiveId(null);
    setSelectedIndex(-1);
    currentIdRef.current = null;
    pendingActivateRef.current = null;
    setPrefilling(null);
    setNotice(null);
    setLastError(null);
  }, []);

  const inject = useCallback(
    (text: string): boolean => {
      const clean = text.trim();
      if (!clean) return false;
      if (!streamingRef.current) {
        setNotice("injection is only meaningful mid-generation — send a prompt first");
        return false;
      }
      if (!emit({ type: "inject", text: clean })) return false;
      setNotice("injection queued — it lands after the current token");
      return true;
    },
    [emit],
  );

  const neuronSteer = useCallback(
    (layer: number, neuron: number, value: number) => {
      emit({ type: "neuron_steer", layer, neuron, value });
    },
    [emit],
  );

  const neuronClear = useCallback(
    (layer: number | null) => {
      emit({ type: "neuron_clear", layer });
    },
    [emit],
  );

  // ------------------------------------------------------------- lifecycle ---
  useEffect(() => {
    closedRef.current = false;
    connectRef.current();
    return () => {
      closedRef.current = true;
      if (retryTimer.current) clearTimeout(retryTimer.current);
      if (steerTimer.current) clearTimeout(steerTimer.current);
      const ws = socketRef.current;
      socketRef.current = null;
      if (ws) {
        ws.onopen = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.onmessage = null;
        try {
          ws.close();
        } catch {
          /* noop */
        }
      }
    };
  }, []);

  useEffect(() => {
    let alive = true;

    const pull = async () => {
      try {
        const [infoRes, healthRes] = await Promise.all([
          fetch("/api/info", { cache: "no-store" }),
          fetch("/api/health", { cache: "no-store" }),
        ]);
        const infoJson = (await infoRes.json().catch(() => null)) as ModelInfo | null;
        if (!alive) return;
        if (infoJson && typeof infoJson.ready === "boolean") {
          setInfo(infoJson);
          infoRef.current = infoJson;
        } else {
          const loading: ModelInfo = { ready: false };
          setInfo(loading);
          infoRef.current = loading;
        }
        const healthJson = (await healthRes.json().catch(() => null)) as Health | null;
        if (alive && healthJson) setHealth(healthJson);
      } catch {
        if (!alive) return;
        const loading: ModelInfo = { ready: false };
        setInfo(loading);
        infoRef.current = loading;
      }
    };

    void pull();
    const timer = setInterval(pull, 4000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  // ---------------------------------------------------------------- derived ---
  const activeMessage = useMemo<AssistantMessage | null>(() => {
    if (activeId) {
      const found = messages.find((m) => m.id === activeId);
      if (found && found.role === "assistant") return found;
    }
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const m = messages[i];
      if (m.role === "assistant") return m;
    }
    return null;
  }, [activeId, messages]);

  const traces = activeMessage?.traces ?? [];
  const clampedIndex =
    traces.length === 0
      ? -1
      : selectedIndex < 0 || selectedIndex >= traces.length
        ? traces.length - 1
        : selectedIndex;
  const selectedTrace = clampedIndex >= 0 ? traces[clampedIndex] : null;

  const canSend =
    socketOpen &&
    info?.ready === true &&
    phase !== "streaming" &&
    phase !== "connecting" &&
    prefilling === null;

  return {
    phase,
    info,
    health,
    socketOpen,
    messages,
    activeId,
    activeMessage,
    traces,
    selectedIndex: clampedIndex,
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
    selectToken: (index: number) => {
      const bounded = Math.max(0, Math.min(traces.length - 1, index));
      setSelectedIndex(Number.isFinite(bounded) ? bounded : -1);
    },
    focusMessage: (id: string) => {
      setActiveId(id);
      setSelectedIndex(-1);
    },
    newChat,
    setSteering,
    dismissNotice: () => setNotice(null),
  };
}
