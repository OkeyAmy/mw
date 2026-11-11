/**
 * Wire types for app/PROTOCOL.md. The backend is the source of truth; fields
 * that are not guaranteed by the protocol are optional so an older server
 * cannot crash the UI.
 */

export type Pair = [string, number];

export interface LayerTrace {
  /** layer index, 0-based */
  l: number;
  /** residual stream L2 norm at this layer, last position */
  norm: number;
  /** top-k logit-lens predictions: [token, probability] */
  lens: Pair[];
  /** top-k MLP intermediate neurons: [index, activation] */
  neurons: [number, number][];
  /**
   * 2-D coordinates from a basis shared across all tokens: PCA of the
   * mean-centred, standardised residual trajectory. Absent on older servers.
   */
  xy?: [number, number] | null;
}

export interface PcaQuality {
  /** fraction of variance carried by PC1+PC2, 0..1 */
  var_explained?: number;
}

export interface TokenTrace {
  type: "token";
  session?: string;
  step: number;
  token: string;
  token_id?: number;
  /** cumulative milliseconds since the generation started */
  ms: number;
  topk?: Pair[];
  layers: LayerTrace[];
  /** true when this frame was produced with an intervention applied */
  steered?: boolean;
  /** coefficient that produced this frame (absent if the server omits it) */
  coef?: number;
  /** layer the intervention was applied at (null/absent if none) */
  steer_layer?: number | null;
  /** quality of the shared PCA basis used for this frame's xy */
  pca?: PcaQuality;
}

export interface Injection {
  atStep: number;
  tokens: number;
  text: string;
}

export type ServerMessage =
  | TokenTrace
  | { type: "prefill"; tokens: number }
  | { type: "done"; session?: string; reason?: string; n_tokens?: number }
  | { type: "error"; message: string }
  | {
      type: "steer_ack";
      layer: number;
      coef?: number;
      neuron?: number;
      value?: number;
    }
  | { type: "injected"; at_step: number; tokens: number; text: string }
  | { type: "restarted"; prompt: string };

export interface ModelInfo {
  ready: boolean;
  model?: string;
  arch?: string;
  layers?: number;
  d_model?: number;
  vocab?: number;
  scaling?: number;
  dtype?: string;
  steer_vectors?: number;
}

export interface Health {
  ok?: boolean;
  busy?: boolean;
  uptime_s?: number;
  generations?: number;
}

/** GET /api/detail/{session}/{step} — top-32 neurons per layer. */
export interface DetailResponse {
  session: string;
  step: number;
  token: string;
  layers: LayerTrace[];
}

/** connecting → ready → streaming → done | error */
export type Phase = "connecting" | "ready" | "streaming" | "done" | "error";

export interface Steering {
  layer: number;
  coef: number;
}

export interface SteerAck {
  layer: number;
  coef?: number;
  neuron?: number;
  value?: number;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  text: string;
  traces: TokenTrace[];
  steering: Steering;
  injections: Injection[];
  sessionId?: string;
  doneReason?: string;
  error?: string;
  finished: boolean;
}

export interface UserMessage {
  id: string;
  role: "user";
  text: string;
}

export type ChatMessage = UserMessage | AssistantMessage;

export const DEFAULT_LAYERS = 28;
export const DEFAULT_STEER_LAYER = 8;

export function isAssistant(m: ChatMessage): m is AssistantMessage {
  return m.role === "assistant";
}
