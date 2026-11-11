/** Small display helpers shared by the panels. */

export function fmt(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs !== 0 && abs < 0.01) return n.toExponential(1);
  return n.toFixed(digits);
}

export function pct(p: number): string {
  if (!Number.isFinite(p)) return "—";
  if (p >= 0.995) return "1.00";
  if (p < 0.001) return "<.001";
  return p.toFixed(3);
}

/** Tokens arrive with real whitespace/newlines from the tokenizer. */
export function showToken(token: string, max = 14): string {
  const visible = token.replace(/\n/g, "\\n").replace(/\t/g, "\\t");
  const trimmed = visible.length > max ? `${visible.slice(0, max)}…` : visible;
  return trimmed.length === 0 ? "∅" : trimmed;
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

export function shortPath(path: string | undefined): string {
  if (!path) return "—";
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : path;
}
