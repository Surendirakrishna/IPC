// The API reports offsets as Unicode code-point indices. JavaScript strings index UTF-16 code
// units, so astral characters (e.g. emoji) occupy two units. Convert explicitly before slicing.

export function codePointToUtf16Index(text: string, cpIndex: number): number {
  let units = 0;
  let cps = 0;
  while (cps < cpIndex && units < text.length) {
    const code = text.charCodeAt(units);
    units += code >= 0xd800 && code <= 0xdbff && units + 1 < text.length ? 2 : 1;
    cps += 1;
  }
  return units;
}

export function sliceByCodePoints(text: string, start: number, end: number): string {
  return text.slice(codePointToUtf16Index(text, start), codePointToUtf16Index(text, end));
}

export interface Segment { text: string; highlighted: boolean }

/** Split a field into plain/highlighted segments for the given code-point spans (merged, sorted). */
export function highlightSegments(text: string, spans: { start: number; end: number }[]): Segment[] {
  const ranges = spans
    .map((s) => [codePointToUtf16Index(text, s.start), codePointToUtf16Index(text, s.end)] as const)
    .filter(([a, b]) => b > a)
    .sort((x, y) => x[0] - y[0]);
  const merged: [number, number][] = [];
  for (const [a, b] of ranges) {
    const last = merged[merged.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else merged.push([a, b]);
  }
  const out: Segment[] = [];
  let pos = 0;
  for (const [a, b] of merged) {
    if (a > pos) out.push({ text: text.slice(pos, a), highlighted: false });
    out.push({ text: text.slice(a, b), highlighted: true });
    pos = b;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), highlighted: false });
  return out;
}
