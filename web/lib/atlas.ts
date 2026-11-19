import type { ObjectSummary } from "@/lib/types";

export interface AtlasBounds {
  minRa: number;
  maxRa: number;
  minDec: number;
  maxDec: number;
  centerRa: number;
  centerDec: number;
}

export interface AtlasPoint { x: number; y: number }

export function computeAtlasBounds(objects: ObjectSummary[]): AtlasBounds {
  const coordinates = objects
    .filter((object) => Number.isFinite(object.ra_deg) && Number.isFinite(object.dec_deg))
    .map((object) => ({ ra: Number(object.ra_deg), dec: Number(object.dec_deg) }));
  if (!coordinates.length) return { minRa: 0, maxRa: 360, minDec: -90, maxDec: 90, centerRa: 180, centerDec: 0 };
  const raValues = coordinates.map((coordinate) => coordinate.ra);
  const decValues = coordinates.map((coordinate) => coordinate.dec);
  const rawMinRa = Math.min(...raValues); const rawMaxRa = Math.max(...raValues);
  const rawMinDec = Math.min(...decValues); const rawMaxDec = Math.max(...decValues);
  const raPadding = Math.max((rawMaxRa - rawMinRa) * .055, .25);
  const decPadding = Math.max((rawMaxDec - rawMinDec) * .075, .25);
  const minRa = rawMinRa - raPadding; const maxRa = rawMaxRa + raPadding;
  const minDec = rawMinDec - decPadding; const maxDec = rawMaxDec + decPadding;
  return { minRa, maxRa, minDec, maxDec, centerRa: (minRa + maxRa) / 2, centerDec: (minDec + maxDec) / 2 };
}

export function projectCatalogCoordinate(
  ra: number,
  dec: number,
  bounds: AtlasBounds,
  width: number,
  height: number,
  paddingX = 54,
  paddingY = 48,
): AtlasPoint {
  const safeWidth = Math.max(width - paddingX * 2, 1);
  const safeHeight = Math.max(height - paddingY * 2, 1);
  const raSpan = Math.max(bounds.maxRa - bounds.minRa, .001);
  const decSpan = Math.max(bounds.maxDec - bounds.minDec, .001);
  return {
    x: paddingX + ((bounds.maxRa - ra) / raSpan) * safeWidth,
    y: paddingY + ((bounds.maxDec - dec) / decSpan) * safeHeight,
  };
}

export function convexHull(points: AtlasPoint[]): AtlasPoint[] {
  if (points.length <= 3) return points;
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (origin: AtlasPoint, a: AtlasPoint, b: AtlasPoint) => (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
  const lower: AtlasPoint[] = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2)!, lower.at(-1)!, point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: AtlasPoint[] = [];
  for (const point of sorted.reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2)!, upper.at(-1)!, point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop(); upper.pop();
  return [...lower, ...upper];
}
