"use client";

import { Minus, Plus, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ObjectSummary } from "@/lib/types";

const COLORS: Record<string, string> = { CONFIRMED: "#7BD7BA", "FALSE POSITIVE": "#ED9A9F", CANDIDATE: "#EFC77C" };

export function StarAtlas({ objects, selectedId, onSelect }: { objects: ObjectSummary[]; selectedId?: string; onSelect: (object: ObjectSummary) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const [hovered, setHovered] = useState<ObjectSummary | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number } | null>(null);
  const positioned = useMemo(() => objects.filter((object) => object.ra_deg != null && object.dec_deg != null), [objects]);

  const coordinates = useCallback((object: ObjectSummary, width: number, height: number) => {
    const baseX = ((360 - (object.ra_deg ?? 0)) / 360) * width;
    const baseY = ((90 - (object.dec_deg ?? 0)) / 180) * height;
    return { x: (baseX - width / 2) * view.zoom + width / 2 + view.x, y: (baseY - height / 2) * view.zoom + height / 2 + view.y };
  }, [view]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const draw = () => {
      const rect = container.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
      canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
      const ctx = canvas.getContext("2d"); if (!ctx) return;
      ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
      const gradient = ctx.createRadialGradient(rect.width * .55, rect.height * .45, 0, rect.width * .5, rect.height * .5, rect.width * .7);
      gradient.addColorStop(0, "rgba(35,91,122,.15)"); gradient.addColorStop(.5, "rgba(11,26,43,.08)"); gradient.addColorStop(1, "rgba(5,10,18,.72)");
      ctx.fillStyle = gradient; ctx.fillRect(0, 0, rect.width, rect.height);
      ctx.strokeStyle = "rgba(112,219,247,.10)"; ctx.lineWidth = 1;
      for (let ra = 0; ra <= 360; ra += 30) { const x = ((ra / 360 * rect.width) - rect.width / 2) * view.zoom + rect.width / 2 + view.x; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, rect.height); ctx.stroke(); }
      for (let dec = -60; dec <= 60; dec += 30) { const y = (((90 - dec) / 180 * rect.height) - rect.height / 2) * view.zoom + rect.height / 2 + view.y; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(rect.width, y); ctx.stroke(); }
      positioned.forEach((object) => {
        const point = coordinates(object, rect.width, rect.height);
        if (point.x < -20 || point.x > rect.width + 20 || point.y < -20 || point.y > rect.height + 20) return;
        const active = object.object_id === selectedId;
        const radius = active ? 5.5 : object.catalog_disposition === "CANDIDATE" ? 3.2 : 2.3;
        if (active) { ctx.beginPath(); ctx.arc(point.x, point.y, 12, 0, Math.PI * 2); ctx.strokeStyle = "rgba(112,219,247,.5)"; ctx.stroke(); }
        ctx.beginPath(); ctx.arc(point.x, point.y, radius, 0, Math.PI * 2); ctx.fillStyle = COLORS[object.catalog_disposition] ?? "#B5C4D0"; ctx.shadowBlur = active ? 18 : 8; ctx.shadowColor = ctx.fillStyle; ctx.fill(); ctx.shadowBlur = 0;
      });
      ctx.fillStyle = "rgba(181,196,208,.72)"; ctx.font = "12px Consolas, monospace";
      ctx.fillText("RA 24h", 18, 24); ctx.fillText("RA 0h", rect.width - 55, 24); ctx.fillText("+90°", 18, 46); ctx.fillText("−90°", 18, rect.height - 18);
    };
    draw(); const observer = new ResizeObserver(draw); observer.observe(container); return () => observer.disconnect();
  }, [positioned, selectedId, coordinates, view]);

  const nearest = (clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect(); if (!rect) return null;
    let match: ObjectSummary | null = null; let distance = 14;
    positioned.forEach((object) => { const point = coordinates(object, rect.width, rect.height); const next = Math.hypot(point.x - (clientX - rect.left), point.y - (clientY - rect.top)); if (next < distance) { distance = next; match = object; } });
    return match;
  };

  return <div className="star-atlas" ref={containerRef}>
    <canvas ref={canvasRef} aria-label={`Equatorial sky atlas with ${positioned.length} catalog objects`} onMouseMove={(event) => { if (drag) { setView((value) => ({ ...value, x: value.x + event.clientX - drag.x, y: value.y + event.clientY - drag.y })); setDrag({ x: event.clientX, y: event.clientY }); } else setHovered(nearest(event.clientX, event.clientY)); }} onMouseDown={(event) => setDrag({ x: event.clientX, y: event.clientY })} onMouseUp={(event) => { if (drag && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 4) { const object = nearest(event.clientX, event.clientY); if (object) onSelect(object); } setDrag(null); }} onMouseLeave={() => { setDrag(null); setHovered(null); }} onWheel={(event) => { event.preventDefault(); setView((value) => ({ ...value, zoom: Math.min(4, Math.max(.7, value.zoom * (event.deltaY > 0 ? .9 : 1.1))) })); }} />
    <div className="atlas-tools" aria-label="Atlas controls"><button aria-label="Zoom in" onClick={() => setView((value) => ({ ...value, zoom: Math.min(4, value.zoom * 1.2) }))}><Plus /></button><button aria-label="Zoom out" onClick={() => setView((value) => ({ ...value, zoom: Math.max(.7, value.zoom / 1.2) }))}><Minus /></button><button aria-label="Reset atlas" onClick={() => setView({ zoom: 1, x: 0, y: 0 })}><RotateCcw /></button></div>
    <div className="atlas-legend"><span><i style={{ background: COLORS.CONFIRMED }} />Confirmed</span><span><i style={{ background: COLORS.CANDIDATE }} />Candidate</span><span><i style={{ background: COLORS["FALSE POSITIVE"] }} />False positive</span></div>
    {hovered && <div className="atlas-tooltip"><strong>{hovered.display_name}</strong><span>KIC {hovered.kic}</span><span>RA {hovered.ra_deg?.toFixed(4)}° · Dec {hovered.dec_deg?.toFixed(4)}°</span><small>{hovered.catalog_disposition}</small></div>}
  </div>;
}
