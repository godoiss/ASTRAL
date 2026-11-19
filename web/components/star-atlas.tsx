"use client";

import { Focus, Layers3, Minus, Plus, RotateCcw, Waypoints } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { computeAtlasBounds, convexHull, projectCatalogCoordinate, type AtlasPoint } from "@/lib/atlas";
import type { ObjectSummary } from "@/lib/types";

const COLORS: Record<string, string> = { CONFIRMED: "#72E8BC", "FALSE POSITIVE": "#FF8F9C", CANDIDATE: "#FFBD7A" };
interface ViewState { zoom: number; x: number; y: number }

export function StarAtlas({ objects, selectedId, onSelect }: { objects: ObjectSummary[]; selectedId?: string; onSelect: (object: ObjectSummary) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<ViewState>({ zoom: 1, x: 0, y: 0 });
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [hovered, setHovered] = useState<ObjectSummary | null>(null);
  const [pointer, setPointer] = useState<AtlasPoint | null>(null);
  const [drag, setDrag] = useState<{ x: number; y: number; moved: boolean } | null>(null);
  const [showMesh, setShowMesh] = useState(true);
  const [showDensity, setShowDensity] = useState(true);
  const positioned = useMemo(() => objects.filter((object) => Number.isFinite(object.ra_deg) && Number.isFinite(object.dec_deg)), [objects]);
  const bounds = useMemo(() => computeAtlasBounds(positioned), [positioned]);

  const baseCoordinate = useCallback((object: ObjectSummary, width: number, height: number) => projectCatalogCoordinate(Number(object.ra_deg), Number(object.dec_deg), bounds, width, height), [bounds]);
  const applyView = useCallback((point: AtlasPoint, width: number, height: number) => ({ x: (point.x - width / 2) * view.zoom + width / 2 + view.x, y: (point.y - height / 2) * view.zoom + height / 2 + view.y }), [view]);
  const coordinates = useCallback((object: ObjectSummary, width: number, height: number) => applyView(baseCoordinate(object, width, height), width, height), [applyView, baseCoordinate]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(([entry]) => setSize({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !size.width || !size.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size.width * ratio); canvas.height = Math.round(size.height * ratio);
    canvas.style.width = `${size.width}px`; canvas.style.height = `${size.height}px`;
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, size.width, size.height);

    const background = ctx.createRadialGradient(size.width * .52, size.height * .48, 0, size.width * .52, size.height * .48, size.width * .72);
    background.addColorStop(0, "#0a1b2d"); background.addColorStop(.5, "#06111e"); background.addColorStop(1, "#02070e");
    ctx.fillStyle = background; ctx.fillRect(0, 0, size.width, size.height);

    const gridRa = 6; const gridDec = 5;
    ctx.font = "10px 'Cascadia Code', monospace"; ctx.lineWidth = 1;
    for (let index = 0; index <= gridRa; index += 1) {
      const ra = bounds.minRa + ((bounds.maxRa - bounds.minRa) * index) / gridRa;
      const base = projectCatalogCoordinate(ra, bounds.centerDec, bounds, size.width, size.height);
      const point = applyView(base, size.width, size.height);
      ctx.strokeStyle = index === Math.round(gridRa / 2) ? "rgba(117,230,255,.18)" : "rgba(117,230,255,.085)";
      ctx.setLineDash(index === Math.round(gridRa / 2) ? [] : [2, 7]); ctx.beginPath(); ctx.moveTo(point.x, 0); ctx.lineTo(point.x, size.height); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = "rgba(143,171,191,.62)"; ctx.fillText(`${(ra / 15).toFixed(2)}h`, Math.max(8, Math.min(point.x + 7, size.width - 48)), 22);
    }
    for (let index = 0; index <= gridDec; index += 1) {
      const dec = bounds.minDec + ((bounds.maxDec - bounds.minDec) * index) / gridDec;
      const base = projectCatalogCoordinate(bounds.centerRa, dec, bounds, size.width, size.height);
      const point = applyView(base, size.width, size.height);
      ctx.strokeStyle = "rgba(117,230,255,.075)"; ctx.setLineDash([2, 7]); ctx.beginPath(); ctx.moveTo(0, point.y); ctx.lineTo(size.width, point.y); ctx.stroke();
      ctx.setLineDash([]); ctx.fillStyle = "rgba(143,171,191,.55)"; ctx.fillText(`${dec.toFixed(1)}°`, 12, Math.max(38, Math.min(point.y - 6, size.height - 12)));
    }

    const plotted = positioned.map((object) => ({ object, point: coordinates(object, size.width, size.height) }));
    const footprint = convexHull(plotted.map(({ point }) => point));
    if (footprint.length > 2) {
      const fieldGradient = ctx.createLinearGradient(0, 0, size.width, size.height);
      fieldGradient.addColorStop(0, "rgba(117,230,255,.025)"); fieldGradient.addColorStop(.55, "rgba(71,145,197,.065)"); fieldGradient.addColorStop(1, "rgba(173,153,255,.025)");
      ctx.beginPath(); footprint.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y)); ctx.closePath();
      ctx.fillStyle = fieldGradient; ctx.fill(); ctx.strokeStyle = "rgba(117,230,255,.16)"; ctx.setLineDash([5, 9]); ctx.stroke(); ctx.setLineDash([]);
    }

    if (showDensity) plotted.forEach(({ point }) => {
      if (point.x < -60 || point.x > size.width + 60 || point.y < -60 || point.y > size.height + 60) return;
      const halo = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, 34 * Math.sqrt(view.zoom));
      halo.addColorStop(0, "rgba(117,230,255,.075)"); halo.addColorStop(1, "rgba(117,230,255,0)");
      ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(point.x, point.y, 34 * Math.sqrt(view.zoom), 0, Math.PI * 2); ctx.fill();
    });

    if (showMesh) {
      ctx.lineWidth = .7; ctx.strokeStyle = "rgba(117,230,255,.105)";
      plotted.forEach(({ point }, index) => {
        const nearest = plotted.map(({ point: target }, targetIndex) => ({ target, targetIndex, distance: Math.hypot(target.x - point.x, target.y - point.y) })).filter(({ targetIndex }) => targetIndex !== index).sort((a, b) => a.distance - b.distance).slice(0, 2);
        nearest.forEach(({ target, targetIndex, distance }) => { if (targetIndex < index || distance > 125 * Math.sqrt(view.zoom)) return; ctx.beginPath(); ctx.moveTo(point.x, point.y); ctx.lineTo(target.x, target.y); ctx.stroke(); });
      });
    }

    plotted.forEach(({ object, point }) => {
      if (point.x < -24 || point.x > size.width + 24 || point.y < -24 || point.y > size.height + 24) return;
      const active = object.object_id === selectedId; const isHovered = object.object_id === hovered?.object_id;
      const radius = active ? 5.8 : object.has_cached_observation ? 3.3 : 2.4;
      const color = COLORS[object.catalog_disposition] ?? "#B8C9D7";
      if (active || isHovered) {
        ctx.strokeStyle = active ? "rgba(117,230,255,.9)" : "rgba(255,255,255,.48)"; ctx.lineWidth = active ? 1.4 : 1;
        ctx.beginPath(); ctx.arc(point.x, point.y, active ? 14 : 10, 0, Math.PI * 2); ctx.stroke();
        if (active) { ctx.beginPath(); ctx.arc(point.x, point.y, 21, 0, Math.PI * 2); ctx.strokeStyle = "rgba(117,230,255,.22)"; ctx.setLineDash([2, 5]); ctx.stroke(); ctx.setLineDash([]); }
      }
      ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.shadowBlur = active ? 22 : object.has_cached_observation ? 10 : 5; ctx.shadowColor = color;
      ctx.beginPath();
      if (object.catalog_disposition === "CANDIDATE") { ctx.moveTo(point.x, point.y - radius - 1); ctx.lineTo(point.x + radius + 1, point.y); ctx.lineTo(point.x, point.y + radius + 1); ctx.lineTo(point.x - radius - 1, point.y); ctx.closePath(); ctx.fill(); }
      else if (object.catalog_disposition === "FALSE POSITIVE") { ctx.moveTo(point.x - radius, point.y - radius); ctx.lineTo(point.x + radius, point.y + radius); ctx.moveTo(point.x + radius, point.y - radius); ctx.lineTo(point.x - radius, point.y + radius); ctx.stroke(); }
      else { ctx.arc(point.x, point.y, radius, 0, Math.PI * 2); ctx.fill(); }
      ctx.shadowBlur = 0;
    });

    const selected = plotted.find(({ object }) => object.object_id === selectedId);
    if (selected && selected.point.x > 0 && selected.point.x < size.width && selected.point.y > 0 && selected.point.y < size.height) {
      const { point, object } = selected; const labelX = Math.min(point.x + 26, size.width - 154); const labelY = Math.max(point.y - 30, 54);
      ctx.strokeStyle = "rgba(117,230,255,.36)"; ctx.beginPath(); ctx.moveTo(point.x + 14, point.y - 9); ctx.lineTo(labelX - 5, labelY + 6); ctx.stroke();
      ctx.fillStyle = "rgba(5,14,25,.88)"; ctx.strokeStyle = "rgba(117,230,255,.2)"; ctx.beginPath(); ctx.roundRect(labelX, labelY - 14, 142, 43, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#EAF7FB"; ctx.font = "600 11px 'Manrope Variable', sans-serif"; ctx.fillText(object.display_name, labelX + 10, labelY + 3);
      ctx.fillStyle = "rgba(143,171,191,.72)"; ctx.font = "9px 'Cascadia Code', monospace"; ctx.fillText(`KIC ${object.kic}`, labelX + 10, labelY + 18);
    }
  }, [applyView, bounds, coordinates, hovered, positioned, selectedId, showDensity, showMesh, size, view.zoom]);

  const nearest = useCallback((clientX: number, clientY: number) => {
    const rect = canvasRef.current?.getBoundingClientRect(); if (!rect) return null;
    let match: ObjectSummary | null = null; let distance = 16;
    positioned.forEach((object) => { const point = coordinates(object, rect.width, rect.height); const next = Math.hypot(point.x - (clientX - rect.left), point.y - (clientY - rect.top)); if (next < distance) { distance = next; match = object; } });
    return match;
  }, [coordinates, positioned]);

  const focusSelected = () => {
    const selected = positioned.find((object) => object.object_id === selectedId); if (!selected || !size.width || !size.height) return;
    const zoom = 2.1; const point = baseCoordinate(selected, size.width, size.height);
    setView({ zoom, x: -(point.x - size.width / 2) * zoom, y: -(point.y - size.height / 2) * zoom });
  };

  const setZoom = (factor: number, origin = { x: size.width / 2, y: size.height / 2 }) => setView((current) => {
    const zoom = Math.min(7, Math.max(1, current.zoom * factor)); const ratio = zoom / current.zoom;
    return { zoom, x: origin.x - size.width / 2 - (origin.x - size.width / 2 - current.x) * ratio, y: origin.y - size.height / 2 - (origin.y - size.height / 2 - current.y) * ratio };
  });

  return <div className="star-atlas" ref={containerRef}>
    <canvas ref={canvasRef} tabIndex={0} aria-label={`Kepler catalog field atlas with ${positioned.length} real coordinates`} onPointerMove={(event) => { const rect = event.currentTarget.getBoundingClientRect(); setPointer({ x: event.clientX - rect.left, y: event.clientY - rect.top }); if (drag) { const deltaX = event.clientX - drag.x; const deltaY = event.clientY - drag.y; setView((value) => ({ ...value, x: value.x + deltaX, y: value.y + deltaY })); setDrag({ x: event.clientX, y: event.clientY, moved: drag.moved || Math.hypot(deltaX, deltaY) > 2 }); } else setHovered(nearest(event.clientX, event.clientY)); }} onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); setDrag({ x: event.clientX, y: event.clientY, moved: false }); }} onPointerUp={(event) => { if (drag && !drag.moved) { const object = nearest(event.clientX, event.clientY); if (object) onSelect(object); } setDrag(null); }} onPointerLeave={() => { setDrag(null); setHovered(null); setPointer(null); }} onWheel={(event) => { event.preventDefault(); const rect = event.currentTarget.getBoundingClientRect(); setZoom(event.deltaY > 0 ? .88 : 1.14, { x: event.clientX - rect.left, y: event.clientY - rect.top }); }} />
    <div className="atlas-readout"><span>Kepler catalog field</span><strong>{(bounds.maxRa / 15).toFixed(2)}h — {(bounds.minRa / 15).toFixed(2)}h</strong><small>Dec {bounds.minDec.toFixed(1)}° to +{bounds.maxDec.toFixed(1)}°</small></div>
    <div className="atlas-layer-tools" aria-label="Atlas layers"><button className={showMesh ? "active" : ""} aria-pressed={showMesh} onClick={() => setShowMesh((value) => !value)}><Waypoints /><span>Proximity mesh</span></button><button className={showDensity ? "active" : ""} aria-pressed={showDensity} onClick={() => setShowDensity((value) => !value)}><Layers3 /><span>Catalog density</span></button></div>
    <div className="atlas-tools" aria-label="Atlas navigation"><button aria-label="Focus selected object" onClick={focusSelected}><Focus /></button><button aria-label="Zoom in" onClick={() => setZoom(1.25)}><Plus /></button><button aria-label="Zoom out" onClick={() => setZoom(.8)}><Minus /></button><button aria-label="Reset atlas" onClick={() => setView({ zoom: 1, x: 0, y: 0 })}><RotateCcw /></button></div>
    <div className="atlas-legend"><span><i className="legend-confirmed" />Confirmed</span><span><i className="legend-candidate" />Candidate</span><span><i className="legend-false" />False positive</span></div>
    <div className="atlas-method">Equirectangular catalog view · mesh indicates screen proximity only</div>
    {hovered && <div className="atlas-tooltip" style={pointer ? { left: Math.min(pointer.x + 18, size.width - 245), top: Math.max(78, pointer.y - 16) } : undefined}><strong>{hovered.display_name}</strong><span>KIC {hovered.kic}</span><span>RA {hovered.ra_deg?.toFixed(5)}° · Dec {hovered.dec_deg?.toFixed(5)}°</span><small>{hovered.catalog_disposition} · {hovered.has_cached_observation ? "observation cached" : "catalog only"}</small></div>}
  </div>;
}
