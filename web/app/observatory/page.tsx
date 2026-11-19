"use client";

import { Activity, Radio, ScanLine } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { OrbitalSystem } from "@/components/orbital-system";
import { ScientificPlot } from "@/components/scientific-plot";
import { AuditTable, ErrorState, Loading, Metric, PageIntro, Panel, ScientificNotice } from "@/components/ui";
import { api } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { Inference, LightCurve, ObjectDetail, ObjectPage, ObjectSummary, Scientific } from "@/lib/types";

interface Observation { mission: string; author: string; cadence_seconds: number; observation_start_bkjd: number; observation_end_bkjd: number; observation_span_days: number; segments: string[]; raw_records: number; processed_records: number; provenance: Record<string, unknown> }

export default function ObservatoryPage() {
  const [objects, setObjects] = useState<ObjectSummary[]>([]); const [objectId, setObjectId] = useState("");
  const [detail, setDetail] = useState<ObjectDetail | null>(null); const [observation, setObservation] = useState<Observation | null>(null);
  const [curves, setCurves] = useState<Record<string, LightCurve>>({}); const [inferences, setInferences] = useState<Inference[]>([]);
  const [mode, setMode] = useState<"processed" | "raw">("processed"); const [error, setError] = useState(""); const [loading, setLoading] = useState(true);

  useEffect(() => { void api<ObjectPage<ObjectSummary>>("/api/v1/objects?observed_only=true&limit=100").then((page) => { setObjects(page.items); const requested = new URLSearchParams(window.location.search).get("object"); setObjectId(page.items.find((item) => item.object_id === requested)?.object_id ?? page.items[0]?.object_id ?? ""); }).catch((reason) => setError(reason.message)); }, []);
  useEffect(() => { if (!objectId) return; setLoading(true); setError(""); void Promise.all([
    api<Scientific<ObjectDetail>>(`/api/v1/objects/${objectId}`), api<Scientific<Observation>>(`/api/v1/objects/${objectId}/observation`),
    api<Scientific<LightCurve>>(`/api/v1/objects/${objectId}/light-curve?view=raw&limit=2200`), api<Scientific<LightCurve>>(`/api/v1/objects/${objectId}/light-curve?view=processed&limit=2200`),
    api<Scientific<LightCurve>>(`/api/v1/objects/${objectId}/light-curve?view=phase&limit=2200`), api<Scientific<LightCurve>>(`/api/v1/objects/${objectId}/light-curve?view=local&limit=900`),
    api<Inference>(`/api/v1/inference/ann-01`, { method: "POST", body: JSON.stringify({ object_id: objectId }) }).catch(() => null), api<Inference>(`/api/v1/inference/ann-02`, { method: "POST", body: JSON.stringify({ object_id: objectId }) }).catch(() => null),
  ]).then(([metadata, obs, raw, processed, phase, local, ann01, ann02]) => { setDetail(metadata.data); setObservation(obs.data); setCurves({ raw: raw.data, processed: processed.data, phase: phase.data, local: local.data }); setInferences([ann01, ann02].filter(Boolean) as Inference[]); }).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, [objectId]);

  const mainPlot = useMemo(() => { const curve = curves[mode]; if (!curve) return []; const x = curve.data.time_bkjd ?? []; const key = mode === "raw" ? "flux" : "detrended_flux"; return [{ x, y: curve.data[key], type: "scattergl", mode: "lines", name: mode === "raw" ? "SAP/PDCSAP observed flux" : "Detrended flux", line: { color: mode === "raw" ? "#D7E1E8" : "#70DBF7", width: 1 }, hovertemplate: "BKJD %{x:.5f}<br>Flux %{y:.6f}<extra></extra>" }]; }, [curves, mode]);
  const phasePlot = useMemo(() => curves.phase ? [{ x: curves.phase.data.phase, y: curves.phase.data.detrended_flux, type: "scattergl", mode: "markers", marker: { color: "#80A8F5", size: 3, opacity: .55 }, hovertemplate: "Phase %{x:.5f}<br>Flux %{y:.6f}<extra></extra>" }] : [], [curves.phase]);
  const localPlot = useMemo(() => curves.local ? [{ x: curves.local.data.phase_hours, y: curves.local.data.flux, type: "scatter", mode: "lines+markers", line: { color: "#BFA7EF", width: 1.4 }, marker: { size: 3 }, hovertemplate: "%{x:.3f} h<br>Flux %{y:.6f}<extra></extra>" }] : [], [curves.local]);

  return <div className="page"><PageIntro eyebrow="Deep Space Observatory / Kepler" title="Read the rhythm of a distant star." description="Inspect observed luminosity, processing stages, orbital phase, catalog context, and both neural interpretations without leaving the scientific record."><label className="object-selector"><span>Active object</span><select value={objectId} onChange={(event) => setObjectId(event.target.value)}>{objects.map((object) => <option key={object.object_id} value={object.object_id}>{object.display_name} · KIC {object.kic}</option>)}</select></label></PageIntro>
    {error && <ErrorState message={error} />}{loading ? <Loading label="Synchronizing observation instruments…" /> : detail && <>
      <section className="object-command"><div><span className={`disposition ${detail.disposition.toLowerCase().replace(" ", "-")}`}>{detail.disposition}</span><h2>{detail.display_name}</h2><p>{detail.kepler_name ?? `KIC ${detail.kic}`} · RA {detail.ra_deg?.toFixed(5)}° · Dec {detail.dec_deg?.toFixed(5)}°</p></div><div className="signal-pulse"><Radio /><span>Observation cache online<small>{observation?.mission} · {observation?.author}</small></span></div></section>
      <section className="metric-strip"><Metric label="Orbital period" value={detail.period_days} unit="days" /><Metric label="Transit duration" value={detail.duration_hours} unit="hours" /><Metric label="Transit depth" value={detail.depth_ppm} unit="ppm" tone="signal" /><Metric label="Observation span" value={observation?.observation_span_days} unit="days" /><Metric label="Cadence" value={observation?.cadence_seconds} unit="seconds" /></section>
      <Panel className="instrument-panel" eyebrow="Observed time domain" title="Luminosity stream" action={<div className="segmented"><button className={mode === "raw" ? "active" : ""} onClick={() => setMode("raw")}>Original</button><button className={mode === "processed" ? "active" : ""} onClick={() => setMode("processed")}>Detrended</button></div>}>
        <ScientificPlot title={`${detail.display_name} ${mode} luminosity`} data={mainPlot} layout={{ height: 440, xaxis: { title: "Time (BKJD)" }, yaxis: { title: "Relative flux" }, dragmode: "pan" }} />
        <div className="plot-caption"><ScanLine /><span>{curves[mode]?.records_returned} of {curves[mode]?.records_total} measurements rendered. Wheel zoom and drag pan are available.</span></div>
      </Panel>
      <section className="two-column">
        <Panel eyebrow="Catalog ephemeris" title="Phase-folded orbit"><ScientificPlot title="Phase-folded light curve" data={phasePlot} layout={{ height: 340, xaxis: { title: "Orbital phase" }, yaxis: { title: "Detrended flux" } }} /><p className="caption">Repeated measurements are folded using the NASA catalog period and transit epoch.</p></Panel>
        <Panel eyebrow="Transit window" title="Expected event detail"><ScientificPlot title="Local transit window" data={localPlot} layout={{ height: 340, xaxis: { title: "Hours from catalog transit center" }, yaxis: { title: "Relative flux" }, shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { color: "#EFB681", dash: "dot" } }] }} /><p className="caption">The center marker follows the catalog ephemeris; it is not a newly measured transit time.</p></Panel>
      </section>
      <section className="orbit-neural-grid">
        <Panel className="orbital-context-panel" eyebrow="System context" title="Orbital representation"><OrbitalSystem object={detail} /></Panel>
        <Panel className="neural-interpretation-panel" eyebrow="Independent model readings" title="Neural interpretation" action={<span className="data-tag"><Activity /> Real inference</span>}>
          <div className="neural-readings">{inferences.map((item) => <article key={item.model_id}>
            <header><span>{item.model_id.toUpperCase()}</span><small>{item.input_kind.replaceAll("_", " ")}</small></header>
            <div className="reading-score"><strong>{formatNumber(item.score, 6)}</strong><span>model score</span></div>
            <div className="reading-decision"><span>{item.predicted_class.replaceAll("_", " ")}</span><small>decision threshold {formatNumber(item.threshold, 3)}</small></div>
            <div className="score-track" aria-label={`${item.model_id} score ${item.score}`}><i style={{ width: `${Math.max(0, Math.min(1, item.score)) * 100}%` }} /><b style={{ left: `${Math.max(0, Math.min(1, item.threshold)) * 100}%` }} /></div>
          </article>)}</div>
          <ScientificNotice>Scores are uncalibrated model outputs. They are neither posterior probabilities nor confirmations.</ScientificNotice>
        </Panel>
      </section>
      <Panel eyebrow="Secondary technical view" title="Observation provenance"><AuditTable headers={["Mission", "Author", "Segments", "Raw records", "Processed records", "BKJD interval"]} rows={[[observation?.mission, observation?.author, observation?.segments.join(", "), observation?.raw_records, observation?.processed_records, `${formatNumber(observation?.observation_start_bkjd, 4)} — ${formatNumber(observation?.observation_end_bkjd, 4)}`]]} /></Panel>
    </>}
  </div>;
}
