"use client";

import { AlertOctagon, Fingerprint, SearchCheck, Waves } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ScientificPlot } from "@/components/scientific-plot";
import { AuditTable, EmptyState, ErrorState, Loading, Metric, PageIntro, Panel, ScientificNotice } from "@/components/ui";
import { api } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { CampaignDetail, CampaignSummary, CampaignTarget, DiscoverySeries, ObjectPage, Scientific } from "@/lib/types";

export default function CandidatesPage() {
  const [campaign, setCampaign] = useState<CampaignDetail | null>(null); const [selected, setSelected] = useState<CampaignTarget | null>(null);
  const [series, setSeries] = useState<DiscoverySeries | null>(null); const [error, setError] = useState("");
  useEffect(() => { void api<ObjectPage<CampaignSummary>>("/api/v1/discovery/campaigns?limit=1").then(async (page) => { if (!page.items[0]) return; const payload = await api<Scientific<CampaignDetail>>(`/api/v1/discovery/campaigns/${page.items[0].campaign_id}`); setCampaign(payload.data); setSelected(payload.data.targets.find((target) => target.interpretation === "unassociated_inconclusive_signal") ?? payload.data.targets[0] ?? null); }).catch((reason) => setError(reason.message)); }, []);
  useEffect(() => { if (!campaign || !selected) return; setSeries(null); void api<Scientific<DiscoverySeries>>(`/api/v1/discovery/campaigns/${campaign.campaign.campaign_id}/targets/${selected.kic}/series?limit=1600`).then((payload) => setSeries(payload.data)).catch((reason) => setError(reason.message)); }, [campaign, selected]);
  const dossiers = useMemo(() => campaign?.targets.filter((target) => target.interpretation === "unassociated_inconclusive_signal") ?? [], [campaign]);
  const timePlot = series ? [{ x: series.time_bkjd, y: series.detrended_flux, type: "scattergl", mode: "lines", line: { color: "#70DBF7", width: 1 }, hovertemplate: "BKJD %{x:.4f}<br>Flux %{y:.6f}<extra></extra>" }] : [];
  const periodPlot = series ? [{ x: series.periods, y: series.powers, type: "scatter", mode: "lines", fill: "tozeroy", fillcolor: "rgba(191,167,239,.10)", line: { color: "#BFA7EF", width: 1.4 }, hovertemplate: "%{x:.5f} d<br>BLS power %{y:.6f}<extra></extra>" }] : [];
  const phasePlot = series ? [{ x: series.phase, y: series.phase_flux, type: "scattergl", mode: "markers", marker: { color: "#80A8F5", size: 2.5, opacity: .28 }, name: "Measurements" }, { x: series.phase_bin_centers, y: series.phase_bin_flux, type: "scatter", mode: "lines", line: { color: "#EFB681", width: 2.3 }, name: "Phase-bin median" }] : [];

  return <div className="page"><PageIntro eyebrow="Candidate intelligence / unresolved evidence" title="Investigate before you conclude." description="Each dossier separates measured screening evidence, catalog context, alternative hypotheses, and the limitations that prevent a discovery claim."><div className="catalog-chip"><Fingerprint />{dossiers.length} unresolved screens</div></PageIntro>
    {error && <ErrorState message={error} />}{!campaign ? <Loading label="Opening investigation archive…" /> : dossiers.length === 0 ? <EmptyState title="No unresolved screens" message="The selected persisted campaign contains no unassociated screening detections." /> : <>
      <section className="dossier-layout"><aside className="dossier-index"><span className="eyebrow">Investigation queue</span>{dossiers.map((target, index) => <button key={target.kic} className={selected?.kic === target.kic ? "active" : ""} onClick={() => setSelected(target)}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>KIC {target.kic}</strong><small>{formatNumber(target.period_days, 4)} d · rank {formatNumber(target.ranking_score, 2)}</small></div><i /></button>)}</aside>
        {selected && <article className="dossier"><header><div><span className="eyebrow">ASTRAL investigative dossier</span><h2>KIC {selected.kic}</h2><p>Selection reference {selected.selection_reference} · campaign {campaign.campaign.campaign_id}</p></div><span className="investigation-state"><Waves />Inconclusive signal</span></header>
          <section className="dossier-metrics"><Metric label="BLS period" value={selected.period_days} unit="days" /><Metric label="Duration" value={selected.duration_hours} unit="hours" /><Metric label="Depth" value={selected.depth_ppm} unit="ppm" /><Metric label="Screening SNR" value={selected.depth_snr} /><Metric label="Observed events" value={selected.observed_events} /></section>
          <div className="evidence-balance"><div><SearchCheck /><span>Evidence recorded<strong>{formatNumber((selected.event_coverage ?? 0) * 100, 1)}% event coverage</strong><small>{formatNumber((selected.phase_coverage ?? 0) * 100, 1)}% phase coverage</small></span></div><div className={selected.possible_eclipsing_binary ? "warning" : ""}><AlertOctagon /><span>Alternative hypothesis<strong>{selected.possible_eclipsing_binary ? "Eclipsing-binary compatible" : "No EB diagnostic triggered"}</strong><small>Instrumental and stellar variability remain possible</small></span></div></div>
        </article>}
      </section>
      {!series ? <Loading label="Loading persisted numerical evidence…" /> : <>
        <Panel eyebrow="Measured luminosity" title="Signal record"><ScientificPlot title={`KIC ${selected?.kic} detrended luminosity`} data={timePlot} layout={{ height: 390, xaxis: { title: "Time (BKJD)" }, yaxis: { title: "Detrended relative flux" } }} /></Panel>
        <section className="two-column"><Panel eyebrow="Box Least Squares" title="Period search landscape"><ScientificPlot title="BLS periodogram" data={periodPlot} layout={{ height: 350, xaxis: { title: "Trial period (days)" }, yaxis: { title: "BLS power" }, shapes: selected?.period_days ? [{ type: "line", x0: selected.period_days, x1: selected.period_days, y0: 0, y1: 1, yref: "paper", line: { color: "#EFB681", dash: "dot" } }] : [] }} /></Panel><Panel eyebrow="Period-aligned evidence" title="Phase-folded signal"><ScientificPlot title="BLS phase fold" data={phasePlot} layout={{ height: 350, xaxis: { title: "Orbital phase" }, yaxis: { title: "Relative flux" }, showlegend: true, legend: { orientation: "h" } }} /></Panel></section>
      </>}
      <Panel eyebrow="Hypotheses and boundaries" title="Investigation state"><div className="limitations-grid">{selected?.limitations.map((limitation) => <div key={limitation}><AlertOctagon /><p>{limitation}</p></div>)}</div><ScientificNotice>No local catalog-period association is not evidence of novelty. This object remains an unresolved screening result, not a planet candidate or discovery.</ScientificNotice></Panel>
      <Panel eyebrow="Secondary audit view" title="Candidate ledger"><AuditTable headers={["KIC", "Reference", "Period (d)", "Depth (ppm)", "SNR", "EB flag", "State"]} rows={dossiers.map((target) => [target.kic, target.selection_reference, formatNumber(target.period_days, 5), formatNumber(target.depth_ppm, 1), formatNumber(target.depth_snr, 2), target.possible_eclipsing_binary ? "Raised" : "Not raised", "Inconclusive"])} /></Panel>
    </>}
  </div>;
}
