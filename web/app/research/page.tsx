"use client";

import { Activity, Archive, CheckCircle2, CircleDashed, Database, GitCompareArrows, LockKeyhole, Microscope, Radio, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ScientificPlot } from "@/components/scientific-plot";
import { AuditTable, ErrorState, Loading, Metric, PageIntro, Panel, ScientificNotice } from "@/components/ui";
import { api } from "@/lib/api";
import { formatDate, formatNumber } from "@/lib/format";
import { experimentTimelineKey, mergeExperimentTimeline } from "@/lib/research";
import type { CandidateIntelligenceIndex, Comparison, EpochMetric, ExperimentSummary, ObjectPage, ResearchCampaignEvent, ResearchCampaignJob, ResearchCampaignProvenance, Scientific } from "@/lib/types";

const activeStates = new Set(["queued", "running"]);

export default function ResearchPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [history01, setHistory01] = useState<EpochMetric[]>([]);
  const [history02, setHistory02] = useState<EpochMetric[]>([]);
  const [candidateArchive, setCandidateArchive] = useState<CandidateIntelligenceIndex | null>(null);
  const [campaigns, setCampaigns] = useState<ResearchCampaignJob[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [events, setEvents] = useState<ResearchCampaignEvent[]>([]);
  const [provenance, setProvenance] = useState<ResearchCampaignProvenance | null>(null);
  const [error, setError] = useState("");

  const loadCampaigns = useCallback(async () => {
    const response = await api<ObjectPage<ResearchCampaignJob>>("/api/v1/discovery/jobs?limit=100");
    setCampaigns(response.items);
    setSelectedId((current) => current || response.items[0]?.job_id || "");
  }, []);

  useEffect(() => {
    void Promise.all([
      api<ObjectPage<ExperimentSummary>>("/api/v1/experiments?stage=stage06&limit=50"),
      api<ObjectPage<ExperimentSummary>>("/api/v1/experiments?stage=stage07&limit=50"),
      api<Scientific<Comparison>>("/api/v1/metrics/comparison"),
      api<ObjectPage<EpochMetric>>("/api/v1/models/ann-01/history?limit=500"),
      api<ObjectPage<EpochMetric>>("/api/v1/models/ann-02/history?limit=500"),
      api<Scientific<CandidateIntelligenceIndex>>("/api/v1/candidate-intelligence"),
      loadCampaigns(),
    ]).then(([stage06, stage07, metrics, ann01, ann02, candidates]) => {
      setExperiments(mergeExperimentTimeline(stage06.items, stage07.items));
      setComparison(metrics.data); setHistory01(ann01.items); setHistory02(ann02.items); setCandidateArchive(candidates.data);
    }).catch((reason: Error) => setError(reason.message));
  }, [loadCampaigns]);

  const selected = campaigns.find((campaign) => campaign.job_id === selectedId) ?? null;
  useEffect(() => {
    if (!selectedId) { setEvents([]); setProvenance(null); return; }
    const loadEvidence = async () => {
      const [eventResponse, provenanceResponse] = await Promise.all([
        api<ObjectPage<ResearchCampaignEvent>>(`/api/v1/discovery/jobs/${selectedId}/events?limit=500`),
        api<Scientific<ResearchCampaignProvenance>>(`/api/v1/discovery/jobs/${selectedId}/provenance`),
      ]);
      setEvents(eventResponse.items); setProvenance(provenanceResponse.data);
      if (activeStates.has(selected?.status ?? "")) await loadCampaigns();
    };
    void loadEvidence().catch((reason: Error) => setError(reason.message));
    if (!activeStates.has(selected?.status ?? "")) return;
    const timer = window.setInterval(() => void loadEvidence().catch(() => undefined), 1600);
    return () => window.clearInterval(timer);
  }, [loadCampaigns, selected?.status, selectedId]);

  const learningPlot = useMemo(() => [
    { x: history01.map((row) => row.epoch), y: history01.map((row) => row.validation_loss), type: "scatter", mode: "lines", name: "ANN-01 validation", line: { color: "#70DBF7", width: 2 } },
    { x: history02.map((row) => row.epoch), y: history02.map((row) => row.validation_loss), type: "scatter", mode: "lines", name: "ANN-02 validation", line: { color: "#BFA7EF", width: 2 } },
  ], [history01, history02]);
  const metricNames = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"] as const;
  const benchmarkPlot = comparison ? [
    { x: metricNames.map(labelMetric), y: metricNames.map((key) => comparison.ann01_metrics[key]), type: "bar", name: "ANN-01", marker: { color: "#70DBF7" } },
    { x: metricNames.map(labelMetric), y: metricNames.map((key) => comparison.ann02_metrics[key]), type: "bar", name: "ANN-02", marker: { color: "#BFA7EF" } },
  ] : [];
  const agreementPlot = comparison ? [{ x: comparison.rows.map((row) => row.ann01_score), y: comparison.rows.map((row) => row.ann02_score), text: comparison.rows.map((row) => `${row.object_id} · ${row.catalog_disposition}`), type: "scatter", mode: "markers", marker: { color: comparison.rows.map((row) => row.models_agree ? "#7BD7BA" : "#EFC77C"), size: 9, line: { color: "#F0F5F8", width: .4 } }, hovertemplate: "%{text}<br>ANN-01 %{x:.4f}<br>ANN-02 %{y:.4f}<extra></extra>" }] : [];
  const completed = campaigns.filter((campaign) => campaign.status === "completed").length;
  const interrupted = campaigns.filter((campaign) => ["failed", "cancelled", "interrupted"].includes(campaign.status)).length;

  return <div className="page research-page"><PageIntro eyebrow="Research archive / reproducible evidence" title="Every result leaves a trace." description="Inspect persistent campaign operations, immutable scientific artifacts, controlled experiments, and equivalent-cohort model evidence from one provenance-aware archive."><div className="catalog-chip"><Archive />{campaigns.length + experiments.length} traceable records</div></PageIntro>
    {error && <ErrorState message={error} />}

    <section className="metric-strip"><Metric label="Campaign plans" value={campaigns.length} unit="records" /><Metric label="Completed" value={completed} tone="good" /><Metric label="Recoverable states" value={interrupted} tone="signal" /><Metric label="Training experiments" value={experiments.length} /><Metric label="Frozen test accessed" value="No" tone="good" /></section>

    <Panel className="research-operations" eyebrow="Stage 12 / operational persistence" title="Discovery campaign archive" action={<span className="data-tag"><Database /> SQLite event ledger</span>}>
      {campaigns.length === 0 ? <ScientificNotice>No operational campaign has been planned yet. Mission Control in Discover creates the first persisted record before execution.</ScientificNotice> : <div className="research-operations-grid">
        <div className="campaign-orbit-timeline" role="list" aria-label="Persisted discovery campaigns">{campaigns.map((campaign) => <button key={campaign.job_id} className={campaign.job_id === selectedId ? "selected" : ""} onClick={() => setSelectedId(campaign.job_id)}><span className={`state-beacon state-${campaign.status}`}>{activeStates.has(campaign.status) ? <Radio /> : campaign.status === "completed" ? <CheckCircle2 /> : <CircleDashed />}</span><span><small>{campaign.status} · {formatDate(campaign.updated_at_utc)}</small><strong>{campaign.campaign_id}</strong><em>{campaign.current}/{campaign.total} targets · {campaign.allow_network ? "network authorized" : "cache only"}</em></span></button>)}</div>
        {selected && <div className="campaign-evidence"><header><div><small>Selected operational record</small><h3>{selected.campaign_id}</h3></div><span className={`mission-state state-${selected.status}`}>{selected.status}</span></header><div className="evidence-facts"><span><small>Current stage</small><strong>{selected.stage.replaceAll("_", " ")}</strong></span><span><small>Last target</small><strong>{selected.current_kic ? `KIC ${selected.current_kic}` : "None"}</strong></span><span><small>Created</small><strong>{formatDate(selected.created_at_utc)}</strong></span><span><small>Artifact</small><strong>{selected.artifact_path ? "Persisted" : "Pending"}</strong></span></div><div className="event-stream compact">{events.slice().reverse().map((event) => <article key={event.event_id} className={`event-${event.level}`}><i><Activity /></i><div><span>{event.event_type} · {event.stage}</span><strong>{event.message}</strong><small>{formatDate(event.created_at_utc)}{event.kic ? ` · KIC ${event.kic}` : ""}</small></div></article>)}</div>{provenance && <div className="provenance-seal"><ShieldCheck /><span><strong>Scientific boundary retained</strong><small>{provenance.mission} · {provenance.selection_strategy.replaceAll("_", " ")} · frozen test excluded</small></span></div>}</div>}
      </div>}
    </Panel>

    {!comparison ? <Loading label="Indexing persisted scientific experiments…" /> : <>
      {candidateArchive && <Panel className="research-candidate-archive" eyebrow="Stage 11 / scientific verification" title="Candidate intelligence archive" action={<span className="data-tag">{candidateArchive.campaign_id}</span>}><div className="research-candidate-summary"><span><strong>{candidateArchive.summary.investigated_signals}</strong>dossiers</span><span><strong>{candidateArchive.summary.catalog_associated}</strong>catalog associated</span><span><strong>{candidateArchive.summary.unassociated_inconclusive}</strong>inconclusive</span><span><strong>{candidateArchive.summary.confirmed_discoveries}</strong>confirmed discoveries</span></div><div className="research-dossier-strip">{candidateArchive.dossiers.map((dossier) => <a key={dossier.kic} href={`/candidates?kic=${dossier.kic}`}><small>{dossier.investigation_state === "catalog_associated" ? "Catalog relation" : "Unassociated"}</small><strong>KIC {dossier.kic}</strong><span>Priority {formatNumber(dossier.priority.score, 1)} · {dossier.review_count} reviews</span></a>)}</div><ScientificNotice>Every dossier retains catalog cross-checks, limitations, ranking components, source provenance, and review history. No record is presented as a confirmed discovery.</ScientificNotice></Panel>}
      <Panel className="timeline-panel" eyebrow="Training history" title="Model experiment timeline"><div className="research-timeline">{experiments.map((experiment, index) => <article key={experimentTimelineKey(experiment)} className={experiment.selected ? "selected" : ""}><div className="timeline-axis"><i /><span>{index < experiments.length - 1 && ""}</span></div><div><span>{experiment.stage} · {experiment.kind.replaceAll("_", " ")}</span><strong>{experiment.experiment_id}</strong><small>{formatDate(experiment.generated_at_utc)}</small>{experiment.selected && <em>Selected checkpoint</em>}</div></article>)}</div></Panel>
      <section className="two-column"><Panel eyebrow="Training telemetry" title="Validation loss history"><ScientificPlot title="Training validation loss" data={learningPlot} layout={{ height: 370, xaxis: { title: "Epoch" }, yaxis: { title: "Binary cross-entropy" }, showlegend: true, legend: { orientation: "h" } }} /><p className="caption">Curves retain their native epoch ranges and do not imply equivalent training protocols.</p></Panel><Panel eyebrow="Equivalent 47-object cohort" title="Model benchmark"><ScientificPlot title="ANN model metric comparison" data={benchmarkPlot} layout={{ height: 370, barmode: "group", yaxis: { title: "Metric", range: [0, 1] }, legend: { orientation: "h" } }} /></Panel></section>
      <section className="two-column"><Panel eyebrow="Like-for-like validation" title="Score agreement map"><ScientificPlot title="ANN-01 and ANN-02 score comparison" data={agreementPlot} layout={{ height: 380, xaxis: { title: "ANN-01 score", range: [0, 1] }, yaxis: { title: "ANN-02 score", range: [0, 1] }, shapes: [{ type: "line", x0: 0, y0: 0, x1: 1, y1: 1, line: { color: "#8295A5", dash: "dot" } }, { type: "line", x0: .5, y0: 0, x1: .5, y1: 1, line: { color: "rgba(239,182,129,.35)", dash: "dot" } }, { type: "line", x0: 0, y0: .5, x1: 1, y1: .5, line: { color: "rgba(239,182,129,.35)", dash: "dot" } }] }} /></Panel><Panel eyebrow="Scientific protocol" title="Comparison boundary"><div className="protocol-stack"><div><GitCompareArrows /><span><strong>Identical identities</strong><small>{comparison.cohort.definition}</small></span></div><div><LockKeyhole /><span><strong>Frozen test remains isolated</strong><small>No test object entered this comparison.</small></span></div><div><Microscope /><span><strong>Validation, not final evidence</strong><small>The cohort participated in ANN-02 checkpoint selection.</small></span></div></div><ScientificNotice>ANN-01 and ANN-02 use different scientific representations. Their uncalibrated scores are not interchangeable probabilities.</ScientificNotice></Panel></section>
      <Panel eyebrow="Secondary audit view" title="Reproducibility register"><AuditTable headers={["Experiment ID", "Stage", "Kind", "Generated (UTC)", "Selected"]} rows={experiments.map((experiment) => [experiment.experiment_id, experiment.stage, experiment.kind, formatDate(experiment.generated_at_utc), experiment.selected ? "Yes" : "No"])} /></Panel>
      <Panel eyebrow="Recorded limitations" title="Interpretation constraints"><div className="limitations-grid">{comparison.limitations.map((limitation) => <div key={limitation}><Microscope /><p>{limitation}</p></div>)}</div></Panel>
    </>}
  </div>;
}

function labelMetric(value: string) { return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()); }
