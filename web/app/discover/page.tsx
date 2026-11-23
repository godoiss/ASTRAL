"use client";

import { Activity, AlertTriangle, CheckCircle2, Crosshair, Database, FileSearch, History, LockKeyhole, Play, Radio, RotateCcw, ShieldCheck, Square, Telescope } from "lucide-react";
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { StarAtlas } from "@/components/star-atlas";
import { AuditTable, ErrorState, Loading, Metric, PageIntro, Panel, ScientificNotice } from "@/components/ui";
import { api, getAllObjects } from "@/lib/api";
import { formatDate, formatNumber } from "@/lib/format";
import type { CampaignDetail, CampaignSummary, ObjectPage, ObjectSummary, ResearchCampaignEvent, ResearchCampaignJob, Scientific } from "@/lib/types";

interface Plan {
  sample_size: number; cached_targets: number; protected_host_overlap: number[]; network_enabled: false;
  execution_enabled: boolean; downloads_enabled: boolean; maximum_campaign_downloads: number;
  targets: Array<{ kic: number; label: -1; cache_available: boolean; selection_reference: string }>;
}

const activeStates = new Set(["queued", "running"]);
const resumableStates = new Set(["cancelled", "failed", "interrupted"]);
const stageOrder = ["queued", "target", "acquisition", "processing", "bls", "completed"];

export default function DiscoverPage() {
  const [objects, setObjects] = useState<ObjectSummary[]>([]);
  const [selected, setSelected] = useState<ObjectSummary | null>(null);
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);
  const [campaign, setCampaign] = useState<CampaignDetail | null>(null);
  const [jobs, setJobs] = useState<ResearchCampaignJob[]>([]);
  const [job, setJob] = useState<ResearchCampaignJob | null>(null);
  const [events, setEvents] = useState<ResearchCampaignEvent[]>([]);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [sampleSize, setSampleSize] = useState(5);
  const [allowNetwork, setAllowNetwork] = useState(false);
  const [maxDownloads, setMaxDownloads] = useState(1);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const refreshArchives = async () => {
    const [scientific, operational] = await Promise.all([
      api<ObjectPage<CampaignSummary>>("/api/v1/discovery/campaigns?limit=20"),
      api<ObjectPage<ResearchCampaignJob>>("/api/v1/discovery/jobs?limit=40"),
    ]);
    setCampaigns(scientific.items); setJobs(operational.items);
    return { scientific: scientific.items, operational: operational.items };
  };

  useEffect(() => { void Promise.all([getAllObjects(), refreshArchives()]).then(async ([catalog, archive]) => {
    setObjects(catalog.items); setSelected(catalog.items.find((item) => item.catalog_disposition === "CANDIDATE") ?? catalog.items[0]);
    const active = archive.operational.find((item) => activeStates.has(item.status)) ?? archive.operational[0]; if (active) setJob(active);
    const reference = archive.scientific.find((item) => item.campaign_id === "stage10-kepler-reference-final") ?? archive.scientific[0];
    if (reference) setCampaign((await api<Scientific<CampaignDetail>>(`/api/v1/discovery/campaigns/${reference.campaign_id}`)).data);
  }).catch((reason) => setError(reason.message)); }, []);

  useEffect(() => { void api<Scientific<Plan>>(`/api/v1/discovery/plan?sample_size=${sampleSize}`).then((payload) => {
    setPlan(payload.data); if (!payload.data.downloads_enabled) setAllowNetwork(false);
  }).catch((reason) => setError(reason.message)); }, [sampleSize]);

  useEffect(() => { if (!job) { setEvents([]); return; } void api<ObjectPage<ResearchCampaignEvent>>(`/api/v1/discovery/jobs/${job.job_id}/events?limit=250`).then((page) => setEvents(page.items)).catch((reason) => setError(reason.message)); }, [job]);

  useEffect(() => { if (!job || !activeStates.has(job.status)) return; const timer = window.setTimeout(() => { void api<ResearchCampaignJob>(`/api/v1/discovery/jobs/${job.job_id}`).then(async (next) => {
    setJob(next); if (next.current_kic) setSelected(objects.find((item) => item.kic === next.current_kic) ?? selected);
    if (!activeStates.has(next.status)) { const archive = await refreshArchives(); if (next.status === "completed") { const detail = await api<Scientific<CampaignDetail>>(`/api/v1/discovery/campaigns/${next.campaign_id}`); setCampaign(detail.data); setCampaigns(archive.scientific); } }
  }).catch((reason) => setError(reason.message)); }, 650); return () => window.clearTimeout(timer); }, [job, objects, selected]);

  const counts = useMemo(() => objects.reduce<Record<string, number>>((result, object) => ({ ...result, [object.catalog_disposition]: (result[object.catalog_disposition] ?? 0) + 1 }), {}), [objects]);
  const progress = job ? Math.min(100, 100 * job.current / Math.max(job.total, 1)) : 0;
  const execute = async (action: "plan" | "start" | "cancel" | "resume") => { setBusy(action); setError(""); try {
    let next: ResearchCampaignJob;
    if (action === "plan") next = await api<ResearchCampaignJob>("/api/v1/discovery/campaigns", { method: "POST", body: JSON.stringify({ mission: "Kepler", sample_size: sampleSize, allow_network: allowNetwork, max_new_downloads: allowNetwork ? maxDownloads : 0 }) });
    else if (!job) return;
    else next = await api<ResearchCampaignJob>(`/api/v1/discovery/jobs/${job.job_id}/${action}`, { method: "POST", body: "{}" });
    setJob(next); await refreshArchives();
  } catch (reason) { setError(reason instanceof Error ? reason.message : "Campaign operation failed."); } finally { setBusy(""); } };

  const openJob = async (candidate: ResearchCampaignJob) => { setJob(candidate); if (candidate.status === "completed") { try { setCampaign((await api<Scientific<CampaignDetail>>(`/api/v1/discovery/campaigns/${candidate.campaign_id}`)).data); } catch { /* Operational record remains inspectable without a scientific artifact. */ } } };

  return <div className="page mission-control-page"><PageIntro eyebrow="Atlas mission control / persistent research" title="Command a real expedition." description="Plan bounded Kepler searches, review every scientific parameter, follow the active star and retain an auditable event trail across service restarts."><div className="catalog-chip"><Crosshair />{objects.filter((item) => item.ra_deg != null).length} positioned objects</div></PageIntro>
    {error && <div className="recoverable-error"><ErrorState message={error} /><button className="button secondary" onClick={() => { setError(""); void refreshArchives(); }}><RotateCcw />Recover state</button></div>}
    <section className="discover-layout">
      <Panel className="atlas-panel" eyebrow="Equatorial coordinates · J2000 catalog fields" title="Kepler stellar atlas" action={<span className="data-tag">Real catalog coordinates</span>}>
        {objects.length ? <StarAtlas objects={objects} selectedId={selected?.object_id} onSelect={setSelected} /> : <Loading label="Loading verified coordinates…" />}
      </Panel>
      <aside className="selection-console">
        <span className="eyebrow">{job?.current_kic === selected?.kic ? "Active research target" : "Selected catalog object"}</span><h2>{selected?.display_name ?? "Select a star"}</h2><p>{selected?.kepler_name ?? `KIC ${selected?.kic ?? "—"}`}</p>
        <div className="coordinate-lock"><span>Right ascension<strong>{selected?.ra_deg?.toFixed(5) ?? "Unavailable"}°</strong></span><span>Declination<strong>{selected?.dec_deg?.toFixed(5) ?? "Unavailable"}°</strong></span></div>
        {selected && <><span className={`disposition ${selected.catalog_disposition.toLowerCase().replace(" ", "-")}`}>{selected.catalog_disposition}</span><dl><div><dt>Orbital period</dt><dd>{formatNumber(selected.period_days, 5)} d</dd></div><div><dt>Signal cache</dt><dd>{selected.has_cached_observation ? "Available" : "Not materialized"}</dd></div><div><dt>Scientific partition</dt><dd>{selected.split}</dd></div></dl><a className="button secondary" href={`/observatory?object=${selected.object_id}`}><Telescope />Open in Observatory</a></>}
      </aside>
    </section>

    <section className="metric-strip"><Metric label="Accessible catalog" value={objects.length} unit="objects" /><Metric label="Cached plan targets" value={plan?.cached_targets ?? 0} /><Metric label="Candidates" value={counts.CANDIDATE ?? 0} tone="signal" /><Metric label="Operational records" value={jobs.length} /><Metric label="Frozen-test overlap" value={plan?.protected_host_overlap.length ?? 0} tone="good" /></section>

    <section className="mission-control-grid">
      <Panel className="campaign-configurator" eyebrow="Step 01 / bounded configuration" title="Compose expedition" action={<span className="data-tag"><LockKeyhole />Local worker · concurrency 1</span>}>
        <div className="mission-parameter-grid"><label>Mission<select value="Kepler" disabled><option>Kepler</option></select></label><label>Target volume<select value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))}><option value={1}>1 star</option><option value={2}>2 stars</option><option value={5}>5 stars</option><option value={10}>10 stars</option><option value={12}>12 stars</option></select></label><label>Selection strategy<input value="Unlabelled candidates · cache first" disabled /></label><label className="network-authority"><input type="checkbox" checked={allowNetwork} disabled={!plan?.downloads_enabled} onChange={(event) => setAllowNetwork(event.target.checked)} /><span>Authorize bounded MAST acquisition<small>{plan?.downloads_enabled ? "Deployment permits explicit downloads" : "Downloads disabled by deployment policy"}</small></span></label>{allowNetwork && <label>Maximum new downloads<select value={maxDownloads} onChange={(event) => setMaxDownloads(Number(event.target.value))}>{Array.from({ length: plan?.maximum_campaign_downloads ?? 1 }, (_, index) => <option key={index + 1} value={index + 1}>{index + 1}</option>)}</select></label>}</div>
        <div className="guardrail-cluster"><span><ShieldCheck />Unlabelled targets</span><span><ShieldCheck />Frozen hosts excluded</span><span><ShieldCheck />No neural confirmation</span><span><Database />{plan?.cached_targets ?? 0}/{plan?.sample_size ?? sampleSize} cached</span></div>
        <button className="button primary" disabled={!plan?.execution_enabled || !!busy || activeStates.has(job?.status ?? "")} onClick={() => void execute("plan")}><FileSearch />{busy === "plan" ? "Persisting plan…" : plan?.execution_enabled ? "Review and persist plan" : "Execution disabled in this deployment"}</button>
      </Panel>

      <Panel className="campaign-review" eyebrow="Step 02 / verified launch contract" title="Review before execution">
        {!job ? <div className="mission-empty"><Radio /><strong>No operational plan selected</strong><p>Configure a bounded expedition to materialize its immutable execution contract.</p></div> : <><div className="campaign-identity"><span>{job.status}</span><strong>{job.campaign_id}</strong><small>Job {job.job_id}</small></div><div className="review-contract"><span><small>Targets</small><strong>{job.total}</strong></span><span><small>Mission</small><strong>{job.mission}</strong></span><span><small>Network</small><strong>{job.allow_network ? `Authorized · ${job.max_new_downloads} max` : "Disabled"}</strong></span><span><small>Quarter / cadence</small><strong>Q{String(job.campaign_config.quarter)} · {String(job.campaign_config.cadence)}</strong></span><span><small>Period search</small><strong>{String(job.detector_config.minimum_period_days)}–{String(job.detector_config.maximum_period_days)} days</strong></span><span><small>Threshold</small><strong>SNR ≥ {String(job.detector_config.minimum_depth_snr)}</strong></span></div><div className="campaign-operation-actions">{job.status === "planned" && <button className="button primary" disabled={!!busy} onClick={() => void execute("start")}><Play />Start real campaign</button>}{activeStates.has(job.status) && <button className="button secondary danger-action" disabled={job.cancel_requested || !!busy} onClick={() => void execute("cancel")}><Square />{job.cancel_requested ? "Cancellation pending" : "Cancel safely"}</button>}{resumableStates.has(job.status) && <button className="button secondary" disabled={!!busy} onClick={() => void execute("resume")}><RotateCcw />Resume persisted campaign</button>}</div></>}
      </Panel>
    </section>

    {job && <section className="active-mission-grid">
      <Panel className="execution-console" eyebrow="Live operational state" title="Campaign telemetry" action={<span className={`campaign-state state-${job.status}`}><i />{job.status}</span>}>
        <div className="execution-readout"><div><span>Stage</span><strong>{job.stage.replaceAll("_", " ")}</strong><small>{job.message}</small></div><div><span>Active star</span><strong>{job.current_kic ? `KIC ${job.current_kic}` : "Awaiting target"}</strong><small>{job.current}/{job.total} target positions reached</small></div><div className="radial-progress" style={{ "--progress": `${progress * 3.6}deg` } as CSSProperties}><span><strong>{formatNumber(progress, 0)}%</strong><small>progress</small></span></div></div>
        <div className="stage-rail">{stageOrder.map((stage, index) => { const currentIndex = Math.max(stageOrder.indexOf(job.stage), job.status === "completed" ? stageOrder.length - 1 : 0); return <span key={stage} className={index < currentIndex ? "complete" : index === currentIndex ? "active" : ""}><i>{index < currentIndex ? <CheckCircle2 /> : <Activity />}</i><small>{stage}</small></span>; })}</div>
        <progress value={job.current} max={job.total} />
      </Panel>
      <Panel className="event-console" eyebrow="Append-only observability" title="Scientific event stream" action={<span className="data-tag">{events.length} events</span>}><div className="event-stream">{events.slice(-10).reverse().map((event) => <article key={event.event_id} className={`event-${event.level}`}><i>{event.level === "error" ? <AlertTriangle /> : event.event_type.includes("completed") ? <CheckCircle2 /> : <Radio />}</i><div><span>{event.stage}{event.kic ? ` · KIC ${event.kic}` : ""}</span><strong>{event.message}</strong><small>{formatDate(event.created_at_utc)}{event.elapsed_seconds != null ? ` · ${formatNumber(event.elapsed_seconds, 2)} s` : ""}</small></div></article>)}</div></Panel>
    </section>}

    <section className="two-column">
      <Panel eyebrow="Persisted scientific evidence" title="Campaign result">
        {campaign ? <><div className="campaign-hero"><span>Campaign</span><strong>{campaign.campaign.campaign_id}</strong><small>{formatDate(campaign.campaign.completed_at_utc)}</small></div><div className="campaign-stats"><Metric label="Investigated" value={campaign.campaign.investigated_stars} /><Metric label="Screening signals" value={campaign.campaign.screening_detections} /><Metric label="Catalog associated" value={campaign.campaign.catalog_associated_detections} /><Metric label="Inconclusive" value={campaign.campaign.unassociated_inconclusive_detections} tone="signal" /></div><div className="result-signal-strip">{campaign.targets.slice(0, 8).map((target) => <button key={target.kic} onClick={() => setSelected(objects.find((item) => item.kic === target.kic) ?? selected)}><span>KIC {target.kic}</span><strong>{formatNumber(target.period_days, 4)} d</strong><small>{target.status.replaceAll("_", " ")}</small></button>)}</div><div className="candidate-actions"><a className="button secondary" href={`/api/v1/discovery/campaigns/${campaign.campaign.campaign_id}/export`}><Database />Export campaign</a>{campaign.campaign.campaign_id === "stage10-kepler-reference-final" && <a className="button secondary" href="/candidates"><Telescope />Open scientific dossiers</a>}</div></> : <Loading label="Opening persisted campaign evidence…" />}
        <ScientificNotice>Screening detections are investigation triggers, not confirmed discoveries. This surface never upgrades a NASA disposition.</ScientificNotice>
      </Panel>
      <Panel eyebrow="Persistent operational archive" title="Previous missions" action={<History />}><div className="mission-history">{jobs.slice(0, 10).map((item) => <button key={item.job_id} className={job?.job_id === item.job_id ? "active" : ""} onClick={() => void openJob(item)}><i className={`status-${item.status}`} /><div><span>{item.status} · {item.total} targets</span><strong>{item.campaign_id}</strong><small>{formatDate(item.updated_at_utc)}{item.current_kic ? ` · KIC ${item.current_kic}` : ""}</small></div></button>)}</div></Panel>
    </section>

    <Panel eyebrow="Secondary audit view" title="Scientific campaign ledger"><AuditTable headers={["Campaign", "Status", "Completed", "Stars", "Signals", "Unassociated", "Confirmed"]} rows={campaigns.map((item) => [item.campaign_id, item.status, formatDate(item.completed_at_utc), item.investigated_stars, item.screening_detections, item.unassociated_inconclusive_detections, item.confirmed_discoveries])} /></Panel>
  </div>;
}
