"use client";

import { Crosshair, Play, ShieldCheck, Telescope } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { StarAtlas } from "@/components/star-atlas";
import { AuditTable, ErrorState, Loading, Metric, PageIntro, Panel, ScientificNotice } from "@/components/ui";
import { api, getAllObjects } from "@/lib/api";
import { formatDate, formatNumber } from "@/lib/format";
import type { CampaignDetail, CampaignSummary, ObjectPage, ObjectSummary, Scientific } from "@/lib/types";

interface Plan { sample_size: number; cached_targets: number; protected_host_overlap: number[]; network_enabled: boolean; execution_enabled: boolean; targets: Array<{ kic: number; label: number; selection_reference: string }> }
interface Job { job_id: string; status: string; stage: string; completed_targets: number; total_targets: number; current_target: string | null; campaign_id: string | null; error: string | null }

export default function DiscoverPage() {
  const [objects, setObjects] = useState<ObjectSummary[]>([]);
  const [selected, setSelected] = useState<ObjectSummary | null>(null);
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([]);
  const [campaign, setCampaign] = useState<CampaignDetail | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [sampleSize, setSampleSize] = useState(12);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");

  useEffect(() => { void Promise.all([getAllObjects(), api<ObjectPage<CampaignSummary>>("/api/v1/discovery/campaigns?limit=12")]).then(async ([catalog, page]) => {
    setObjects(catalog.items); setSelected(catalog.items.find((item) => item.catalog_disposition === "CANDIDATE") ?? catalog.items[0]); setCampaigns(page.items);
    if (page.items[0]) setCampaign((await api<Scientific<CampaignDetail>>(`/api/v1/discovery/campaigns/${page.items[0].campaign_id}`)).data);
  }).catch((reason) => setError(reason.message)); }, []);

  useEffect(() => { void api<Scientific<Plan>>(`/api/v1/discovery/plan?sample_size=${sampleSize}`).then((payload) => setPlan(payload.data)).catch((reason) => setError(reason.message)); }, [sampleSize]);

  useEffect(() => { if (!job || !["queued", "running"].includes(job.status)) return; const timer = window.setTimeout(() => { void api<Job>(`/api/v1/discovery/jobs/${job.job_id}`).then(async (next) => { setJob(next); if (next.status === "completed" && next.campaign_id) { const detail = await api<Scientific<CampaignDetail>>(`/api/v1/discovery/campaigns/${next.campaign_id}`); setCampaign(detail.data); } }).catch((reason) => setError(reason.message)); }, 700); return () => window.clearTimeout(timer); }, [job]);

  const counts = useMemo(() => objects.reduce<Record<string, number>>((result, object) => ({ ...result, [object.catalog_disposition]: (result[object.catalog_disposition] ?? 0) + 1 }), {}), [objects]);
  const launch = async () => { setError(""); try { setJob(await api<Job>("/api/v1/discovery/jobs", { method: "POST", body: JSON.stringify({ sample_size: sampleSize }) })); } catch (reason) { setError(reason instanceof Error ? reason.message : "Campaign could not start."); } };

  return <div className="page"><PageIntro eyebrow="Mission control / equatorial atlas" title="Discover the observable field." description="Navigate real catalog coordinates, select a Kepler host, and inspect the evidence produced by bounded ASTRAL expeditions."><div className="catalog-chip"><Crosshair />{objects.filter((item) => item.ra_deg != null).length} positioned objects</div></PageIntro>
    {error && <ErrorState message={error} />}
    <section className="discover-layout">
      <Panel className="atlas-panel" eyebrow="Equatorial coordinates · J2000 catalog fields" title="Kepler stellar atlas" action={<span className="data-tag">RA / Dec are catalog values</span>}>
        {objects.length ? <StarAtlas objects={objects} selectedId={selected?.object_id} onSelect={setSelected} /> : <Loading label="Loading verified coordinates…" />}
      </Panel>
      <aside className="selection-console">
        <span className="eyebrow">Selected catalog object</span><h2>{selected?.display_name ?? "Select a star"}</h2><p>{selected?.kepler_name ?? `KIC ${selected?.kic ?? "—"}`}</p>
        <div className="coordinate-lock"><span>Right ascension<strong>{selected?.ra_deg?.toFixed(5) ?? "Unavailable"}°</strong></span><span>Declination<strong>{selected?.dec_deg?.toFixed(5) ?? "Unavailable"}°</strong></span></div>
        {selected && <><span className={`disposition ${selected.catalog_disposition.toLowerCase().replace(" ", "-")}`}>{selected.catalog_disposition}</span><dl><div><dt>Orbital period</dt><dd>{formatNumber(selected.period_days, 5)} d</dd></div><div><dt>Signal cache</dt><dd>{selected.has_cached_observation ? "Available" : "Not materialized"}</dd></div><div><dt>Scientific partition</dt><dd>{selected.split}</dd></div></dl><a className="button secondary" href={`/observatory?object=${selected.object_id}`}><Telescope />Open in Observatory</a></>}
      </aside>
    </section>

    <section className="metric-strip"><Metric label="Accessible catalog" value={objects.length} unit="objects" /><Metric label="Confirmed" value={counts.CONFIRMED ?? 0} /><Metric label="Candidates" value={counts.CANDIDATE ?? 0} tone="signal" /><Metric label="False positives" value={counts["FALSE POSITIVE"] ?? 0} /><Metric label="Frozen-test overlap" value={plan?.protected_host_overlap.length ?? 0} tone="good" /></section>

    <section className="expedition-grid">
      <Panel eyebrow="Protected, cache-only execution" title="Configure expedition">
        <div className="control-grid"><label>Target volume<select value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))}><option value={5}>5 stars</option><option value={10}>10 stars</option><option value={12}>12 stars</option></select></label><div className="contract-readout"><span><ShieldCheck />Network disabled</span><span><ShieldCheck />Candidates unlabeled</span><span><ShieldCheck />Frozen test excluded</span></div></div>
        {plan && <div className="plan-summary"><strong>{plan.cached_targets}/{plan.sample_size}</strong><span>targets available in local cache</span></div>}
        <button className="button primary" disabled={!plan?.execution_enabled || job?.status === "running"} onClick={launch}><Play />{plan?.execution_enabled ? "Launch cache-only expedition" : "Execution disabled in this deployment"}</button>
        {job && <div className="job-progress"><div><span>{job.stage}</span><strong>{job.completed_targets}/{job.total_targets}</strong></div><progress value={job.completed_targets} max={job.total_targets} /><small>{job.current_target ?? job.status}{job.error ? ` · ${job.error}` : ""}</small></div>}
      </Panel>
      <Panel eyebrow="Persisted evidence" title="Latest expedition">
        {campaign ? <><div className="campaign-hero"><span>Campaign</span><strong>{campaign.campaign.campaign_id}</strong><small>{formatDate(campaign.campaign.completed_at_utc)}</small></div><div className="campaign-stats"><Metric label="Investigated" value={campaign.campaign.investigated_stars} /><Metric label="Screening signals" value={campaign.campaign.screening_detections} /><Metric label="Catalog associated" value={campaign.campaign.catalog_associated_detections} /><Metric label="Inconclusive" value={campaign.campaign.unassociated_inconclusive_detections} tone="signal" /></div></> : <Loading />}
        <ScientificNotice>Screening detections are not confirmed discoveries. The reference campaign reports {campaign?.campaign.confirmed_discoveries ?? 0} confirmed discoveries.</ScientificNotice>
      </Panel>
    </section>

    <Panel eyebrow="Secondary audit view" title="Expedition ledger">
      <AuditTable headers={["Campaign", "Completed", "Stars", "Signals", "Unassociated", "Confirmed"]} rows={campaigns.map((item) => [item.campaign_id, formatDate(item.completed_at_utc), item.investigated_stars, item.screening_detections, item.unassociated_inconclusive_detections, item.confirmed_discoveries])} />
    </Panel>
  </div>;
}
