"use client";

import { BrainCircuit, Cpu, Eye, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { NeuralGraph } from "@/components/neural-graph";
import { ScientificPlot } from "@/components/scientific-plot";
import { AuditTable, ErrorState, Loading, Metric, PageIntro, Panel, ScientificNotice } from "@/components/ui";
import { api } from "@/lib/api";
import { formatNumber } from "@/lib/format";
import type { ExplanationDetail, ExplanationIndex, Inference, ModelEntry, NeuralVisualization, Scientific } from "@/lib/types";

export default function NeuralPage() {
  const [models, setModels] = useState<ModelEntry[]>([]); const [modelId, setModelId] = useState<"ann-01" | "ann-02">("ann-01");
  const [index, setIndex] = useState<ExplanationIndex | null>(null); const [objectId, setObjectId] = useState("K00753.01");
  const [states, setStates] = useState<Partial<Record<"ann-01" | "ann-02", NeuralVisualization>>>({});
  const [inferences, setInferences] = useState<Partial<Record<"ann-01" | "ann-02", Inference>>>({});
  const [explanation, setExplanation] = useState<ExplanationDetail | null>(null); const [error, setError] = useState(""); const [loading, setLoading] = useState(true);
  useEffect(() => { void Promise.all([api<Scientific<{ models: ModelEntry[] }>>("/api/v1/models"), api<Scientific<ExplanationIndex>>("/api/v1/explanations")]).then(([catalog, explanations]) => { setModels(catalog.data.models); setIndex(explanations.data); setObjectId(explanations.data.objects.find((item) => item.catalog_disposition === "CANDIDATE")?.object_id ?? explanations.data.objects[0]?.object_id ?? "K00753.01"); }).catch((reason) => setError(reason.message)); }, []);
  useEffect(() => { if (!objectId) return; setLoading(true); setError(""); void Promise.all([
    api<Inference>("/api/v1/inference/ann-01", { method: "POST", body: JSON.stringify({ object_id: objectId }) }), api<Inference>("/api/v1/inference/ann-02", { method: "POST", body: JSON.stringify({ object_id: objectId }) }),
    api<Scientific<NeuralVisualization>>(`/api/v1/models/ann-01/visualization?object_id=${encodeURIComponent(objectId)}`), api<Scientific<NeuralVisualization>>(`/api/v1/models/ann-02/visualization?object_id=${encodeURIComponent(objectId)}`),
    api<Scientific<ExplanationDetail>>(`/api/v1/explanations/${encodeURIComponent(objectId)}?include_saliency=true&saliency_points=500`),
  ]).then(([ann01, ann02, state01, state02, detail]) => { setInferences({ "ann-01": ann01, "ann-02": ann02 }); setStates({ "ann-01": state01.data, "ann-02": state02.data }); setExplanation(detail.data); }).catch((reason) => setError(reason.message)).finally(() => setLoading(false)); }, [objectId]);
  const model = models.find((item) => item.id === modelId); const state = states[modelId]; const inference = inferences[modelId];
  const weights = useMemo(() => state?.parameters.filter((tensor) => tensor.role === "weight").flatMap((tensor) => flatten(tensor.values)) ?? [], [state]);
  const weightPlot = [{ x: weights, type: "histogram", nbinsx: 36, marker: { color: "#70DBF7", line: { color: "#B5C4D0", width: .4 } }, hovertemplate: "Weight %{x:.5f}<br>Count %{y}<extra></extra>" }];
  const featureMap = state?.activations.find((tensor) => tensor.role === "feature_map");
  const featurePlot = featureMap ? [{ z: featureMap.values, type: "heatmap", colorscale: [[0, "#070D17"], [.35, "#26394D"], [.7, "#70DBF7"], [1, "#EFB681"]], colorbar: { title: "activation", thickness: 10 }, hovertemplate: "Channel %{y}<br>Bin %{x}<br>Activation %{z:.5f}<extra></extra>" }] : [];
  const saliencyEntry = explanation?.ann02.saliency ? Object.entries(explanation.ann02.saliency)[0] : null;
  const saliencyPlot = saliencyEntry ? [{ x: saliencyEntry[1].indices, y: saliencyEntry[1].values, type: "scatter", mode: "lines", line: { color: "#EFB681", width: 1.5 }, fill: "tozeroy", fillcolor: "rgba(239,182,129,.10)", hovertemplate: "Signal bin %{x}<br>Sensitivity %{y:.6f}<extra></extra>" }] : [];
  const contributionPlot = explanation?.ann01 ? [{ y: explanation.ann01.features.map((item) => item.feature), x: explanation.ann01.features.map((item) => item.local_contribution), type: "bar", orientation: "h", marker: { color: explanation.ann01.features.map((item) => item.local_contribution >= 0 ? "#70DBF7" : "#BFA7EF") }, hovertemplate: "%{y}<br>Local contribution %{x:.6f}<extra></extra>" }] : [];

  return <div className="page"><PageIntro eyebrow="Neural observatory / real retained state" title="Look inside the intelligence." description="Trace the architecture, parameters, activations, convolutional feature maps, and model-faithful explanations for one real astronomical record."><label className="object-selector"><span>Explanation target</span><select value={objectId} onChange={(event) => setObjectId(event.target.value)}>{index?.objects.map((object) => <option key={object.object_id} value={object.object_id}>{object.object_id} · {object.catalog_disposition}</option>)}</select></label></PageIntro>
    {error && <ErrorState message={error} />}
    <div className="model-switcher">{models.map((item) => <button key={item.id} className={modelId === item.id ? "active" : ""} onClick={() => setModelId(item.id)}><span>{item.id === "ann-01" ? <BrainCircuit /> : <Cpu />}</span><div><strong>{item.name}</strong><small>{item.framework} · {item.trainable_parameters.toLocaleString()} parameters</small></div></button>)}</div>
    {loading || !model || !state || !inference ? <Loading label="Retaining real neural state…" /> : <>
      <section className="metric-strip"><Metric label="Model score" value={inference.score} /><Metric label="Decision threshold" value={inference.threshold} /><Metric label="Prediction" value={inference.predicted_class} tone="signal" /><Metric label="Catalog class" value={inference.original_disposition} /><Metric label="Parameters" value={model.trainable_parameters} /></section>
      <Panel className="neural-panel" eyebrow="Interactive computation map" title={`${model.name} · ${objectId}`} action={<span className="data-tag"><Eye /> Real checkpoint state</span>}><NeuralGraph model={model} state={state} /><p className="caption">Hover ANN-01 neurons to inspect retained activations. Connection color and intensity encode the sign and magnitude of actual checkpoint weights. ANN-02 modules and map dimensions come from its PyTorch graph and current forward pass.</p></Panel>
      <section className="two-column"><Panel eyebrow="Checkpoint parameter field" title="Weight distribution"><ScientificPlot title={`${model.name} weight distribution`} data={weightPlot} layout={{ height: 340, xaxis: { title: "Parameter value" }, yaxis: { title: "Count" }, bargap: .05 }} /></Panel><Panel eyebrow={modelId === "ann-02" ? "Convolutional response" : "Local explanation"} title={modelId === "ann-02" ? `${featureMap?.name ?? "Feature map"}` : "ANN-01 feature contribution"}>{modelId === "ann-02" ? <ScientificPlot title="Real ANN-02 feature map" data={featurePlot} layout={{ height: 340, xaxis: { title: "Signal bin" }, yaxis: { title: "Feature channel" } }} /> : <ScientificPlot title="ANN-01 local contributions" data={contributionPlot} layout={{ height: 340, xaxis: { title: "Signed local contribution" }, margin: { l: 120, r: 20, t: 20, b: 48 } }} />}</Panel></section>
      <section className="two-column"><Panel eyebrow="Explainable AI" title="Signal sensitivity"><ScientificPlot title="ANN-02 input sensitivity" data={saliencyPlot} layout={{ height: 330, xaxis: { title: "Input bin" }, yaxis: { title: "Gradient × input sensitivity" } }} /><p className="caption">Sensitivity identifies input regions that influenced this model score; it does not identify physical causality.</p></Panel><Panel eyebrow="Occlusion analysis" title="Influential signal regions"><div className="region-list">{explanation?.ann02.regions.slice(0, 7).map((region, indexValue) => <div key={`${region.view}-${indexValue}`}><span>{region.view}</span><strong>{formatNumber(region.start_coordinate, 4)} — {formatNumber(region.end_coordinate, 4)}</strong><small>Score Δ {formatNumber(region.score_delta, 6)} · observed {formatNumber(region.observed_fraction * 100, 1)}%</small></div>)}</div></Panel></section>
      <ScientificNotice><ShieldCheck /> All tensors are read from persisted checkpoints or a real forward pass for {objectId}. Empty or unavailable states are not synthesized. Explanations describe model sensitivity, not astrophysical causality.</ScientificNotice>
      <Panel eyebrow="Secondary technical view" title="Tensor register"><AuditTable headers={["Tensor", "Role", "Shape", "Minimum", "Maximum", "Mean", "Std. deviation"]} rows={[...state.parameters, ...state.activations].map((tensor) => [tensor.name, tensor.role, tensor.shape.join(" × "), formatNumber(tensor.minimum, 6), formatNumber(tensor.maximum, 6), formatNumber(tensor.mean, 6), formatNumber(tensor.standard_deviation, 6)])} /></Panel>
    </>}
  </div>;
}

function flatten(values: unknown[]): number[] { return values.flat(Infinity).filter((value): value is number => typeof value === "number"); }
