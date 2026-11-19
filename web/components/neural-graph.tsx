"use client";

import { Activity, Eye, Network, RotateCcw } from "lucide-react";
import type { CSSProperties } from "react";
import { useMemo, useState } from "react";
import type { ModelEntry, NeuralVisualization } from "@/lib/types";

type Layer = { index?: number; name: string; units: number; activation?: string };
type SignFilter = "all" | "positive" | "negative";
type NodeSelection = { layerIndex: number; neuron: number };
const INPUT_FEATURES = ["Period", "Duration", "Depth", "Radius", "Model SNR", "Equilibrium T"];

export function NeuralGraph({ model, state }: { model: ModelEntry; state: NeuralVisualization }) {
  return model.id === "ann-02" ? <Ann02Graph model={model} state={state} /> : <Ann01Graph model={model} state={state} />;
}

function Ann01Graph({ model, state }: { model: ModelEntry; state: NeuralVisualization }) {
  const layers = model.architecture as Layer[];
  const weightLayers = state.parameters.filter((tensor) => tensor.role === "weight");
  const biasLayers = state.parameters.filter((tensor) => tensor.role === "bias");
  const activations = state.activations;
  const [selection, setSelection] = useState<NodeSelection>({ layerIndex: layers.length - 1, neuron: 0 });
  const [signFilter, setSignFilter] = useState<SignFilter>("all");
  const width = 920; const height = 500; const padX = 90; const graphTop = 112; const graphBottom = 438;
  const xFor = (index: number) => padX + index * ((width - padX * 2) / (layers.length - 1));
  const yFor = (index: number, count: number) => graphTop + (index + 1) * ((graphBottom - graphTop) / (count + 1));
  const allWeights = weightLayers.flatMap((tensor) => flatten(tensor.values));
  const maxWeight = Math.max(...allWeights.map(Math.abs), .001);
  const selectedLayer = layers[selection.layerIndex];
  const selectedActivation = flatten(activations[selection.layerIndex]?.values ?? [])[selection.neuron] ?? 0;
  const selectedBiases = selection.layerIndex ? flatten(biasLayers[selection.layerIndex - 1]?.values ?? []) : [];
  const selectedBias = selectedBiases[selection.neuron];
  const incoming = selection.layerIndex ? ((weightLayers[selection.layerIndex - 1]?.values as number[][] | undefined) ?? []).map((row, source) => ({ source, value: row[selection.neuron] ?? 0 })) : [];
  const outgoingMatrix = weightLayers[selection.layerIndex]?.values as number[][] | undefined;
  const outgoing = outgoingMatrix?.[selection.neuron]?.map((value, target) => ({ target, value })) ?? [];
  const strongest = [...incoming].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))[0];
  const visibleWeightCount = allWeights.filter((value) => signFilter === "all" || (signFilter === "positive" ? value >= 0 : value < 0)).length;
  const nodeLabel = selection.layerIndex === 0 ? INPUT_FEATURES[selection.neuron] : `${selectedLayer.name.replaceAll("_", " ")} · N${String(selection.neuron + 1).padStart(2, "0")}`;

  return <div className="ann01-workbench">
    <div className="neural-toolbar">
      <div className="neural-topology"><Network /><span><small>Live topology</small><strong>{layers.map((layer) => layer.units).join(" → ")}</strong></span><span><small>Checkpoint</small><strong>{model.trainable_parameters} parameters</strong></span></div>
      <div className="weight-filter" aria-label="Connection filter"><span>Connections</span>{(["all", "positive", "negative"] as SignFilter[]).map((filter) => <button key={filter} className={signFilter === filter ? "active" : ""} onClick={() => setSignFilter(filter)}>{filter}</button>)}<button aria-label="Reset neural selection" onClick={() => { setSelection({ layerIndex: layers.length - 1, neuron: 0 }); setSignFilter("all"); }}><RotateCcw /></button></div>
    </div>
    <div className="ann01-stage">
      <div className="neural-canvas"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ANN-01 interactive real weighted neural architecture">
        <defs><filter id="node-glow"><feGaussianBlur stdDeviation="5" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter><linearGradient id="layer-surface" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#12283d" stopOpacity=".66" /><stop offset="1" stopColor="#07111e" stopOpacity=".15" /></linearGradient></defs>
        <g className="neural-grid" aria-hidden="true">{Array.from({ length: 11 }).map((_, index) => <line key={`grid-${index}`} x1="24" y1={70 + index * 36} x2={width - 24} y2={70 + index * 36} />)}</g>
        {layers.map((layer, index) => <g key={`${layer.name}-surface`} className="layer-surface"><rect x={xFor(index) - 58} y="72" width="116" height="390" rx="20" /><text x={xFor(index)} y="34" textAnchor="middle">{layer.name.replaceAll("_", " ")}</text><text className="layer-meta" x={xFor(index)} y="54" textAnchor="middle">{layer.units} neurons · {layer.activation}</text></g>)}
        <g className="neural-connections">{layers.slice(0, -1).flatMap((layer, layerIndex) => {
          const next = layers[layerIndex + 1]; const weights = weightLayers[layerIndex]?.values as number[][] | undefined;
          return Array.from({ length: layer.units }).flatMap((_, source) => Array.from({ length: next.units }).map((__, target) => {
            const value = weights?.[source]?.[target] ?? 0; const magnitude = Math.abs(value) / maxWeight;
            const connected = (selection.layerIndex === layerIndex && selection.neuron === source) || (selection.layerIndex === layerIndex + 1 && selection.neuron === target);
            const signVisible = signFilter === "all" || (signFilter === "positive" ? value >= 0 : value < 0);
            const opacity = signVisible ? (connected ? .28 + magnitude * .72 : .025 + magnitude * .19) : .012;
            const x1 = xFor(layerIndex) + 13; const x2 = xFor(layerIndex + 1) - 13; const y1 = yFor(source, layer.units); const y2 = yFor(target, next.units);
            return <path key={`${layerIndex}-${source}-${target}`} d={`M ${x1} ${y1} C ${x1 + 72} ${y1}, ${x2 - 72} ${y2}, ${x2} ${y2}`} fill="none" stroke={value >= 0 ? "#75E6FF" : "#AD99FF"} strokeOpacity={opacity} strokeWidth={connected ? .9 + magnitude * 2.6 : .45 + magnitude * 1.1}><title>{layer.name} N{source + 1} → {next.name} N{target + 1}: weight {value.toFixed(6)}</title></path>;
          }));
        })}</g>
        <g className="neural-nodes">{layers.flatMap((layer, layerIndex) => {
          const values = flatten(activations[layerIndex]?.values ?? []); const max = Math.max(...values.map(Math.abs), 1);
          return Array.from({ length: layer.units }).map((_, neuron) => {
            const value = values[neuron] ?? 0; const intensity = Math.min(1, Math.abs(value) / max); const selected = selection.layerIndex === layerIndex && selection.neuron === neuron;
            const circumference = 2 * Math.PI * 15; const dash = Math.max(2, intensity * circumference);
            return <g key={`${layer.name}-${neuron}`} className={selected ? "selected" : ""} role="button" tabIndex={0} aria-label={`${layer.name} neuron ${neuron + 1}, activation ${value.toFixed(6)}`} onClick={() => setSelection({ layerIndex, neuron })} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelection({ layerIndex, neuron }); }}>
              <circle className="node-target" cx={xFor(layerIndex)} cy={yFor(neuron, layer.units)} r="18" />
              <circle className="node-activation" cx={xFor(layerIndex)} cy={yFor(neuron, layer.units)} r="14" strokeDasharray={`${dash} ${circumference - dash}`} transform={`rotate(-90 ${xFor(layerIndex)} ${yFor(neuron, layer.units)})`} filter={selected || intensity > .65 ? "url(#node-glow)" : undefined} />
              <circle className="node-core" cx={xFor(layerIndex)} cy={yFor(neuron, layer.units)} r={selected ? 7 : 5.5} />
              {layerIndex === 0 && <text className="input-label" x={xFor(layerIndex) - 25} y={yFor(neuron, layer.units) + 4} textAnchor="end">{INPUT_FEATURES[neuron]}</text>}
              {selected && <text className="node-value" x={xFor(layerIndex) + 24} y={yFor(neuron, layer.units) + 4}>{value.toFixed(4)}</text>}
            </g>;
          });
        })}</g>
      </svg></div>
      <aside className="neuron-inspector" aria-live="polite">
        <div className="inspector-heading"><span><Eye /> Selected state</span><small>Real forward pass</small></div>
        <strong>{nodeLabel}</strong><p>{selectedLayer.activation} activation</p>
        <div className="activation-orb" style={{ "--activation": Math.min(1, Math.abs(selectedActivation)) } as CSSProperties}><span>{selectedActivation.toFixed(6)}</span><small>activation</small></div>
        <dl><div><dt>Bias</dt><dd>{selectedBias == null ? "Not applicable" : selectedBias.toFixed(6)}</dd></div><div><dt>Incoming</dt><dd>{incoming.length} weights</dd></div><div><dt>Outgoing</dt><dd>{outgoing.length} weights</dd></div>{strongest && <div><dt>Strongest input</dt><dd>N{strongest.source + 1} · {strongest.value.toFixed(5)}</dd></div>}</dl>
        <div className="inspector-foot"><Activity /><span>{visibleWeightCount} of {allWeights.length} connections visible</span></div>
      </aside>
    </div>
    <div className="neural-legend"><span><i className="positive" />Positive weight</span><span><i className="negative" />Negative weight</span><span>Curve intensity = |weight|</span><span>Node ring = retained activation</span></div>
  </div>;
}

function Ann02Graph({ model, state }: { model: ModelEntry; state: NeuralVisualization }) {
  const architecture = model.architecture as { modules?: Array<{ name: string; type: string; parameters: number }> };
  const modules = (architecture.modules ?? []).filter((module) => module.parameters > 0 || module.type.includes("Pool") || module.type === "ReLU");
  const grouped = useMemo(() => ({ global: modules.filter((m) => m.name.startsWith("global")), local: modules.filter((m) => m.name.startsWith("local")), fusion: modules.filter((m) => !m.name.startsWith("global") && !m.name.startsWith("local")) }), [modules]);
  return <div className="cnn-graph"><div className="cnn-input"><strong>Kepler Q3</strong><span>Flux + availability mask</span></div><div className="branch-stack"><Branch label="Global view · 2 × 2001" modules={grouped.global} /><Branch label="Local transit · 2 × 201" modules={grouped.local} /></div><div className="fusion-node"><span>Late fusion</span>{grouped.fusion.map((module) => <b key={module.name}>{module.type}<small>{module.parameters} params</small></b>)}</div><div className="cnn-output"><strong>1 logit</strong><span>Sigmoid score</span></div><div className="map-readout">{state.activations.map((tensor) => <span key={tensor.name}><i />{tensor.name}<small>{tensor.shape.join(" × ")} real activations</small></span>)}</div></div>;
}

function Branch({ label, modules }: { label: string; modules: Array<{ name: string; type: string; parameters: number }> }) {
  return <div className="cnn-branch"><strong>{label}</strong><div>{modules.map((module) => <span key={module.name}>{module.type}<small>{module.parameters ? `${module.parameters} p` : module.name.split(".").at(-1)}</small></span>)}</div></div>;
}

function flatten(values: unknown[]): number[] { return values.flat(Infinity).filter((value): value is number => typeof value === "number"); }
