"use client";

import { useMemo } from "react";
import type { ModelEntry, NeuralVisualization } from "@/lib/types";

type Layer = { index?: number; name: string; units: number; activation?: string };

export function NeuralGraph({ model, state }: { model: ModelEntry; state: NeuralVisualization }) {
  if (model.id === "ann-02") return <Ann02Graph model={model} state={state} />;
  const layers = model.architecture as Layer[];
  const weightLayers = state.parameters.filter((tensor) => tensor.role === "weight");
  const activations = state.activations;
  const width = 920; const height = 430; const padX = 92;
  const xFor = (index: number) => padX + index * ((width - padX * 2) / (layers.length - 1));
  const yFor = (index: number, count: number) => height / 2 + (index - (count - 1) / 2) * Math.min(46, 310 / Math.max(1, count - 1));
  const maxWeight = Math.max(...weightLayers.flatMap((tensor) => flatten(tensor.values).map(Math.abs)), .001);
  return <div className="neural-canvas"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ANN-01 real weighted neural architecture">
    <defs><filter id="node-glow"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
    {layers.slice(0, -1).flatMap((layer, layerIndex) => {
      const next = layers[layerIndex + 1]; const weights = weightLayers[layerIndex]?.values as number[][] | undefined;
      return Array.from({ length: layer.units }).flatMap((_, source) => Array.from({ length: next.units }).map((__, target) => {
        const value = weights?.[source]?.[target] ?? 0; const opacity = .08 + Math.abs(value) / maxWeight * .62;
        return <line key={`${layerIndex}-${source}-${target}`} x1={xFor(layerIndex)} y1={yFor(source, layer.units)} x2={xFor(layerIndex + 1)} y2={yFor(target, next.units)} stroke={value >= 0 ? "#70DBF7" : "#BFA7EF"} strokeOpacity={opacity} strokeWidth={.6 + Math.abs(value) / maxWeight * 1.4} />;
      }));
    })}
    {layers.flatMap((layer, layerIndex) => {
      const values = flatten(activations[layerIndex]?.values ?? []); const max = Math.max(...values.map(Math.abs), 1);
      return Array.from({ length: layer.units }).map((_, neuron) => { const value = values[neuron] ?? 0; const intensity = Math.min(1, Math.abs(value) / max); return <g key={`${layer.name}-${neuron}`}><circle cx={xFor(layerIndex)} cy={yFor(neuron, layer.units)} r={layerIndex === layers.length - 1 ? 13 : 10} fill={value >= 0 ? `rgba(112,219,247,${.18 + intensity * .72})` : `rgba(191,167,239,${.18 + intensity * .72})`} stroke="#B5C4D0" strokeOpacity=".6" filter={intensity > .5 ? "url(#node-glow)" : undefined}><title>{layer.name} neuron {neuron + 1}: activation {value.toFixed(6)}</title></circle></g>; });
    })}
    {layers.map((layer, index) => <g key={`${layer.name}-label`}><text x={xFor(index)} y="28" textAnchor="middle" fill="#F0F5F8" fontSize="14">{layer.name.replaceAll("_", " ")}</text><text x={xFor(index)} y="48" textAnchor="middle" fill="#8295A5" fontSize="11">{layer.units} units · {layer.activation}</text></g>)}
  </svg><div className="neural-legend"><span><i className="positive" />Positive weight</span><span><i className="negative" />Negative weight</span><span>Line intensity = |weight|</span><span>Node intensity = activation</span></div></div>;
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
