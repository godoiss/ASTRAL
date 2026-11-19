import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { NeuralGraph } from "@/components/neural-graph";
import { OrbitalSystem } from "@/components/orbital-system";
import { computeAtlasBounds, projectCatalogCoordinate } from "@/lib/atlas";
import { formatNumber } from "@/lib/format";
import { mergeExperimentTimeline } from "@/lib/research";
import type { ExperimentSummary, ModelEntry, NeuralVisualization, ObjectDetail, ObjectSummary } from "@/lib/types";

describe("scientific visualization contracts", () => {
  it("labels orbital geometry as illustrative while preserving catalog values", () => {
    const object = {
      display_name: "KOI-753.01", object_id: "K00753.01", kic: 10811496, kepler_name: null,
      disposition: "CANDIDATE", period_days: 19.89913995, epoch_bkjd: 134.45, epoch_bjd: 2454967.45,
      duration_hours: 3.2, depth_ppm: 500, planet_radius_earth: 2.1, equilibrium_temperature_k: 600,
      stellar_temperature_k: 5500, stellar_radius_solar: 1, stellar_mass_solar: 1, stellar_logg: 4.3,
      semi_major_axis_au: .14, insolation_earth: 30, kepler_magnitude: 14, split: "candidate", class_label: -1,
      has_cached_observation: true, ra_deg: 297.00482, dec_deg: 48.134129,
    } satisfies ObjectDetail;
    render(<OrbitalSystem object={object} />);
    expect(screen.getByText(/Illustrative orbital projection/)).toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "STRONG" && element.textContent === "0.14 AU")).toBeInTheDocument();
  });

  it("renders ANN-01 from real architecture and tensor state", () => {
    const model = {
      id: "ann-01", name: "ANN-01", framework: "NumPy", trainable_parameters: 4,
      experiment_id: "real", served_checkpoint: "real", definitive_scientific_validation: false,
      architecture: [{ name: "input", units: 1, activation: "identity" }, { name: "output", units: 1, activation: "sigmoid" }],
      shared_validation_metrics: { accuracy: .8, balanced_accuracy: .8, precision: .8, recall: .8, f1: .8, roc_auc: .8, pr_auc: .8, binary_cross_entropy: .4, confusion_matrix: [[1,0],[0,1]], records: 2, positive_prevalence: .5, threshold: .5 },
    } satisfies ModelEntry;
    const tensor = (name: string, role: "weight" | "activation", values: unknown[]) => ({ name, role, values, shape: [1], minimum: 0, maximum: 1, mean: .5, standard_deviation: .1 });
    const state = { model_id: "ann-01", object_id: "K00753.01", architecture: model.architecture, parameters: [tensor("layer_1.weights", "weight", [[.4]])], activations: [tensor("input.activation", "activation", [.2]), tensor("output.activation", "activation", [.7])], notes: [] } satisfies NeuralVisualization;
    render(<NeuralGraph model={model} state={state} />);
    expect(screen.getByLabelText("ANN-01 interactive real weighted neural architecture")).toBeInTheDocument();
    expect(screen.getByText("Positive weight")).toBeInTheDocument();
  });

  it("formats unavailable scientific values without inventing a number", () => {
    expect(formatNumber(null)).toBe("Unavailable");
  });

  it("fits real catalog coordinates to the Kepler field while preserving RA orientation", () => {
    const object = (object_id: string, ra_deg: number, dec_deg: number): ObjectSummary => ({ object_id, display_name: object_id, kic: 1, kepler_name: null, catalog_disposition: "CONFIRMED", split: "train", period_days: 1, has_cached_observation: true, ra_deg, dec_deg });
    const objects = [object("west", 280, 37), object("east", 301, 52)];
    const bounds = computeAtlasBounds(objects);
    const west = projectCatalogCoordinate(280, 37, bounds, 900, 600);
    const east = projectCatalogCoordinate(301, 52, bounds, 900, 600);
    expect(west.x).toBeGreaterThan(east.x);
    expect(west.y).toBeGreaterThan(east.y);
    expect(Math.abs(west.x - east.x)).toBeGreaterThan(650);
  });

  it("deduplicates experiment artifacts by stage and scientific identity", () => {
    const artifact = (generated_at_utc: string | null): ExperimentSummary => ({
      experiment_id: "ann02-20261118T124527Z", stage: "stage06", kind: "training_run",
      generated_at_utc, selected: false,
    });
    const timeline = mergeExperimentTimeline(
      [artifact(null)],
      [artifact("2026-11-18T12:45:27.000Z")],
    );

    expect(timeline).toHaveLength(1);
    expect(timeline[0].generated_at_utc).toBe("2026-11-18T12:45:27.000Z");
  });
});
