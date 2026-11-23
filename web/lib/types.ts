export interface Provenance { source?: string | null; artifact?: string | null; cohort?: string | null; mission?: string | null; status?: string | null }
export interface Scientific<T> { data: T; limitations: string[]; provenance: Provenance }
export interface Pagination { offset: number; limit: number; returned: number; total: number }
export interface ObjectPage<T> { items: T[]; pagination: Pagination }

export interface ObjectSummary {
  object_id: string; display_name: string; kic: number; kepler_name: string | null;
  catalog_disposition: string; split: string; period_days: number | null;
  has_cached_observation: boolean; ra_deg: number | null; dec_deg: number | null;
}

export interface ObjectDetail {
  object_id: string; display_name: string; kic: number; kepler_name: string | null; disposition: string;
  period_days: number; epoch_bkjd: number; epoch_bjd: number; duration_hours: number; depth_ppm: number | null;
  planet_radius_earth: number | null; equilibrium_temperature_k: number | null; stellar_temperature_k: number | null;
  stellar_radius_solar: number | null; stellar_mass_solar: number | null; stellar_logg: number | null;
  semi_major_axis_au: number | null; insolation_earth: number | null; kepler_magnitude: number | null;
  split: string; class_label: number; has_cached_observation: boolean; ra_deg: number | null; dec_deg: number | null;
}

export interface LightCurve {
  object_id: string; mission: string; view: "raw" | "processed" | "phase" | "local";
  records_total: number; records_returned: number; downsampled: boolean;
  data: Record<string, Array<number | null>>;
}

export interface ClassificationMetrics {
  accuracy: number; balanced_accuracy: number; precision: number; recall: number; f1: number;
  roc_auc: number; pr_auc: number; binary_cross_entropy: number; confusion_matrix: number[][];
  records: number; positive_prevalence: number; threshold: number;
}

export interface ModelEntry {
  id: "ann-01" | "ann-02"; name: string; framework: string; architecture: unknown;
  trainable_parameters: number; experiment_id: string; served_checkpoint: string; seed?: number | null;
  shared_validation_metrics: ClassificationMetrics; definitive_scientific_validation: boolean;
}

export interface Inference {
  model_id: string; model_experiment_id: string; object_id: string; kic: number; original_disposition: string;
  score: number; threshold: number; predicted_class: string; input_kind: string;
  details: Record<string, unknown>; scientific_notice: string;
}

export interface NeuralTensor {
  name: string; role: "weight" | "bias" | "activation" | "feature_map"; shape: number[];
  values: unknown[]; minimum: number | null; maximum: number | null; mean: number | null; standard_deviation: number | null;
}

export interface NeuralVisualization {
  model_id: "ann-01" | "ann-02"; object_id: string; architecture: unknown;
  parameters: NeuralTensor[]; activations: NeuralTensor[]; notes: string[];
}

export interface CampaignSummary {
  campaign_id: string; status: string; started_at_utc: string; completed_at_utc: string | null;
  investigated_stars: number; screening_detections: number; catalog_associated_detections: number;
  unassociated_inconclusive_detections: number; below_threshold: number; failed_or_excluded_stars: number;
  confirmed_discoveries: number;
}

export interface CampaignTarget {
  kic: number; selection_reference: string; status: string; period_days: number | null; duration_hours: number | null;
  depth_ppm: number | null; depth_snr: number | null; ranking_score: number | null; observed_events: number | null;
  event_coverage: number | null; phase_coverage: number | null; possible_eclipsing_binary: boolean;
  catalog_period_association: boolean; catalog_relations: string[]; interpretation: string; limitations: string[];
}

export interface CampaignDetail { campaign: CampaignSummary & Record<string, unknown>; targets: CampaignTarget[]; guardrails: Record<string, boolean>; limitations: string[] }
export interface DiscoverySeries { kic: number; records_returned: number; time_bkjd: number[]; detrended_flux: number[]; periods: number[]; powers: number[]; phase: number[]; phase_flux: number[]; phase_bin_centers: number[]; phase_bin_flux: number[] }

export type ResearchCampaignState = "planned" | "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
export interface ResearchCampaignJob {
  job_id: string; campaign_id: string; mission: "Kepler"; selection_strategy: string;
  status: ResearchCampaignState; stage: string; current: number; total: number; current_kic: number | null;
  message: string; error_code: string | null; cancel_requested: boolean; allow_network: boolean; max_new_downloads: number;
  campaign_config: Record<string, unknown>; detector_config: Record<string, unknown>; provenance: Record<string, unknown>;
  resource: Record<string, number>; result_summary: Record<string, number> | null; artifact_path: string | null;
  created_at_utc: string; queued_at_utc: string | null; started_at_utc: string | null;
  completed_at_utc: string | null; updated_at_utc: string;
}
export interface ResearchCampaignEvent {
  event_id: number; job_id: string; created_at_utc: string; event_type: string; level: "info" | "warning" | "error";
  stage: string; current: number; total: number; kic: number | null; message: string; elapsed_seconds: number | null;
  details: Record<string, unknown>;
}
export interface ResearchCampaignProvenance {
  job_id: string; campaign_id: string; mission: "Kepler"; selection_strategy: string;
  campaign_config: Record<string, unknown>; detector_config: Record<string, unknown>; provenance: Record<string, unknown>;
  resources: Record<string, number>; artifact_path: string | null; scientific_guardrails: Record<string, boolean>;
}

export interface CandidatePriority {
  score: number; tier: "high" | "medium" | "low"; meaning: string; formula_version: string;
  components: Record<string, number>;
}
export interface CandidateDossierSummary {
  dossier_id: string; kic: number; selection_reference: string | null;
  investigation_state: "catalog_associated" | "unassociated_inconclusive";
  period_days: number; duration_hours: number; depth_ppm: number; depth_snr: number;
  observed_events: number; event_coverage: number; possible_eclipsing_binary: boolean;
  priority: CandidatePriority; review_count: number;
}
export interface CandidateIntelligenceIndex {
  campaign_id: string; generated_at_utc: string; artifact_sha256: string | null;
  summary: { investigated_signals: number; catalog_associated: number; unassociated_inconclusive: number; possible_eclipsing_binary: number; confirmed_discoveries: 0; neural_models_invoked: false; frozen_test_accessed: false };
  ranking_method: Record<string, unknown>; catalog_sources: Array<{ name: string; table: string; url: string; records: number }>;
  scientific_boundaries: string[]; dossiers: CandidateDossierSummary[];
}
export interface HumanReview { review_id: string; kic: number; reviewer: string; decision: string; notes: string; created_at_utc: string }
export interface CandidateDossier {
  dossier_id: string; campaign_id: string; generated_at_utc: string; kic: number; selection_reference: string | null;
  mission: "Kepler"; investigation_state: "catalog_associated" | "unassociated_inconclusive"; interpretation: string;
  observation: { source_kind: string; products: Array<{ productFilename?: string; mission?: string; author?: string }>; raw_sha256: string; start_bkjd: number; end_bkjd: number; span_days: number; raw_records: number; retained_records: number; retention_fraction: number; quality_rejected: number; quarter_bias: string };
  measured_event: { period_days: number; period_uncertainty_days: number; transit_time_bkjd: number; transit_time_uncertainty_days: number; duration_hours: number; duration_uncertainty_hours: number; depth_ppm: number; depth_error_fraction: number; depth_snr: number; expected_events: number; observed_events: number; event_coverage: number; phase_coverage: number; event_depths: number[]; event_depth_mad: number | null };
  diagnostics: { aliases: Array<{ period_days: number; depth_snr: number; ratio_to_primary: number; relation: string | null }>; odd_depth: number | null; even_depth: number | null; odd_even_sigma: number | null; secondary_depth: number | null; secondary_sigma: number | null; possible_eclipsing_binary: boolean };
  catalog_verification: { checked_at_utc: string; koi_records: Array<Record<string, unknown>>; tce_records: Array<Record<string, unknown>>; local_period_association_found: boolean; novelty_established: false; eclipsing_binary_catalog: Record<string, string> };
  instrumental_hypotheses: string[];
  evidence: { favorable: string[]; contrary: string[]; inconclusive: string[] };
  priority: CandidatePriority;
  neural_compatibility: { ann01: { available: false; reason: string }; ann02: { available: false; reason: string }; models_invoked: false };
  review_history: HumanReview[]; limitations: string[];
}

export interface ExperimentSummary { experiment_id: string; stage: string; kind: string; generated_at_utc: string | null; selected: boolean }
export interface EpochMetric { epoch: number; train_loss: number; train_accuracy: number; validation_loss: number; validation_accuracy: number }

export interface Comparison {
  cohort: { definition: string; records: number; stars: number; class_0: number; class_1: number; test_accessed: boolean };
  ann01_metrics: ClassificationMetrics; ann02_metrics: ClassificationMetrics;
  agreement: { records: number; fraction: number; divergences: number };
  rows: Array<{ object_id: string; kic: number; target: number; catalog_disposition: string; ann01_score: number; ann02_score: number; models_agree: boolean }>;
  limitations: string[];
}

export interface ExplanationIndex {
  ann01_permutation_importance: { method: string; records: number; features: Array<{ feature: string; mean_bce_increase: number; std_bce_increase: number }> };
  objects: Array<{ object_id: string; kic: number; split: string; catalog_disposition: string }>;
  guardrails: Record<string, boolean>;
}

export interface ExplanationDetail {
  object_id: string; kic: number; split: string; catalog_disposition: string;
  ann01: null | { score: number; method: string; features: Array<{ feature: string; local_contribution: number; raw_value: number | null }> ; limitations: string[] };
  ann02: { score: number; method: string; regions: Array<{ view: string; start_coordinate: number; end_coordinate: number; score_delta: number; observed_fraction: number }>; saliency?: Record<string, { indices: number[]; values: Array<number | null> }>; limitations: string[] };
}
