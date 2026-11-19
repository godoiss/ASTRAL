"""Strong Pydantic contracts for the versioned HTTP boundary."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


T = TypeVar("T")


class StrictModel(BaseModel):
    """Forbid undocumented fields at the stable API boundary."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ModelAvailability(StrictModel):
    ann01: bool = Field(alias="ann-01")
    ann02: bool = Field(alias="ann-02")


class ArtifactAvailability(StrictModel):
    dataset_intelligence: bool
    model_comparison: bool
    explainability: bool
    tess_pilot: bool


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    stage: int
    local_only: bool
    models: ModelAvailability
    artifacts: ArtifactAvailability


class DocumentationLinks(StrictModel):
    openapi: str
    swagger: str
    redoc: str


class ScientificGuardrails(StrictModel):
    frozen_test_accessed: bool
    training_enabled: bool
    downloads_enabled: bool
    automatic_ensemble_enabled: bool
    kepler_tess_mixed: bool


class ApiInfoResponse(StrictModel):
    name: str
    version: str
    stage: int
    api_version: str
    documentation: DocumentationLinks
    scientific_guardrails: ScientificGuardrails


class Pagination(StrictModel):
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    returned: int = Field(ge=0)
    total: int = Field(ge=0)


class PageResponse(StrictModel, Generic[T]):
    items: list[T]
    pagination: Pagination


class ProvenanceReference(StrictModel):
    source: str | None = None
    artifact: str | None = None
    cohort: str | None = None
    mission: str | None = None
    status: str | None = None


class ScientificResponse(StrictModel, Generic[T]):
    data: T
    limitations: list[str] = Field(default_factory=list)
    provenance: ProvenanceReference = Field(default_factory=ProvenanceReference)


class NumericSummary(StrictModel):
    records: int = Field(ge=0)
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None
    mean: float | None


class DispositionCounts(StrictModel):
    confirmed: int = Field(default=0, ge=0, alias="CONFIRMED")
    false_positive: int = Field(default=0, ge=0, alias="FALSE POSITIVE")
    candidate: int = Field(default=0, ge=0, alias="CANDIDATE")


class ClassDistribution(StrictModel):
    train: DispositionCounts | None = None
    validation: DispositionCounts | None = None
    candidate: DispositionCounts | None = None


class TessObservationSummary(StrictModel):
    mission: str
    author: str
    sector: int
    target_id: str
    time_format: str
    time_scale: str
    cadence_seconds_median: float
    records: int
    finite_records: int
    quality_flagged_records: int
    time_start: float
    time_end: float
    time_span_days: float


class DatasetEntry(StrictModel):
    id: str
    mission: Literal["Kepler", "TESS"]
    kind: Literal["catalog", "processed_light_curves", "isolated_experimental_light_curve"]
    records_available_to_api: int | None = None
    objects: int | None = None
    version_id: str | None = None
    frozen_test_excluded: bool | None = None
    sha256: str | None = None
    requested: int | None = None
    accepted: int | None = None
    excluded: int | None = None
    failed: int | None = None
    stars: int | None = None
    class_distribution: ClassDistribution | None = None
    target: str | None = None
    downloaded: bool | None = None
    observation: TessObservationSummary | None = None
    mixed_with_kepler_training: bool | None = None


class DatasetCollection(StrictModel):
    datasets: list[DatasetEntry]


class CohortDatasetSummary(StrictModel):
    sha256: str
    requested: int
    accepted: int
    excluded: int
    failed: int
    stars: int


class QualitySummary(StrictModel):
    global_coverage: NumericSummary
    local_coverage: NumericSummary
    observed_transits: NumericSummary
    raw_cadences: NumericSummary
    retained_cadences: NumericSummary
    retention_fraction: NumericSummary
    segments: NumericSummary
    cadence_seconds: NumericSummary
    observation_span_days: NumericSummary


class AstronomySummary(StrictModel):
    orbital_period_days: NumericSummary
    transit_duration_hours: NumericSummary
    transit_depth_ppm: NumericSummary
    catalog_model_snr: NumericSummary
    kepler_magnitude: NumericSummary


class BiasAnalysis(StrictModel):
    quarter: int | None
    period_limit_days: float
    eligible_labelled_records_before_period_limit: int
    eligible_labelled_records_after_period_limit: int
    fraction_eligible_removed_by_period_limit: float
    eligible_period_days: NumericSummary
    selected_period_days: NumericSummary
    eligible_model_snr: NumericSummary
    selected_model_snr: NumericSummary
    documented_effects: list[str]


class DatasetGuardrails(StrictModel):
    frozen_test_accessed: bool
    candidates_used_as_labels: bool
    cross_mission_training: bool


class ObservatorySummaryData(StrictModel):
    dataset: CohortDatasetSummary
    class_distribution: ClassDistribution
    quality: QualitySummary
    astronomy: AstronomySummary
    exclusions: dict[str, int]
    bias_analysis: BiasAnalysis
    guardrails: DatasetGuardrails


class ObjectSummary(StrictModel):
    object_id: str
    display_name: str
    kic: int
    kepler_name: str | None
    catalog_disposition: str
    split: Literal["train", "validation", "candidate", "unassigned"]
    period_days: float | None
    has_cached_observation: bool


class KoiMetadataResponse(StrictModel):
    object_id: str
    display_name: str
    kic: int
    kepler_name: str | None
    disposition: str
    period_days: float
    epoch_bkjd: float
    epoch_bjd: float
    duration_hours: float
    depth_ppm: float | None
    planet_radius_earth: float | None
    equilibrium_temperature_k: float | None
    stellar_temperature_k: float | None
    stellar_radius_solar: float | None
    stellar_mass_solar: float | None
    stellar_logg: float | None
    semi_major_axis_au: float | None
    insolation_earth: float | None
    kepler_magnitude: float | None
    split: Literal["train", "validation", "candidate", "unassigned"]
    class_label: int
    has_cached_observation: bool | None = None


class ObservationData(StrictModel):
    object: KoiMetadataResponse
    mission: str
    author: str
    cadence_seconds: float
    observation_start_bkjd: float
    observation_end_bkjd: float
    observation_span_days: float
    segments: list[str]
    raw_records: int
    processed_records: int
    provenance: dict[str, Any]


class RawCurveArrays(StrictModel):
    time_bkjd: list[float | None]
    flux: list[float | None]
    flux_error: list[float | None]
    quality: list[int]


class ProcessedCurveArrays(StrictModel):
    time_bkjd: list[float | None]
    normalized_flux: list[float | None]
    detrended_flux: list[float | None]


class PhaseCurveArrays(StrictModel):
    phase: list[float | None]
    detrended_flux: list[float | None]


class LocalCurveArrays(StrictModel):
    phase_hours: list[float | None]
    flux: list[float | None]


class LightCurveData(StrictModel):
    object_id: str
    mission: str
    view: Literal["raw", "processed", "phase", "local"]
    records_total: int
    records_returned: int
    downsampled: bool
    data: RawCurveArrays | ProcessedCurveArrays | PhaseCurveArrays | LocalCurveArrays


class ClassificationMetrics(StrictModel):
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    binary_cross_entropy: float
    confusion_matrix: list[list[int]]
    records: int
    positive_prevalence: float
    threshold: float


class ModelEntry(StrictModel):
    id: Literal["ann-01", "ann-02"]
    name: str
    framework: Literal["NumPy", "PyTorch"]
    architecture: list[dict[str, Any]] | dict[str, Any]
    trainable_parameters: int
    experiment_id: str
    served_checkpoint: str
    seed: int | None = None
    shared_validation_metrics: ClassificationMetrics
    definitive_scientific_validation: bool


class ModelCollection(StrictModel):
    models: list[ModelEntry]


class EpochMetrics(StrictModel):
    epoch: float
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


class Ann01InferenceDetails(StrictModel):
    raw_features: dict[str, float | None]
    normalized_features: dict[str, float]
    activations: list[list[float]]


class Ann02InferenceDetails(StrictModel):
    split: Literal["train", "validation", "candidate"]
    device: str
    elapsed_seconds: float
    feature_map_shapes: dict[str, list[int]]


class InferenceRequest(StrictModel):
    object_id: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9 ._-]+$")

    @field_validator("object_id")
    @classmethod
    def normalize_object_id(cls, value: str) -> str:
        return value.strip()


class InferenceResponse(StrictModel):
    model_id: Literal["ann-01", "ann-02"]
    model_experiment_id: str
    object_id: str
    kic: int
    original_disposition: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_class: str
    input_kind: str
    details: Ann01InferenceDetails | Ann02InferenceDetails
    scientific_notice: str


class ComparisonCohort(StrictModel):
    definition: str
    records: int
    stars: int
    class_0: int
    class_1: int
    test_accessed: bool


class ComparisonAgreement(StrictModel):
    records: int
    fraction: float
    divergences: int


class ComparisonRow(StrictModel):
    object_id: str
    kic: int
    target: int
    catalog_disposition: str
    ann01_score: float
    ann01_class: int
    ann02_score: float
    ann02_class: int
    models_agree: bool
    ann01_correct: bool
    ann02_correct: bool


class ModelComparisonData(StrictModel):
    schema_version: int
    cohort: ComparisonCohort
    ann01_metrics: ClassificationMetrics
    ann02_metrics: ClassificationMetrics
    agreement: ComparisonAgreement
    rows: list[ComparisonRow]
    limitations: list[str]


class ImportanceFeature(StrictModel):
    feature: str
    mean_bce_increase: float
    std_bce_increase: float


class PermutationImportance(StrictModel):
    method: str
    records: int
    baseline_bce: float
    repeats: int
    seed: int
    features: list[ImportanceFeature]
    test_accessed: bool


class ExplanationObject(StrictModel):
    object_id: str
    kic: int
    split: str
    catalog_disposition: str


class ExplanationGuardrails(StrictModel):
    test_accessed: bool
    candidate_used_as_label: bool
    causal_or_astrophysical_claim: bool


class ExplanationIndexData(StrictModel):
    ann01_permutation_importance: PermutationImportance
    objects: list[ExplanationObject]
    guardrails: ExplanationGuardrails


class LocalFeatureEffect(StrictModel):
    feature: str
    raw_value: float | None
    normalized_value: float
    normalized_gradient: float
    raw_unit_gradient: float
    local_contribution: float


class Ann01LocalExplanation(StrictModel):
    method: str
    score: float
    baseline: str
    features: list[LocalFeatureEffect]
    limitations: list[str]


class SignalRegionEffect(StrictModel):
    view: Literal["global", "local"]
    start_bin: int
    end_bin: int
    start_coordinate: float
    end_coordinate: float
    observed_fraction: float
    score_without_region: float
    score_delta: float
    mean_abs_gradient_x_input: float


class SaliencySeries(StrictModel):
    source_points: int
    returned_points: int
    indices: list[int]
    values: list[float | None]


class Ann02LocalExplanation(StrictModel):
    method: str
    object_id: str
    kic: int
    score: float
    regions: list[SignalRegionEffect]
    saliency: dict[str, SaliencySeries] | None = None
    limitations: list[str]


class ExplanationDetailData(StrictModel):
    object_id: str
    kic: int
    split: str
    catalog_disposition: str
    ann01: Ann01LocalExplanation | None
    ann02: Ann02LocalExplanation


class ExperimentSummary(StrictModel):
    experiment_id: str
    stage: Literal["stage06", "stage07"]
    kind: Literal["campaign", "training_run"]
    generated_at_utc: str | None
    selected: bool


class CampaignRunSummary(StrictModel):
    name: str
    changed_factor: str
    experiment_id: str
    best_epoch: int
    epochs_completed: int
    duration_seconds: float
    validation_metrics: ClassificationMetrics


class ExperimentDetailData(StrictModel):
    experiment_id: str
    stage: Literal["stage06", "stage07"]
    kind: Literal["campaign", "training_run"]
    generated_at_utc: str | None
    architecture: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None
    best_epoch: int | None = None
    epochs_completed: int | None = None
    duration_seconds: float | None = None
    early_stopped: bool | None = None
    history: list[EpochMetrics] = Field(default_factory=list)
    validation_metrics: ClassificationMetrics | None = None
    selected_trial: str | None = None
    selected_experiment_id: str | None = None
    selection_metric: str | None = None
    runs: list[CampaignRunSummary] = Field(default_factory=list)
    guardrails: dict[str, bool] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class TessCompatibilityDifference(StrictModel):
    topic: str
    kepler: str
    tess: str


class TessCompatibility(StrictModel):
    schema_version: int
    status: str
    common_contract: dict[str, Any]
    differences: list[TessCompatibilityDifference]
    requirements_before_cross_mission_validation: list[str]
    sources: list[str]
    mixed_with_kepler_training: bool


class TessGuardrails(StrictModel):
    files_requested: int
    mixed_with_kepler_training: bool


class TessExperimentalData(StrictModel):
    schema_version: int
    generated_at_utc: str
    target: str
    sector_requested: int
    exptime_requested_seconds: int
    search_records: int
    selected_records: int
    product: dict[str, str]
    downloaded: bool
    compatibility: TessCompatibility
    guardrails: TessGuardrails
    observation: TessObservationSummary
    elapsed_seconds: float


class DiscoveryPlanTarget(StrictModel):
    kic: int
    mission: Literal["Kepler"]
    cache_available: bool
    selection_reference: str | None
    label: Literal[-1]


class DiscoveryPlanData(StrictModel):
    sample_size: int = Field(ge=1, le=25)
    cached_targets: int = Field(ge=0)
    network_enabled: Literal[False]
    execution_enabled: bool
    protected_host_overlap: list[int]
    targets: list[DiscoveryPlanTarget]


class DiscoveryCampaignSummary(StrictModel):
    campaign_id: str
    status: str
    started_at_utc: str
    completed_at_utc: str | None
    investigated_stars: int = Field(ge=0)
    screening_detections: int = Field(ge=0)
    catalog_associated_detections: int = Field(ge=0)
    unassociated_inconclusive_detections: int = Field(ge=0)
    below_threshold: int = Field(ge=0)
    failed_or_excluded_stars: int = Field(ge=0)
    confirmed_discoveries: Literal[0]


class DiscoveryDetection(StrictModel):
    kic: int
    selection_reference: str | None
    status: str
    period_days: float
    duration_hours: float
    depth_ppm: float
    depth_snr: float
    ranking_score: float
    observed_events: int
    event_coverage: float
    phase_coverage: float
    possible_eclipsing_binary: bool
    catalog_period_association: bool
    catalog_relations: list[str]
    interpretation: str
    limitations: list[str]


class DiscoveryCampaignDetail(StrictModel):
    campaign: DiscoveryCampaignSummary
    targets: list[DiscoveryDetection]
    failures: list[dict[str, Any]]
    scientific_boundaries: list[str]


class DiscoverySeriesData(StrictModel):
    campaign_id: str
    kic: int
    records_total: int
    records_returned: int
    downsampled: bool
    time_bkjd: list[float]
    detrended_flux: list[float]
    periods: list[float]
    powers: list[float]
    phase: list[float]
    phase_flux: list[float]
    phase_bin_centers: list[float | None]
    phase_bin_flux: list[float | None]


class DiscoveryRunRequest(StrictModel):
    sample_size: int = Field(default=10, ge=1, le=25)


class DiscoveryJob(StrictModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    current: int = Field(ge=0)
    total: int = Field(ge=0)
    message: str
    campaign_id: str | None = None
    error_code: str | None = None


class ErrorBody(StrictModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None
    request_id: str | None = None


class ErrorResponse(StrictModel):
    error: ErrorBody
