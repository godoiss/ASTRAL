import type { ExperimentSummary } from "@/lib/types";

/**
 * Merge persisted experiment indexes by their scientific identity.
 *
 * Multiple artifacts may describe the same run. The timeline shows one entry,
 * retaining the timestamp-bearing representation and any selected flag.
 */
export function mergeExperimentTimeline(...groups: ExperimentSummary[][]): ExperimentSummary[] {
  const byIdentity = new Map<string, ExperimentSummary>();

  for (const experiment of groups.flat()) {
    const identity = `${experiment.stage}:${experiment.experiment_id}`;
    const current = byIdentity.get(identity);
    if (!current) {
      byIdentity.set(identity, experiment);
      continue;
    }

    const preferred = !current.generated_at_utc && experiment.generated_at_utc ? experiment : current;
    byIdentity.set(identity, { ...preferred, selected: current.selected || experiment.selected });
  }

  return [...byIdentity.values()].sort((left, right) =>
    (right.generated_at_utc ?? "").localeCompare(left.generated_at_utc ?? ""),
  );
}

export function experimentTimelineKey(experiment: ExperimentSummary): string {
  return `${experiment.stage}:${experiment.experiment_id}:${experiment.kind}`;
}
