import type { GpuSample, RunRecord, RunSummary, TrialRecord, ArtifactRecord } from "./types";

export function summarizeRun(
  run: RunRecord,
  trials: TrialRecord[],
  gpuSamples: GpuSample[] = [],
  artifacts: ArtifactRecord[] = [],
): RunSummary {
  const accepted = trials.filter((trial) => trial.accepted);
  const rejected = trials.filter((trial) => !trial.accepted);
  const speedups = trials
    .map((trial) => trial.bestSpeedupAfter ?? trial.speedup ?? 0)
    .filter((value) => Number.isFinite(value));
  const latencies = trials
    .map((trial) => trial.bestLatencyAfter ?? trial.latencyMs)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0);

  return {
    run,
    trials,
    gpuSamples,
    artifacts,
    bestSpeedup: speedups.length ? Math.max(...speedups) : 0,
    bestLatencyMs: latencies.length ? Math.min(...latencies) : null,
    acceptedCount: accepted.length,
    rejectedCount: rejected.length,
    latestTrial: trials.at(-1) ?? null,
  };
}

export function opLeaderboard(trials: TrialRecord[]) {
  const byOp = new Map<string, TrialRecord[]>();
  for (const trial of trials) {
    byOp.set(trial.op, [...(byOp.get(trial.op) ?? []), trial]);
  }
  return [...byOp.entries()]
    .map(([op, rows]) => {
      const bestSpeedup = Math.max(...rows.map((row) => row.bestSpeedupAfter ?? row.speedup ?? 0));
      const latencyRows = rows
        .map((row) => row.bestLatencyAfter ?? row.latencyMs)
        .filter((value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0);
      return {
        op,
        trials: rows.length,
        accepted: rows.filter((row) => row.accepted).length,
        bestSpeedup,
        bestLatencyMs: latencyRows.length ? Math.min(...latencyRows) : null,
        latestReward: rows.at(-1)?.reward ?? 0,
      };
    })
    .sort((a, b) => b.bestSpeedup - a.bestSpeedup);
}
