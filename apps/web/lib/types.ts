export type RunStatus = "queued" | "running" | "completed" | "failed" | "stopped";

export type RunRecord = {
  id: string;
  name: string;
  runner: string | null;
  gpu: string | null;
  status: RunStatus;
  startedAt: string;
  endedAt: string | null;
  hudJobUrl: string | null;
  gitSha: string | null;
  policy: string | null;
  controller: string | null;
  notes: string | null;
};

export type TrialRecord = {
  id: string;
  runId: string;
  op: string;
  trialIndex: number;
  candidate: string;
  accepted: boolean;
  correct: boolean;
  reward: number;
  speedup: number | null;
  latencyMs: number | null;
  bestSpeedupAfter: number | null;
  bestLatencyAfter: number | null;
  costUsd: number | null;
  tokens: number | null;
  sourceBlobUrl: string | null;
  failureReason: string | null;
  createdAt: string;
};

export type GpuSample = {
  id: string;
  runId: string;
  timestamp: string;
  gpuUtil: number;
  memUtil: number;
  memUsedMib: number;
  memTotalMib: number;
};

export type ArtifactRecord = {
  id: string;
  runId: string;
  trialId: string | null;
  kind: string;
  filename: string;
  blobUrl: string;
  sizeBytes: number | null;
  createdAt: string;
};

export type RunSummary = {
  run: RunRecord;
  trials: TrialRecord[];
  gpuSamples: GpuSample[];
  artifacts: ArtifactRecord[];
  bestSpeedup: number;
  bestLatencyMs: number | null;
  acceptedCount: number;
  rejectedCount: number;
  latestTrial: TrialRecord | null;
};
