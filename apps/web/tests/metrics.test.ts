import assert from "node:assert/strict";
import test from "node:test";
import { summarizeRun } from "../lib/metrics";
import type { RunRecord, TrialRecord } from "../lib/types";

const run: RunRecord = {
  id: "run",
  name: "test",
  runner: "modal",
  gpu: "B200",
  status: "running",
  startedAt: new Date(0).toISOString(),
  endedAt: null,
  hudJobUrl: null,
  gitSha: null,
  policy: "fireworks",
  controller: null,
  notes: null,
};

function trial(partial: Partial<TrialRecord>): TrialRecord {
  return {
    id: partial.id ?? crypto.randomUUID(),
    runId: "run",
    op: "rmsnorm",
    trialIndex: partial.trialIndex ?? 1,
    candidate: partial.candidate ?? "candidate",
    accepted: partial.accepted ?? false,
    correct: partial.correct ?? true,
    reward: partial.reward ?? 0,
    speedup: partial.speedup ?? null,
    latencyMs: partial.latencyMs ?? null,
    bestSpeedupAfter: partial.bestSpeedupAfter ?? null,
    bestLatencyAfter: partial.bestLatencyAfter ?? null,
    costUsd: partial.costUsd ?? null,
    tokens: partial.tokens ?? null,
    sourceBlobUrl: partial.sourceBlobUrl ?? null,
    failureReason: partial.failureReason ?? null,
    createdAt: partial.createdAt ?? new Date().toISOString(),
  };
}

test("summary preserves rewards greater than one", () => {
  const summary = summarizeRun(run, [
    trial({ accepted: true, reward: 3.2, speedup: 3.2, bestSpeedupAfter: 3.2, latencyMs: 0.02 }),
  ]);
  assert.equal(summary.latestTrial?.reward, 3.2);
  assert.equal(summary.bestSpeedup, 3.2);
  assert.equal(summary.acceptedCount, 1);
});

test("summary does not treat correct rejected trials as accepted", () => {
  const summary = summarizeRun(run, [
    trial({ accepted: false, correct: true, reward: 0.8, speedup: 0.8, latencyMs: 0.03 }),
    trial({ accepted: false, correct: false, reward: 0, speedup: 0, latencyMs: null }),
  ]);
  assert.equal(summary.acceptedCount, 0);
  assert.equal(summary.rejectedCount, 2);
  assert.equal(summary.bestSpeedup, 0.8);
});

test("summary computes the lowest latency across trial fields", () => {
  const summary = summarizeRun(run, [
    trial({ latencyMs: 0.04, bestLatencyAfter: 0.04 }),
    trial({ latencyMs: 0.03, bestLatencyAfter: 0.025 }),
  ]);
  assert.equal(summary.bestLatencyMs, 0.025);
});
