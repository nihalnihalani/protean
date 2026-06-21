import { isAuthorized, unauthorized } from "@/lib/auth";
import { insertTrial } from "@/lib/db";
import type { NextRequest } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  if (!isAuthorized(request)) {
    return unauthorized();
  }
  try {
    const body = await request.json();
    const trial = await insertTrial({
      runId: body.runId,
      op: body.op,
      trialIndex: Number(body.trialIndex),
      candidate: body.candidate,
      accepted: Boolean(body.accepted),
      correct: Boolean(body.correct),
      reward: Number(body.reward ?? 0),
      speedup: body.speedup == null ? null : Number(body.speedup),
      latencyMs: body.latencyMs == null ? null : Number(body.latencyMs),
      bestSpeedupAfter: body.bestSpeedupAfter == null ? null : Number(body.bestSpeedupAfter),
      bestLatencyAfter: body.bestLatencyAfter == null ? null : Number(body.bestLatencyAfter),
      costUsd: body.costUsd == null ? null : Number(body.costUsd),
      tokens: body.tokens == null ? null : Number(body.tokens),
      sourceBlobUrl: body.sourceBlobUrl ?? null,
      failureReason: body.failureReason ?? null,
      id: body.id,
      createdAt: body.createdAt,
    });
    return Response.json({ trial });
  } catch (error) {
    return Response.json({ error: "trial_ingest_failed", message: String(error) }, { status: 500 });
  }
}
