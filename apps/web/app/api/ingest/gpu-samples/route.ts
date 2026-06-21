import { isAuthorized, unauthorized } from "@/lib/auth";
import { insertGpuSamples } from "@/lib/db";
import type { NextRequest } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  if (!isAuthorized(request)) {
    return unauthorized();
  }
  try {
    const body = await request.json();
    const inputRows = Array.isArray(body) ? body : body.samples;
    const samples = await insertGpuSamples(
      inputRows.map((sample: Record<string, unknown>) => ({
        id: sample.id as string | undefined,
        runId: String(sample.runId),
        timestamp: String(sample.timestamp),
        gpuUtil: Number(sample.gpuUtil),
        memUtil: Number(sample.memUtil),
        memUsedMib: Number(sample.memUsedMib),
        memTotalMib: Number(sample.memTotalMib),
      })),
    );
    return Response.json({ samples });
  } catch (error) {
    return Response.json({ error: "gpu_sample_ingest_failed", message: String(error) }, { status: 500 });
  }
}
