import { isAuthorized, unauthorized } from "@/lib/auth";
import { upsertRun } from "@/lib/db";
import type { NextRequest } from "next/server";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  if (!isAuthorized(request)) {
    return unauthorized();
  }
  try {
    const run = await upsertRun(await request.json());
    return Response.json({ run });
  } catch (error) {
    return Response.json({ error: "run_ingest_failed", message: String(error) }, { status: 500 });
  }
}
