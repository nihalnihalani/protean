import { listRuns, listTrials } from "@/lib/db";
import { summarizeRun } from "@/lib/metrics";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  const [runs, trials] = await Promise.all([listRuns(), listTrials()]);
  const summaries = runs.map((run) =>
    summarizeRun(
      run,
      trials.filter((trial) => trial.runId === run.id),
    ),
  );
  return Response.json({ runs: summaries });
}
