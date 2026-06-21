import { listRuns, listTrials } from "@/lib/db";
import { summarizeRun } from "@/lib/metrics";
import { RunsTable } from "@/components/run-table";

export const dynamic = "force-dynamic";

export default async function RunsPage() {
  const [runs, trials] = await Promise.all([listRuns(), listTrials()]);
  const summaries = runs.map((run) =>
    summarizeRun(
      run,
      trials.filter((trial) => trial.runId === run.id),
    ),
  );

  return (
    <main className="page">
      <section className="page-header">
        <div>
          <div className="eyebrow">Run history</div>
          <h1>Every overnight run, smoke test, and live optimizer session.</h1>
          <p className="lede">Filter in the browser with the search box for now; production data comes from Postgres.</p>
        </div>
      </section>
      <div className="filterbar">
        <input className="input" placeholder="Search runs, GPUs, policies" aria-label="Search runs" />
        <select className="select" aria-label="Status filter" defaultValue="all">
          <option value="all">All statuses</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
        </select>
      </div>
      <section className="card">
        <div className="card-header">
          <h2>Runs</h2>
          <span className="badge">{summaries.length} total</span>
        </div>
        <RunsTable runs={summaries} />
      </section>
    </main>
  );
}
