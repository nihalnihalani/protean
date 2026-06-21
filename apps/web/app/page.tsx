import Link from "next/link";
import { Activity, ExternalLink, Timer, TrendingDown, TrendingUp } from "lucide-react";
import { getRun, listRuns, listTrials } from "@/lib/db";
import { summarizeRun } from "@/lib/metrics";
import { seedGpuSamples } from "@/lib/seed";
import { GpuChart, ImprovementChart, RewardChart } from "@/components/charts";
import { StatusBadge, TrialsTable } from "@/components/run-table";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [runs, trials] = await Promise.all([listRuns(), listTrials()]);
  const summaries = runs.map((run) =>
    summarizeRun(
      run,
      trials.filter((trial) => trial.runId === run.id),
    ),
  );
  const active = summaries.find((summary) => summary.run.status === "running") ?? summaries[0];
  const activeDetail = active ? await getRun(active.run.id) : null;
  const latest = activeDetail?.latestTrial ?? null;

  return (
    <main className="page">
      <section className="page-header">
        <div>
          <div className="eyebrow">Live GPU optimization</div>
          <h1>Protean turns kernel edits into measurable speedup.</h1>
          <p className="lede">
            Track every candidate, correctness verdict, latency shift, reward, GPU utilization sample, and HUD trace
            from Spark or Modal while the optimizer keeps improving kernels.
          </p>
        </div>
        {active ? <StatusBadge status={active.run.status} /> : null}
      </section>

      <section className="grid metrics">
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Best speedup</div>
            <div className="metric-value">{active ? `${active.bestSpeedup.toFixed(2)}x` : "0.00x"}</div>
            <div className="metric-foot">
              <TrendingUp size={14} /> accepted edits move this up
            </div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Best latency</div>
            <div className="metric-value">
              {active?.bestLatencyMs ? `${active.bestLatencyMs.toFixed(4)} ms` : "n/a"}
            </div>
            <div className="metric-foot">
              <TrendingDown size={14} /> lower is better
            </div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Accepted / trials</div>
            <div className="metric-value">
              {active ? `${active.acceptedCount}/${active.trials.length}` : "0/0"}
            </div>
            <div className="metric-foot">Rejected rows stay in the audit log</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Latest reward</div>
            <div className="metric-value">{latest ? latest.reward.toFixed(3) : "0.000"}</div>
            <div className="metric-foot">Rewards greater than 1 are preserved</div>
          </div>
        </article>
      </section>

      <section className="grid dashboard-grid" style={{ marginTop: 14 }}>
        <div className="grid">
          <article className="card">
            <div className="card-header">
              <h2>Speedup up, latency down</h2>
              {active ? (
                <Link className="button" href={`/runs/${active.run.id}`}>
                  <Activity size={15} /> Run detail
                </Link>
              ) : null}
            </div>
            <div className="card-body">
              <ImprovementChart trials={activeDetail?.trials ?? active?.trials ?? []} />
            </div>
          </article>

          <article className="card">
            <div className="card-header">
              <h2>Trial reward curve</h2>
              <span className="badge">accepted edits are chart annotations</span>
            </div>
            <div className="card-body">
              <RewardChart trials={activeDetail?.trials ?? active?.trials ?? []} />
            </div>
          </article>
        </div>

        <aside className="grid">
          <article className="card">
            <div className="card-header">
              <h2>Active run</h2>
              {active?.run.hudJobUrl ? (
                <a className="button" href={active.run.hudJobUrl} target="_blank">
                  HUD <ExternalLink size={14} />
                </a>
              ) : null}
            </div>
            <div className="card-body">
              <div className="mono">{active?.run.name ?? "No run yet"}</div>
              <p className="muted">
                {active?.run.runner ?? "runner"} on {active?.run.gpu ?? "gpu"} with {active?.run.policy ?? "policy not set"}
              </p>
              <p className="muted">
                <Timer size={14} /> Started {active ? new Date(active.run.startedAt).toLocaleString() : "n/a"}
              </p>
            </div>
          </article>

          <article className="card">
            <div className="card-header">
              <h2>GPU utilization</h2>
              <span className="badge good">live samples</span>
            </div>
            <div className="card-body">
              <GpuChart samples={activeDetail?.gpuSamples.length ? activeDetail.gpuSamples : seedGpuSamples} />
            </div>
          </article>
        </aside>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <div className="card-header">
          <h2>Latest candidates</h2>
          <Link className="button" href="/runs">
            All runs
          </Link>
        </div>
        <TrialsTable trials={(activeDetail?.trials ?? []).slice(-8).reverse()} />
      </section>
    </main>
  );
}
