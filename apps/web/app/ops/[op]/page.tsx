import { notFound } from "next/navigation";
import { getOp } from "@/lib/db";
import { ImprovementChart, RewardChart } from "@/components/charts";
import { TrialsTable } from "@/components/run-table";

export const dynamic = "force-dynamic";

export default async function OpPage({ params }: { params: Promise<{ op: string }> }) {
  const { op } = await params;
  const detail = await getOp(decodeURIComponent(op));
  if (!detail.trials.length) {
    notFound();
  }
  const bestSpeedup = Math.max(...detail.trials.map((trial) => trial.bestSpeedupAfter ?? trial.speedup ?? 0));
  const accepted = detail.trials.filter((trial) => trial.accepted);
  const best = accepted.sort((a, b) => (b.speedup ?? 0) - (a.speedup ?? 0))[0] ?? detail.trials.at(-1);

  return (
    <main className="page">
      <section className="page-header">
        <div>
          <div className="eyebrow">Operation leaderboard</div>
          <h1 className="mono">{detail.op}</h1>
          <p className="lede">Per-op history across all runs, candidates, reward changes, and accepted improvements.</p>
        </div>
        <span className="badge good">{bestSpeedup.toFixed(2)}x best</span>
      </section>

      <section className="grid metrics">
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Best speedup</div>
            <div className="metric-value">{bestSpeedup.toFixed(2)}x</div>
            <div className="metric-foot">Across {detail.trials.length} trials</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Accepted candidates</div>
            <div className="metric-value">{accepted.length}</div>
            <div className="metric-foot">{detail.trials.length - accepted.length} rejected</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Best candidate</div>
            <div className="metric-value" style={{ fontSize: 18 }}>
              {best?.candidate ?? "n/a"}
            </div>
            <div className="metric-foot">Accepted source artifact links appear in run detail</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Latest reward</div>
            <div className="metric-value">{(detail.trials.at(-1)?.reward ?? 0).toFixed(3)}</div>
            <div className="metric-foot">Rewards may exceed 1</div>
          </div>
        </article>
      </section>

      <section className="grid two-col" style={{ marginTop: 14 }}>
        <article className="card">
          <div className="card-header">
            <h2>Speedup and latency</h2>
          </div>
          <div className="card-body">
            <ImprovementChart trials={detail.trials} />
          </div>
        </article>
        <article className="card">
          <div className="card-header">
            <h2>Reward curve</h2>
          </div>
          <div className="card-body">
            <RewardChart trials={detail.trials} />
          </div>
        </article>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <div className="card-header">
          <h2>Op trials</h2>
        </div>
        <TrialsTable trials={detail.trials.slice().reverse()} />
      </section>
    </main>
  );
}
