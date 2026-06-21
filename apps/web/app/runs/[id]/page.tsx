import Link from "next/link";
import { notFound } from "next/navigation";
import { ExternalLink } from "lucide-react";
import { getRun } from "@/lib/db";
import { GpuChart, ImprovementChart, RewardChart } from "@/components/charts";
import { StatusBadge, TrialsTable } from "@/components/run-table";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const detail = await getRun(id);
  if (!detail) notFound();

  const bestSource = detail.artifacts.find((artifact) => artifact.kind.includes("source"));

  return (
    <main className="page">
      <section className="page-header">
        <div>
          <div className="eyebrow">Run detail</div>
          <h1>{detail.run.name}</h1>
          <p className="lede">
            {detail.run.runner ?? "runner unknown"} on {detail.run.gpu ?? "gpu unknown"} with{" "}
            {detail.run.policy ?? "policy not set"}. Git {detail.run.gitSha ?? "unknown"}.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <StatusBadge status={detail.run.status} />
          {detail.run.hudJobUrl ? (
            <a className="button" href={detail.run.hudJobUrl} target="_blank">
              Open HUD <ExternalLink size={14} />
            </a>
          ) : null}
        </div>
      </section>

      <section className="grid metrics">
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Best speedup</div>
            <div className="metric-value">{detail.bestSpeedup.toFixed(2)}x</div>
            <div className="metric-foot">{detail.acceptedCount} accepted edits</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Best latency</div>
            <div className="metric-value">{detail.bestLatencyMs ? `${detail.bestLatencyMs.toFixed(4)} ms` : "n/a"}</div>
            <div className="metric-foot">Lower is better</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Trials</div>
            <div className="metric-value">{detail.trials.length}</div>
            <div className="metric-foot">{detail.rejectedCount} rejected</div>
          </div>
        </article>
        <article className="card metric">
          <div className="card-body">
            <div className="metric-label">Artifacts</div>
            <div className="metric-value">{detail.artifacts.length}</div>
            <div className="metric-foot">Blob-backed source and logs</div>
          </div>
        </article>
      </section>

      <section className="grid two-col" style={{ marginTop: 14 }}>
        <article className="card">
          <div className="card-header">
            <h2>Improvement curve</h2>
          </div>
          <div className="card-body">
            <ImprovementChart trials={detail.trials} />
          </div>
        </article>
        <article className="card">
          <div className="card-header">
            <h2>GPU samples</h2>
          </div>
          <div className="card-body">
            <GpuChart samples={detail.gpuSamples} />
          </div>
        </article>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <div className="card-header">
          <h2>Reward</h2>
          <span className="badge">raw reward preserved</span>
        </div>
        <div className="card-body">
          <RewardChart trials={detail.trials} />
        </div>
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <div className="card-header">
          <h2>Trial audit log</h2>
          {bestSource ? (
            <a className="button" href={bestSource.blobUrl} target="_blank">
              Best source
            </a>
          ) : null}
        </div>
        <TrialsTable trials={detail.trials.slice().reverse()} />
      </section>

      <section className="card" style={{ marginTop: 14 }}>
        <div className="card-header">
          <h2>Artifacts</h2>
          <Link className="button" href="/runs">
            Back to runs
          </Link>
        </div>
        <div className="card-body">
          {detail.artifacts.length ? (
            <ul>
              {detail.artifacts.map((artifact) => (
                <li key={artifact.id}>
                  <a href={artifact.blobUrl} target="_blank">
                    {artifact.kind}: <span className="mono">{artifact.filename}</span>
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No artifacts have been uploaded for this run yet.</p>
          )}
        </div>
      </section>
    </main>
  );
}
