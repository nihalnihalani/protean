import Link from "next/link";
import type { RunSummary, TrialRecord } from "@/lib/types";

export function StatusBadge({ status }: { status: string }) {
  const tone = status === "running" ? "good" : status === "completed" ? "" : "bad";
  return (
    <span className={`badge ${tone}`}>
      <span className={`status-dot ${status}`} />
      {status}
    </span>
  );
}

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Status</th>
            <th>Runner</th>
            <th>Best Speedup</th>
            <th>Best Latency</th>
            <th>Trials</th>
            <th>HUD</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((summary) => (
            <tr key={summary.run.id}>
              <td>
                <Link href={`/runs/${summary.run.id}`} className="mono">
                  {summary.run.name}
                </Link>
                <div className="muted">{new Date(summary.run.startedAt).toLocaleString()}</div>
              </td>
              <td>
                <StatusBadge status={summary.run.status} />
              </td>
              <td>
                {summary.run.runner ?? "unknown"}
                <div className="muted">{summary.run.gpu ?? "gpu not set"}</div>
              </td>
              <td className="mono">{summary.bestSpeedup.toFixed(2)}x</td>
              <td className="mono">{summary.bestLatencyMs ? `${summary.bestLatencyMs.toFixed(4)} ms` : "n/a"}</td>
              <td>
                {summary.trials.length}
                <div className="muted">{summary.acceptedCount} accepted</div>
              </td>
              <td>
                {summary.run.hudJobUrl ? (
                  <a className="button" href={summary.run.hudJobUrl} target="_blank">
                    Open HUD
                  </a>
                ) : (
                  <span className="muted">none</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TrialsTable({ trials }: { trials: TrialRecord[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Trial</th>
            <th>Op</th>
            <th>Candidate</th>
            <th>Verdict</th>
            <th>Reward</th>
            <th>Speedup</th>
            <th>Latency</th>
            <th>Cost</th>
          </tr>
        </thead>
        <tbody>
          {trials.map((trial) => (
            <tr key={trial.id}>
              <td className="mono">#{trial.trialIndex}</td>
              <td className="mono">{trial.op}</td>
              <td>
                {trial.sourceBlobUrl ? (
                  <a href={trial.sourceBlobUrl} target="_blank" className="mono">
                    {trial.candidate}
                  </a>
                ) : (
                  <span className="mono">{trial.candidate}</span>
                )}
                {trial.failureReason ? <div className="muted">{trial.failureReason}</div> : null}
              </td>
              <td>
                <span className={`badge ${trial.accepted ? "good" : trial.correct ? "warn" : "bad"}`}>
                  {trial.accepted ? "accepted" : trial.correct ? "rejected" : "incorrect"}
                </span>
              </td>
              <td className="mono">{trial.reward.toFixed(3)}</td>
              <td className="mono">{trial.speedup == null ? "n/a" : `${trial.speedup.toFixed(2)}x`}</td>
              <td className="mono">{trial.latencyMs == null ? "n/a" : `${trial.latencyMs.toFixed(4)} ms`}</td>
              <td className="mono">{trial.costUsd == null ? "n/a" : `$${trial.costUsd.toFixed(4)}`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
