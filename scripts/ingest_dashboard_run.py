#!/usr/bin/env python3
"""Ingest an existing Protean optimizer run into the Vercel dashboard.

This script is intentionally dependency-free so it can run on Spark or Modal.
It reads Protean's existing JSONL artifacts and posts normalized rows to the
dashboard ingestion API.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post(base_url: str, token: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{path} failed: HTTP {exc.code}: {detail}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None


def _first_summary(run_dir: Path) -> dict[str, Any]:
    for path in sorted(run_dir.glob("**/summary_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _trial_payload(run_id: str, op: str, row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
    best_after = row.get("best_summary_after") if isinstance(row.get("best_summary_after"), dict) else {}
    eval_error = row.get("eval_error") if isinstance(row.get("eval_error"), dict) else None
    speedup = (
        summary.get("mean_held_out_speedup")
        or row.get("candidate_held_out_speedup")
        or row.get("speedup")
    )
    latency = row.get("candidate_latency_ms") or row.get("candidate_geomean_mean_ms")
    failure = None
    if eval_error:
        failure = f"{eval_error.get('type', 'error')}: {eval_error.get('message', '')}".strip()
    elif not row.get("accepted") and row.get("correct") is False:
        failure = "incorrect candidate"
    elif not row.get("accepted"):
        failure = "did not beat current best"

    return {
        "id": f"{run_id}:{op}:{row.get('trial', row.get('trial_index', int(time.time())))}:{row.get('edit', row.get('candidate', 'candidate'))}",
        "runId": run_id,
        "op": op,
        "trialIndex": int(row.get("trial", row.get("trial_index", 0)) or 0),
        "candidate": str(row.get("edit", row.get("candidate", "candidate"))),
        "accepted": bool(row.get("accepted")),
        "correct": False if eval_error else bool(summary.get("correct_held_out", row.get("correct", True))),
        "reward": float(row.get("candidate_optimizer_reward", summary.get("mean_optimizer_reward", row.get("reward", 0))) or 0),
        "speedup": None if speedup is None else float(speedup),
        "latencyMs": None if latency is None else float(latency),
        "bestSpeedupAfter": best_after.get("mean_held_out_speedup") or row.get("best_held_out_speedup_after"),
        "bestLatencyAfter": row.get("best_latency_ms_after") or row.get("best_geomean_mean_ms_after"),
        "costUsd": row.get("model_cost_usd"),
        "tokens": row.get("tokens"),
        "sourceBlobUrl": None,
        "failureReason": failure,
        "createdAt": row.get("created_at") or row.get("time") and time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(row["time"]))),
    }


def _upload_artifact(base_url: str, token: str, run_id: str, trial_id: str | None, path: Path, kind: str) -> str | None:
    if not path.exists() or path.stat().st_size > 512_000:
        return None
    payload = {
        "runId": run_id,
        "trialId": trial_id,
        "kind": kind,
        "filename": path.name,
        "contentBase64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
    result = _post(base_url, token, "/api/ingest/artifacts", payload)
    artifact = result.get("artifact") if isinstance(result, dict) else None
    return artifact.get("blobUrl") if isinstance(artifact, dict) else None


def ingest(args: argparse.Namespace) -> int:
    base_url = args.site_url or os.environ.get("PROTEAN_DASHBOARD_URL")
    token = args.token or os.environ.get("PROTEAN_INGEST_TOKEN")
    if not base_url or not token:
        print("PROTEAN_DASHBOARD_URL/site-url and PROTEAN_INGEST_TOKEN/token are required", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir).resolve()
    summary = _first_summary(run_dir)
    run_id = args.run_id or run_dir.name
    run_payload = {
        "id": run_id,
        "name": args.name or run_dir.name,
        "runner": args.runner,
        "gpu": args.gpu,
        "status": args.status,
        "startedAt": args.started_at,
        "endedAt": args.ended_at,
        "hudJobUrl": summary.get("hud_job_url"),
        "gitSha": args.git_sha or _git_sha(Path.cwd()),
        "policy": summary.get("edit_policy") or args.policy,
        "controller": summary.get("controller_path"),
        "notes": args.notes,
    }
    _post(base_url, token, "/api/ingest/runs", run_payload)

    posted = 0
    for path in sorted(run_dir.glob("**/trials.jsonl")):
        op = path.parent.name
        for row in _jsonl(path):
            if row.get("event") != "trial":
                continue
            payload = _trial_payload(run_id, op, row)
            candidate_path = Path(row.get("source_path", ""))
            if not candidate_path.is_absolute():
                candidate_path = (Path.cwd() / candidate_path).resolve()
            if candidate_path.exists():
                blob_url = _upload_artifact(base_url, token, run_id, payload["id"], candidate_path, "candidate_source")
                if blob_url:
                    payload["sourceBlobUrl"] = blob_url
            _post(base_url, token, "/api/ingest/trials", payload)
            posted += 1

    gpu_path = run_dir / "gpu_utilization.jsonl"
    samples = []
    for row in _jsonl(gpu_path):
        samples.append(
            {
                "id": f"{run_id}:gpu:{row.get('time', len(samples))}",
                "runId": run_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(row.get("time", time.time())))),
                "gpuUtil": float(row.get("gpu_util", 0)),
                "memUtil": float(row.get("mem_util", 0)),
                "memUsedMib": float(row.get("mem_used_mib", 0)),
                "memTotalMib": float(row.get("mem_total_mib", 0)),
            }
        )
    for offset in range(0, len(samples), 100):
        _post(base_url, token, "/api/ingest/gpu-samples", {"samples": samples[offset : offset + 100]})

    print(json.dumps({"run_id": run_id, "trials_posted": posted, "gpu_samples_posted": len(samples)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--site-url")
    parser.add_argument("--token")
    parser.add_argument("--run-id")
    parser.add_argument("--name")
    parser.add_argument("--runner", default="spark/modal")
    parser.add_argument("--gpu")
    parser.add_argument("--status", default="completed")
    parser.add_argument("--started-at")
    parser.add_argument("--ended-at")
    parser.add_argument("--git-sha")
    parser.add_argument("--policy")
    parser.add_argument("--notes")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(ingest(parse_args()))
