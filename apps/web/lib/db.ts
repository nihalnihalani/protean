import postgres from "postgres";
import { summarizeRun } from "./metrics";
import { seedArtifacts, seedGpuSamples, seedRuns, seedTrials } from "./seed";
import type { ArtifactRecord, GpuSample, RunRecord, RunStatus, TrialRecord } from "./types";

type Sql = ReturnType<typeof postgres>;

let client: Sql | null = null;
let schemaReady = false;

export function hasDatabase() {
  return Boolean(process.env.DATABASE_URL);
}

function getSql(): Sql | null {
  if (!process.env.DATABASE_URL) {
    return null;
  }
  if (!client) {
    client = postgres(process.env.DATABASE_URL, {
      max: 3,
      prepare: false,
    });
  }
  return client;
}

function runFromRow(row: Record<string, unknown>): RunRecord {
  return {
    id: String(row.id),
    name: String(row.name),
    runner: row.runner ? String(row.runner) : null,
    gpu: row.gpu ? String(row.gpu) : null,
    status: String(row.status) as RunStatus,
    startedAt: new Date(String(row.started_at)).toISOString(),
    endedAt: row.ended_at ? new Date(String(row.ended_at)).toISOString() : null,
    hudJobUrl: row.hud_job_url ? String(row.hud_job_url) : null,
    gitSha: row.git_sha ? String(row.git_sha) : null,
    policy: row.policy ? String(row.policy) : null,
    controller: row.controller ? String(row.controller) : null,
    notes: row.notes ? String(row.notes) : null,
  };
}

function trialFromRow(row: Record<string, unknown>): TrialRecord {
  return {
    id: String(row.id),
    runId: String(row.run_id),
    op: String(row.op),
    trialIndex: Number(row.trial_index),
    candidate: String(row.candidate),
    accepted: Boolean(row.accepted),
    correct: Boolean(row.correct),
    reward: Number(row.reward),
    speedup: row.speedup === null ? null : Number(row.speedup),
    latencyMs: row.latency_ms === null ? null : Number(row.latency_ms),
    bestSpeedupAfter: row.best_speedup_after === null ? null : Number(row.best_speedup_after),
    bestLatencyAfter: row.best_latency_after === null ? null : Number(row.best_latency_after),
    costUsd: row.cost_usd === null ? null : Number(row.cost_usd),
    tokens: row.tokens === null ? null : Number(row.tokens),
    sourceBlobUrl: row.source_blob_url ? String(row.source_blob_url) : null,
    failureReason: row.failure_reason ? String(row.failure_reason) : null,
    createdAt: new Date(String(row.created_at)).toISOString(),
  };
}

function gpuFromRow(row: Record<string, unknown>): GpuSample {
  return {
    id: String(row.id),
    runId: String(row.run_id),
    timestamp: new Date(String(row.timestamp)).toISOString(),
    gpuUtil: Number(row.gpu_util),
    memUtil: Number(row.mem_util),
    memUsedMib: Number(row.mem_used_mib),
    memTotalMib: Number(row.mem_total_mib),
  };
}

function artifactFromRow(row: Record<string, unknown>): ArtifactRecord {
  return {
    id: String(row.id),
    runId: String(row.run_id),
    trialId: row.trial_id ? String(row.trial_id) : null,
    kind: String(row.kind),
    filename: String(row.filename),
    blobUrl: String(row.blob_url),
    sizeBytes: row.size_bytes === null ? null : Number(row.size_bytes),
    createdAt: new Date(String(row.created_at)).toISOString(),
  };
}

async function ensureSchema() {
  const sql = getSql();
  if (!sql || schemaReady) {
    return;
  }

  await sql`
    create table if not exists runs (
      id text primary key,
      name text not null,
      runner text,
      gpu text,
      status text not null default 'queued',
      started_at timestamptz not null default now(),
      ended_at timestamptz,
      hud_job_url text,
      git_sha text,
      policy text,
      controller text,
      notes text
    )
  `;
  await sql`
    create table if not exists trials (
      id text primary key,
      run_id text not null references runs(id) on delete cascade,
      op text not null,
      trial_index integer not null,
      candidate text not null,
      accepted boolean not null default false,
      correct boolean not null default false,
      reward double precision not null default 0,
      speedup double precision,
      latency_ms double precision,
      best_speedup_after double precision,
      best_latency_after double precision,
      cost_usd double precision,
      tokens integer,
      source_blob_url text,
      failure_reason text,
      created_at timestamptz not null default now()
    )
  `;
  await sql`
    create table if not exists gpu_samples (
      id text primary key,
      run_id text not null references runs(id) on delete cascade,
      timestamp timestamptz not null default now(),
      gpu_util double precision not null,
      mem_util double precision not null,
      mem_used_mib double precision not null,
      mem_total_mib double precision not null
    )
  `;
  await sql`
    create table if not exists artifacts (
      id text primary key,
      run_id text not null references runs(id) on delete cascade,
      trial_id text references trials(id) on delete set null,
      kind text not null,
      filename text not null,
      blob_url text not null,
      size_bytes integer,
      created_at timestamptz not null default now()
    )
  `;
  await sql`create index if not exists trials_run_created_idx on trials(run_id, created_at)`;
  await sql`create index if not exists trials_op_created_idx on trials(op, created_at)`;
  await sql`create index if not exists gpu_samples_run_time_idx on gpu_samples(run_id, timestamp)`;
  schemaReady = true;
}

export async function listRuns(): Promise<RunRecord[]> {
  const sql = getSql();
  if (!sql) {
    return seedRuns;
  }
  await ensureSchema();
  const rows = await sql`select * from runs order by started_at desc limit 100`;
  return rows.map(runFromRow);
}

export async function listTrials(): Promise<TrialRecord[]> {
  const sql = getSql();
  if (!sql) {
    return seedTrials;
  }
  await ensureSchema();
  const rows = await sql`select * from trials order by created_at asc`;
  return rows.map(trialFromRow);
}

export async function getRun(id: string) {
  const sql = getSql();
  if (!sql) {
    const run = seedRuns.find((item) => item.id === id);
    if (!run) return null;
    return summarizeRun(
      run,
      seedTrials.filter((trial) => trial.runId === id),
      seedGpuSamples.filter((sample) => sample.runId === id),
      seedArtifacts.filter((artifact) => artifact.runId === id),
    );
  }
  await ensureSchema();
  const runRows = await sql`select * from runs where id = ${id} limit 1`;
  if (!runRows[0]) return null;
  const [trialRows, gpuRows, artifactRows] = await Promise.all([
    sql`select * from trials where run_id = ${id} order by created_at asc`,
    sql`select * from gpu_samples where run_id = ${id} order by timestamp asc limit 2000`,
    sql`select * from artifacts where run_id = ${id} order by created_at desc`,
  ]);
  return summarizeRun(
    runFromRow(runRows[0]),
    trialRows.map(trialFromRow),
    gpuRows.map(gpuFromRow),
    artifactRows.map(artifactFromRow),
  );
}

export async function getOp(op: string) {
  const sql = getSql();
  if (!sql) {
    const trials = seedTrials.filter((trial) => trial.op === op);
    return { op, trials };
  }
  await ensureSchema();
  const rows = await sql`select * from trials where op = ${op} order by created_at asc`;
  return { op, trials: rows.map(trialFromRow) };
}

export async function upsertRun(input: Partial<RunRecord> & { id?: string; name?: string }) {
  const sql = getSql();
  if (!sql) throw new Error("DATABASE_URL is required for ingestion");
  await ensureSchema();
  const id = input.id || crypto.randomUUID();
  const name = input.name || id;
  const status = input.status || "running";
  const startedAt = input.startedAt ? new Date(input.startedAt) : new Date();
  const endedAt = input.endedAt ? new Date(input.endedAt) : null;
  const rows = await sql`
    insert into runs (
      id, name, runner, gpu, status, started_at, ended_at, hud_job_url, git_sha, policy, controller, notes
    ) values (
      ${id}, ${name}, ${input.runner ?? null}, ${input.gpu ?? null}, ${status}, ${startedAt},
      ${endedAt}, ${input.hudJobUrl ?? null}, ${input.gitSha ?? null}, ${input.policy ?? null},
      ${input.controller ?? null}, ${input.notes ?? null}
    )
    on conflict (id) do update set
      name = excluded.name,
      runner = excluded.runner,
      gpu = excluded.gpu,
      status = excluded.status,
      ended_at = excluded.ended_at,
      hud_job_url = excluded.hud_job_url,
      git_sha = excluded.git_sha,
      policy = excluded.policy,
      controller = excluded.controller,
      notes = excluded.notes
    returning *
  `;
  return runFromRow(rows[0]);
}

export async function insertTrial(input: Omit<TrialRecord, "id" | "createdAt"> & { id?: string; createdAt?: string }) {
  const sql = getSql();
  if (!sql) throw new Error("DATABASE_URL is required for ingestion");
  await ensureSchema();
  const rows = await sql`
    insert into trials (
      id, run_id, op, trial_index, candidate, accepted, correct, reward, speedup, latency_ms,
      best_speedup_after, best_latency_after, cost_usd, tokens, source_blob_url, failure_reason, created_at
    ) values (
      ${input.id || crypto.randomUUID()}, ${input.runId}, ${input.op}, ${input.trialIndex}, ${input.candidate},
      ${input.accepted}, ${input.correct}, ${input.reward}, ${input.speedup}, ${input.latencyMs},
      ${input.bestSpeedupAfter}, ${input.bestLatencyAfter}, ${input.costUsd}, ${input.tokens},
      ${input.sourceBlobUrl}, ${input.failureReason}, ${input.createdAt ? new Date(input.createdAt) : new Date()}
    )
    on conflict (id) do update set
      accepted = excluded.accepted,
      correct = excluded.correct,
      reward = excluded.reward,
      speedup = excluded.speedup,
      latency_ms = excluded.latency_ms,
      best_speedup_after = excluded.best_speedup_after,
      best_latency_after = excluded.best_latency_after,
      source_blob_url = excluded.source_blob_url,
      failure_reason = excluded.failure_reason
    returning *
  `;
  return trialFromRow(rows[0]);
}

export async function insertGpuSamples(samples: Array<Omit<GpuSample, "id"> & { id?: string }>) {
  const sql = getSql();
  if (!sql) throw new Error("DATABASE_URL is required for ingestion");
  await ensureSchema();
  const inserted: GpuSample[] = [];
  for (const sample of samples) {
    const rows = await sql`
      insert into gpu_samples (id, run_id, timestamp, gpu_util, mem_util, mem_used_mib, mem_total_mib)
      values (
        ${sample.id || crypto.randomUUID()}, ${sample.runId}, ${new Date(sample.timestamp)},
        ${sample.gpuUtil}, ${sample.memUtil}, ${sample.memUsedMib}, ${sample.memTotalMib}
      )
      on conflict (id) do update set
        gpu_util = excluded.gpu_util,
        mem_util = excluded.mem_util,
        mem_used_mib = excluded.mem_used_mib,
        mem_total_mib = excluded.mem_total_mib
      returning *
    `;
    inserted.push(gpuFromRow(rows[0]));
  }
  return inserted;
}

export async function insertArtifact(input: Omit<ArtifactRecord, "id" | "createdAt"> & { id?: string; createdAt?: string }) {
  const sql = getSql();
  if (!sql) throw new Error("DATABASE_URL is required for ingestion");
  await ensureSchema();
  const rows = await sql`
    insert into artifacts (id, run_id, trial_id, kind, filename, blob_url, size_bytes, created_at)
    values (
      ${input.id || crypto.randomUUID()}, ${input.runId}, ${input.trialId}, ${input.kind}, ${input.filename},
      ${input.blobUrl}, ${input.sizeBytes}, ${input.createdAt ? new Date(input.createdAt) : new Date()}
    )
    on conflict (id) do update set
      kind = excluded.kind,
      filename = excluded.filename,
      blob_url = excluded.blob_url,
      size_bytes = excluded.size_bytes
    returning *
  `;
  return artifactFromRow(rows[0]);
}
