# Protean Vercel Dashboard

The web dashboard lives in `apps/web` and is intended to be deployed as the
Vercel project root directory. Spark and Modal remain the GPU execution plane;
Vercel stores and displays metrics, trials, source artifacts, and HUD links.

## Environment

Required for production ingestion:

```bash
DATABASE_URL=...
BLOB_READ_WRITE_TOKEN=...
PROTEAN_INGEST_TOKEN=...
NEXT_PUBLIC_SITE_URL=https://your-vercel-domain
```

Public pages render seeded demo data when `DATABASE_URL` is not set. Ingestion
endpoints always require `PROTEAN_INGEST_TOKEN` and a database.

## Deploy

Create a Vercel project from `nihalnihalani/protean` with root directory:

```text
apps/web
```

Then add Vercel Postgres/Neon and Vercel Blob from the project Storage tab, set
the environment variables above, and deploy.

Local verification:

```bash
cd apps/web
npm install
npm run test
npm run lint
npm run typecheck
npm run build
```

## Ingest A Run

From Spark or Modal after an optimizer run:

```bash
export PROTEAN_DASHBOARD_URL=https://your-vercel-domain
export PROTEAN_INGEST_TOKEN=...
python scripts/ingest_dashboard_run.py runs/protean-overnight-2026-06-21 \
  --runner spark \
  --gpu "GB10" \
  --status completed
```

The script posts:

- run metadata to `/api/ingest/runs`
- trial rows to `/api/ingest/trials`
- GPU utilization samples to `/api/ingest/gpu-samples`
- small candidate source files to `/api/ingest/artifacts`

The dashboard keeps HUD as an external proof surface by storing `hud_job_url`
when the optimizer summary includes it.
