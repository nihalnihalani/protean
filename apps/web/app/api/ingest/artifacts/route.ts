import { isAuthorized, unauthorized } from "@/lib/auth";
import { insertArtifact } from "@/lib/db";
import type { NextRequest } from "next/server";

export const runtime = "nodejs";

async function maybeUploadBlob(body: Record<string, unknown>) {
  if (!body.content && !body.contentBase64) {
    return {
      blobUrl: body.blobUrl ? String(body.blobUrl) : null,
      sizeBytes: body.sizeBytes == null ? null : Number(body.sizeBytes),
    };
  }
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    throw new Error("BLOB_READ_WRITE_TOKEN is required when artifact content is included");
  }
  const { put } = await import("@vercel/blob");
  const filename = String(body.filename || `artifact-${Date.now()}.txt`);
  const bytes = body.contentBase64
    ? Buffer.from(String(body.contentBase64), "base64")
    : Buffer.from(String(body.content));
  const blob = await put(`protean/${Date.now()}-${filename}`, bytes, {
    access: "public",
    token: process.env.BLOB_READ_WRITE_TOKEN,
  });
  return { blobUrl: blob.url, sizeBytes: bytes.byteLength };
}

export async function POST(request: NextRequest) {
  if (!isAuthorized(request)) {
    return unauthorized();
  }
  try {
    const body = await request.json();
    const upload = await maybeUploadBlob(body);
    if (!upload.blobUrl) {
      throw new Error("blobUrl or content is required");
    }
    const artifact = await insertArtifact({
      id: body.id,
      runId: body.runId,
      trialId: body.trialId ?? null,
      kind: body.kind,
      filename: body.filename,
      blobUrl: upload.blobUrl,
      sizeBytes: upload.sizeBytes,
      createdAt: body.createdAt,
    });
    return Response.json({ artifact });
  } catch (error) {
    return Response.json({ error: "artifact_ingest_failed", message: String(error) }, { status: 500 });
  }
}
