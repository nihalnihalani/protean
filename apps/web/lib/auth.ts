import type { NextRequest } from "next/server";

export function isAuthorized(request: NextRequest): boolean {
  const expected = process.env.PROTEAN_INGEST_TOKEN;
  if (!expected) {
    return false;
  }
  const bearer = request.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
  const headerToken = request.headers.get("x-protean-token");
  return bearer === expected || headerToken === expected;
}

export function unauthorized() {
  return Response.json({ error: "unauthorized" }, { status: 401 });
}
