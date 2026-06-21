import assert from "node:assert/strict";
import test from "node:test";
import { isAuthorized } from "../lib/auth";
import type { NextRequest } from "next/server";

function request(headers: Record<string, string>): NextRequest {
  return { headers: new Headers(headers) } as NextRequest;
}

test("ingest auth rejects when the server token is unset", () => {
  delete process.env.PROTEAN_INGEST_TOKEN;
  assert.equal(isAuthorized(request({ authorization: "Bearer anything" })), false);
});

test("ingest auth accepts bearer token", () => {
  process.env.PROTEAN_INGEST_TOKEN = "secret";
  assert.equal(isAuthorized(request({ authorization: "Bearer secret" })), true);
});

test("ingest auth accepts x-protean-token and rejects wrong values", () => {
  process.env.PROTEAN_INGEST_TOKEN = "secret";
  assert.equal(isAuthorized(request({ "x-protean-token": "secret" })), true);
  assert.equal(isAuthorized(request({ "x-protean-token": "wrong" })), false);
});
