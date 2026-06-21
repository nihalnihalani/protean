import { getOp } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: Promise<{ op: string }> }) {
  const { op } = await params;
  const detail = await getOp(decodeURIComponent(op));
  return Response.json(detail);
}
