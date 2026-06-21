import Link from "next/link";

export default function NotFound() {
  return (
    <main className="page">
      <section className="card">
        <div className="card-body">
          <div className="eyebrow">Not found</div>
          <h1>That Protean view does not exist.</h1>
          <p className="lede">The run or op may not have been ingested yet.</p>
          <Link className="button" href="/">
            Back to live dashboard
          </Link>
        </div>
      </section>
    </main>
  );
}
