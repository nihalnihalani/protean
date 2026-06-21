import type { Metadata } from "next";
import Link from "next/link";
import { Activity, Gauge } from "lucide-react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Protean Dashboard",
  description: "Live GPU-kernel improvement dashboard for Protean.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <header className="topbar">
            <Link href="/" className="brand">
              <span className="brand-mark">
                <Activity size={18} />
              </span>
              <span>
                <span className="brand-title">Protean</span>
                <span className="brand-subtitle">Kernel improvement control plane</span>
              </span>
            </Link>
            <nav className="nav" aria-label="Primary navigation">
              <Link href="/">Live</Link>
              <Link href="/runs">Runs</Link>
              <Link href="/ops/rmsnorm">Ops</Link>
              <a href="https://www.hud.ai/tasksets/6d2feb10-b23c-4928-a1f9-e8b53db364d7" target="_blank">
                HUD
              </a>
              <span className="badge">
                <Gauge size={14} /> public read
              </span>
            </nav>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
