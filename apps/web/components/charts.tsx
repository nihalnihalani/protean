"use client";

import { useSyncExternalStore } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GpuSample, TrialRecord } from "@/lib/types";

function ChartFrame({ children, height }: { children: React.ReactNode; height: number }) {
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
  return (
    <div style={{ width: "100%", height }}>
      {mounted ? children : <div className="muted" style={{ padding: 12 }}>Preparing chart...</div>}
    </div>
  );
}

function trialPoint(trial: TrialRecord) {
  return {
    name: `#${trial.trialIndex}`,
    reward: trial.reward,
    speedup: trial.bestSpeedupAfter ?? trial.speedup ?? 0,
    latency: trial.bestLatencyAfter ?? trial.latencyMs ?? 0,
    accepted: trial.accepted ? 1 : 0,
  };
}

export function ImprovementChart({ trials }: { trials: TrialRecord[] }) {
  const data = trials.map(trialPoint);
  return (
    <ChartFrame height={300}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#263241" strokeDasharray="4 4" />
          <XAxis dataKey="name" stroke="#93a3b8" tickLine={false} />
          <YAxis yAxisId="left" stroke="#93a3b8" tickLine={false} width={42} />
          <YAxis yAxisId="right" orientation="right" stroke="#93a3b8" tickLine={false} width={42} />
          <Tooltip
            contentStyle={{ background: "#10151d", border: "1px solid #263241", borderRadius: 8 }}
            labelStyle={{ color: "#eef4ff" }}
          />
          <Line yAxisId="left" type="monotone" dataKey="speedup" stroke="#34d399" strokeWidth={2} dot={false} />
          <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#60a5fa" strokeWidth={2} dot={false} />
          <Line yAxisId="left" type="stepAfter" dataKey="accepted" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function RewardChart({ trials }: { trials: TrialRecord[] }) {
  const data = trials.map(trialPoint);
  return (
    <ChartFrame height={240}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#263241" strokeDasharray="4 4" />
          <XAxis dataKey="name" stroke="#93a3b8" tickLine={false} />
          <YAxis stroke="#93a3b8" tickLine={false} width={42} />
          <Tooltip contentStyle={{ background: "#10151d", border: "1px solid #263241", borderRadius: 8 }} />
          <Area type="monotone" dataKey="reward" stroke="#34d399" fill="#34d399" fillOpacity={0.18} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function GpuChart({ samples }: { samples: GpuSample[] }) {
  const data = samples.map((sample) => ({
    time: new Date(sample.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    gpu: sample.gpuUtil,
    memory: sample.memUtil,
  }));
  return (
    <ChartFrame height={220}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#263241" strokeDasharray="4 4" />
          <XAxis dataKey="time" stroke="#93a3b8" tickLine={false} />
          <YAxis stroke="#93a3b8" tickLine={false} width={42} domain={[0, 100]} />
          <Tooltip contentStyle={{ background: "#10151d", border: "1px solid #263241", borderRadius: 8 }} />
          <Area type="monotone" dataKey="gpu" stroke="#34d399" fill="#34d399" fillOpacity={0.18} />
          <Area type="monotone" dataKey="memory" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.12} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}
