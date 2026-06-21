import types

from scripts import run_hud_optimizer_agent as agent
from protean.task_catalog import OPS


def test_hud_optimizer_agent_reuses_one_session_for_all_ops(monkeypatch, tmp_path):
    calls = {"runs": []}
    session = types.SimpleNamespace(
        job_url="https://hud.ai/jobs/session-1",
        name="protean-live-test",
    )

    monkeypatch.setattr(agent, "assert_hud_auth", lambda: None)
    monkeypatch.setattr(agent, "start_hud_stream_session", lambda **kwargs: session)

    def fake_run_optimization(**kwargs):
        calls["runs"].append(kwargs)
        return {
            "op": kwargs["op"],
            "trials": 1,
            "accepted": 0,
            "hud_job_url": session.job_url,
        }

    monkeypatch.setattr(agent, "run_optimization", fake_run_optimization)

    result = agent.main(
        [
            "--policy",
            "local",
            "--all-ops",
            "--max-rounds",
            "1",
            "--group",
            "2",
            "--job-name",
            "protean-live-test",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert [call["op"] for call in calls["runs"]] == [op.name for op in OPS]
    assert all(call["hud_session"] is session for call in calls["runs"])
    assert all(call["stream_hud"] is True for call in calls["runs"])
    assert all(call["hud_group"] == 2 for call in calls["runs"])
