from train.calibrate import calibrate


def test_calibration_skip_escape_hatch(monkeypatch):
    monkeypatch.setenv("PROTEAN_SKIP_CALIBRATION", "1")
    assert calibrate(tasks=[{"op": "elementwise_add_relu"}], base_model_runner=lambda _: ["bad"]) == "SKIPPED"
