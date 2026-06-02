"""Tests for the throughput/RPS sizing logic.

Most tests are pure arithmetic over `workload.py` (no model, no I/O) so they run
in milliseconds. One integration test exercises `recommend_workload` end-to-end
against the shipped predictor + DNS catalogue and is skipped if either is absent.

Run:  python -m pytest tests/ -q      (or:  make test-workload)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import workload as wl  # noqa: E402


# ------------------------------------------------------------ stream parsing

def test_parse_stream_arg_registered():
    s = wl.parse_stream_arg("model=yolov8n.pt,img=480,frames=3000")
    assert s.name == "yolov8n.pt"
    assert s.predict_model == "yolov8n.pt"
    assert s.img_size == 480 and s.frames == 3000
    assert s.needs_prediction is True
    assert s.custom_per_frame_latency() is None


def test_parse_stream_arg_custom_latency_bypasses_predictor():
    s = wl.parse_stream_arg("name=RoMa,img=560,frames=500,fixed_latency_ms=120")
    assert s.needs_prediction is False
    assert s.predict_model is None
    assert s.custom_per_frame_latency() == pytest.approx(0.120)


def test_parse_stream_arg_fps_to_latency():
    s = wl.parse_stream_arg("name=X,frames=10,fps=50")
    assert s.custom_per_frame_latency() == pytest.approx(0.02)


def test_proxy_is_used_for_prediction():
    s = wl.stream_from_dict({"name": "yolo26l", "proxy": "yolov8l.pt",
                             "img_size": 640, "frames": 48000})
    assert s.is_proxied and s.needs_prediction
    assert s.predict_model == "yolov8l.pt"


def test_frames_from_rate_times_duration():
    s = wl.stream_from_dict({"name": "v8n", "model": "yolov8n.pt",
                             "img": 480, "rate_fps": 25, "duration_s": 120})
    assert s.frames == 3000


def test_pinned_batch_limits_sweep():
    s = wl.parse_stream_arg("model=yolov8n.pt,img=480,frames=10,batch=8")
    assert s.candidate_batches() == (8,)


def test_validate_rejects_modelless_stream():
    with pytest.raises(ValueError):
        wl.stream_from_dict({"name": "bad", "img_size": 640, "frames": 100})


def test_validate_rejects_nonpositive_frames():
    with pytest.raises(ValueError):
        wl.stream_from_dict({"name": "bad", "model": "yolov8n.pt", "frames": 0})


# ------------------------------------------------------------ batch / vram

def test_choose_best_batch_minimises_per_frame():
    # latency grows sublinearly with batch → larger batch wins on per-frame.
    lat = {1: 0.0157, 2: 0.0157, 5: 0.0245, 8: 0.0314, 10: 0.0375, 16: 0.0356}
    batch, per_frame = wl.choose_best_batch(lat)
    assert batch == 16
    assert per_frame == pytest.approx(0.0356 / 16)


def test_choose_best_batch_respects_feasible_set():
    lat = {1: 0.0157, 2: 0.0157, 16: 0.0356}
    batch, per_frame = wl.choose_best_batch(lat, feasible_batches={1, 2})
    assert batch == 2  # 16 excluded by VRAM


def test_choose_best_batch_none_when_nothing_feasible():
    assert wl.choose_best_batch({1: 0.01}, feasible_batches=set()) is None


def test_fits_vram_rejects_oversized_batch():
    # 16 GB card, a model whose activations blow past it at large batch.
    params, peak_act, ref = 40e6, 200e6, 640
    assert wl.fits_vram(params, peak_act, ref, 640, 1, 16.0) is True
    assert wl.fits_vram(params, peak_act, ref, 640, 256, 16.0) is False


def test_fits_vram_unknown_memory_allows():
    assert wl.fits_vram(40e6, 200e6, 640, 640, 8, float("nan")) is True


def test_vram_scales_with_resolution_and_batch():
    base = wl.vram_bytes(10e6, 100e6, 640, 640, 1)
    assert wl.vram_bytes(10e6, 100e6, 640, 1280, 1) > base   # 2x res ~ 4x act
    assert wl.vram_bytes(10e6, 100e6, 640, 640, 4) > base    # batch scales act


# ------------------------------------------------------------ aggregate

def _streams_for_aggregate():
    return [
        wl.stream_from_dict({"name": "light", "model": "yolov8n.pt",
                             "img": 480, "frames": 1000}),
        wl.stream_from_dict({"name": "heavy", "model": "yolov8l.pt",
                             "img": 640, "frames": 9000}),
    ]


def test_aggregate_gpu_seconds_and_single_gpu():
    streams = _streams_for_aggregate()
    # 1000×0.001 = 1s ; 9000×0.01 = 90s ; total 91s /0.9 util ≈ 101.1s ≤ 180
    per_frame = {"light": 0.001, "heavy": 0.010}
    r = wl.aggregate(streams, per_frame, {"light": 8, "heavy": 16},
                     deadline_s=180, util=0.9, max_gpus=4)
    assert r.feasible and r.gpus_needed == 1
    assert r.required_gpu_seconds == pytest.approx(91 / 0.9)
    assert r.bottleneck_stream == "heavy"
    assert 0 < r.headroom_frac < 1


def test_aggregate_shards_to_multiple_gpus():
    streams = _streams_for_aggregate()
    # heavy 9000×0.05 = 450s ; +1s ; 451/0.9 ≈ 501s → ceil(501/180) = 3 GPUs
    r = wl.aggregate(streams, {"light": 0.001, "heavy": 0.050}, {},
                     deadline_s=180, util=0.9, max_gpus=4)
    assert r.gpus_needed == math.ceil((451 / 0.9) / 180) == 3
    assert r.feasible is True


def test_aggregate_infeasible_when_exceeds_max_gpus():
    streams = _streams_for_aggregate()
    r = wl.aggregate(streams, {"light": 0.001, "heavy": 0.050}, {},
                     deadline_s=180, util=0.9, max_gpus=2)
    assert r.feasible is False and r.gpus_needed == 3
    assert "max_gpus" in r.reason


def test_aggregate_unserved_stream_is_infeasible():
    streams = _streams_for_aggregate()
    r = wl.aggregate(streams, {"light": 0.001, "heavy": None}, {},
                     deadline_s=180)
    assert r.feasible is False and r.gpus_needed == 0
    assert any(not s.feasible for s in r.streams)


def test_aggregate_custom_stream_counts_without_prediction():
    s = wl.parse_stream_arg("name=RoMa,frames=500,fixed_latency_ms=120")
    r = wl.aggregate([s], {"RoMa": s.custom_per_frame_latency()}, {},
                     deadline_s=180, util=1.0, max_gpus=4)
    assert r.required_gpu_seconds == pytest.approx(60.0)  # 500 × 0.12
    assert r.feasible and r.gpus_needed == 1


# ------------------------------------------------------------ load workload

def test_load_example_workload():
    path = PROJECT_ROOT / "examples" / "workload_3min.json"
    deadline, streams = wl.load_workload(path)
    assert deadline == 180
    names = {s.name for s in streams}
    assert {"yolov8n", "yolo11l", "rtdetr-l"} <= names
    heavy = next(s for s in streams if s.name == "yolo11l")
    assert heavy.needs_prediction and heavy.frames == 48000
    assert heavy.predict_model == "yolo11l.pt"
    light = next(s for s in streams if s.name == "yolov8n")
    assert light.frames == 3000  # 25 fps x 120 s


# ------------------------------------------------------------ integration

@pytest.mark.integration
def test_recommend_workload_end_to_end():
    """Smoke test the full pipeline on the shipped predictor + catalogue."""
    weights = PROJECT_ROOT / "data_base" / "reg_weights_base" / "catboost_model.cbm"
    dns = PROJECT_ROOT / "specs" / "dns_pcs.json"
    if not weights.is_file() or not dns.is_file():
        pytest.skip("predictor weights or DNS catalogue not present")

    import recommend_workload as rw

    deadline, streams = wl.load_workload(
        PROJECT_ROOT / "examples" / "workload_3min.json")
    # Generous budget + max_gpus so something is guaranteed feasible.
    df = rw.recommend_workload(streams, deadline_s=deadline,
                               max_budget_rub=2_000_000, max_gpus=8, top_k=5)
    assert not df.empty
    top = df.iloc[0]
    assert top["gpus_needed"] >= 1
    assert top["total_price_rub"] > 0
    # The 48k-frame heavy stream must dominate the GPU-budget.
    result = top["_result"]
    assert result.bottleneck_stream == "yolo11l"
    assert result.feasible
    # required GPU-seconds is the util-adjusted sum of per-stream contributions.
    assert result.required_gpu_seconds == pytest.approx(
        sum(s.gpu_seconds for s in result.streams) / result.util, rel=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
