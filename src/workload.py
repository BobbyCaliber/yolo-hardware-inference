"""Throughput / RPS workload model — the math behind `recommend_workload.py`.

The single-point predictor answers "how long is one forward pass?". A *workload*
is several models running **concurrently** on the same hardware, each with its
own image size and a number of frames to chew through before a shared deadline.

Concurrency model — **additive GPU-seconds**. A GPU is one resource: parallel
processes time-share it, they don't multiply its FLOPs. So the GPU must spend

    Σ_streams ( frames_i × best_per_frame_latency_i )  /  util   seconds

of compute, and that has to fit inside `deadline`. `util` (default 0.9) is a
duty-cycle haircut keeping the estimate slightly conservative — you never get a
100 % GPU duty cycle across process hand-offs. If the work doesn't fit one card,
`gpus_needed = ceil(required_gpu_seconds / deadline)` identical GPUs share it
(frames of any stream can be split across cards).

`best_per_frame_latency_i` is the *lowest* per-frame latency reachable by
batching: larger batches amortise launch overhead and raise throughput until
they stop fitting in VRAM, so we sweep candidate batch sizes and keep the best
feasible one.

This module is deliberately I/O- and model-free (stdlib only) so the arithmetic
is trivially unit-testable. `recommend_workload.py` supplies the predicted
latency tables; everything here is pure functions over plain numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Batch sizes the dataset was swept over (src/run_benchmark.py); also the search
# grid we use to find the throughput-optimal batch per stream.
CANDIDATE_BATCHES: tuple[int, ...] = (1, 2, 4, 5, 6, 8, 10, 16)

DEFAULT_DEADLINE_S = 180.0
DEFAULT_UTIL = 0.9
DEFAULT_MAX_GPUS = 4
DEFAULT_RAM_GB = 16

# VRAM headroom: leave ~15 % for CUDA context, framework caches, fragmentation.
VRAM_USABLE_FRAC = 0.85
BYTES_PER_PARAM = 4  # fp32 weights
_BYTES_PER_GB = 1024 ** 3


# --------------------------------------------------------------------- stream

@dataclass
class Stream:
    """One concurrent inference stream in a workload.

    Exactly one source of per-frame cost must be resolvable:
      * a registered `model` (predicted by the CatBoost model), or
      * a `proxy` registered model standing in for an unknown one, or
      * a user-measured `fixed_latency_s` / `fixed_fps` (custom model).
    """

    name: str
    frames: int
    img_size: int = 640
    model: str | None = None
    proxy: str | None = None
    fixed_latency_s: float | None = None
    fixed_fps: float | None = None
    batch: int | None = None  # pin a batch instead of sweeping

    @property
    def predict_model(self) -> str | None:
        """Registered model name to feed the predictor (own model or proxy)."""
        return self.model or self.proxy

    def custom_per_frame_latency(self) -> float | None:
        """Per-frame seconds from a user-supplied latency/fps, else None."""
        if self.fixed_latency_s is not None:
            return float(self.fixed_latency_s)
        if self.fixed_fps:
            return 1.0 / float(self.fixed_fps)
        return None

    @property
    def needs_prediction(self) -> bool:
        """True when we must call the predictor (no user-supplied latency)."""
        return self.custom_per_frame_latency() is None and self.predict_model is not None

    @property
    def is_proxied(self) -> bool:
        return self.model is None and self.proxy is not None

    def candidate_batches(self) -> tuple[int, ...]:
        return (self.batch,) if self.batch else CANDIDATE_BATCHES

    def validate(self) -> None:
        if self.frames is None or self.frames <= 0:
            raise ValueError(f"stream {self.name!r}: 'frames' must be a positive int")
        if self.custom_per_frame_latency() is None and self.predict_model is None:
            raise ValueError(
                f"stream {self.name!r}: give a registered 'model'/'proxy', or a "
                "'fixed_latency_ms'/'fixed_fps' for a custom model"
            )
        if self.custom_per_frame_latency() is not None and self.custom_per_frame_latency() <= 0:
            raise ValueError(f"stream {self.name!r}: fixed latency/fps must be positive")


# --------------------------------------------------------------------- loaders

def _frames_from_dict(d: dict[str, Any]) -> int:
    """`frames`, or `rate_fps × duration_s` (the way real captures are sized)."""
    if d.get("frames") is not None:
        return int(d["frames"])
    rate = d.get("rate_fps")
    dur = d.get("duration_s")
    if rate is not None and dur is not None:
        return int(math.ceil(float(rate) * float(dur)))
    raise ValueError(f"stream {d.get('name', d)!r}: need 'frames' or 'rate_fps'+'duration_s'")


def stream_from_dict(d: dict[str, Any]) -> Stream:
    """Build a Stream from a dict, accepting friendly aliases.

    Aliases: `img`→img_size, `fixed_latency_ms`/`latency_ms`→fixed_latency_s,
    `fps`→fixed_fps, `rate_fps`+`duration_s`→frames.
    """
    img = d.get("img_size", d.get("img", 640))
    lat_ms = d.get("fixed_latency_ms", d.get("latency_ms"))
    fixed_latency_s = d.get("fixed_latency_s")
    if fixed_latency_s is None and lat_ms is not None:
        fixed_latency_s = float(lat_ms) / 1000.0
    name = d.get("name") or d.get("model") or d.get("proxy") or "stream"
    s = Stream(
        name=str(name),
        frames=_frames_from_dict(d),
        img_size=int(img),
        model=d.get("model"),
        proxy=d.get("proxy"),
        fixed_latency_s=fixed_latency_s,
        fixed_fps=d.get("fixed_fps", d.get("fps")),
        batch=int(d["batch"]) if d.get("batch") else None,
    )
    s.validate()
    return s


def parse_stream_arg(arg: str) -> Stream:
    """Parse a CLI `--stream "model=yolov8n.pt,img=480,frames=3000"` token."""
    d: dict[str, Any] = {}
    for piece in arg.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"bad --stream fragment {piece!r} (expected key=value)")
        k, v = piece.split("=", 1)
        d[k.strip()] = v.strip()
    # numeric coercion for the numeric keys (strings are fine for model/proxy/name)
    for k in ("img_size", "img", "frames", "batch", "duration_s"):
        if k in d:
            d[k] = float(d[k]) if k == "duration_s" else int(float(d[k]))
    for k in ("fixed_latency_ms", "latency_ms", "fixed_latency_s", "fixed_fps", "fps", "rate_fps"):
        if k in d:
            d[k] = float(d[k])
    return stream_from_dict(d)


def load_workload(src: str | Path | dict[str, Any]) -> tuple[float, list[Stream]]:
    """Load `(deadline_s, streams)` from a JSON file/path or an in-memory dict."""
    if isinstance(src, (str, Path)):
        payload = json.loads(Path(src).read_text(encoding="utf-8"))
    else:
        payload = src
    deadline = float(payload.get("deadline_s", DEFAULT_DEADLINE_S))
    streams = [stream_from_dict(d) for d in payload["streams"]]
    if not streams:
        raise ValueError("workload has no streams")
    return deadline, streams


# ------------------------------------------------------------------- vram + batch

def vram_bytes(params: float, peak_activation: float, img_size_ref: float,
               img_size: int, batch: int) -> float:
    """Rough peak VRAM for a forward pass: fp32 weights + scaled activations.

    `peak_activation` is the working-set peak measured at `img_size_ref`
    (batch 1); activations scale ~quadratically with resolution and linearly
    with batch.
    """
    scale = (float(img_size) / float(img_size_ref)) ** 2 if img_size_ref else 1.0
    return params * BYTES_PER_PARAM + peak_activation * scale * batch


def fits_vram(params: float, peak_activation: float, img_size_ref: float,
              img_size: int, batch: int, gpu_mem_gb: float) -> bool:
    if not gpu_mem_gb or gpu_mem_gb <= 0 or math.isnan(gpu_mem_gb):
        return True  # unknown memory → don't block (predictor still gates latency)
    budget = gpu_mem_gb * _BYTES_PER_GB * VRAM_USABLE_FRAC
    return vram_bytes(params, peak_activation, img_size_ref, img_size, batch) <= budget


def choose_best_batch(latency_by_batch: dict[int, float],
                      feasible_batches: set[int] | None = None
                      ) -> tuple[int, float] | None:
    """Pick the batch minimising per-frame latency (`t / batch`).

    `latency_by_batch` maps batch → predicted per-*batch* seconds. Only batches
    in `feasible_batches` (VRAM-fitting) are considered; None means all fit.
    Returns `(batch, per_frame_latency_s)` or None if nothing fits.
    """
    best: tuple[int, float] | None = None
    for batch, t in latency_by_batch.items():
        if feasible_batches is not None and batch not in feasible_batches:
            continue
        if t is None or t <= 0:
            continue
        per_frame = t / batch
        if best is None or per_frame < best[1]:
            best = (batch, per_frame)
    return best


# ------------------------------------------------------------------- aggregate

@dataclass
class StreamResult:
    name: str
    frames: int
    img_size: int
    model: str | None
    is_custom: bool
    batch: int | None
    per_frame_latency_s: float | None
    fps_per_gpu: float | None       # sustained throughput one GPU gives this stream
    required_fps: float | None      # frames / deadline (what the deadline demands)
    gpu_seconds: float | None       # frames × per_frame_latency
    feasible: bool                  # a batch fit VRAM / latency was resolved


@dataclass
class WorkloadResult:
    deadline_s: float
    util: float
    max_gpus: int
    required_gpu_seconds: float          # already util-adjusted
    gpus_needed: int
    feasible: bool
    headroom_frac: float                 # spare capacity at gpus_needed (0..1)
    bottleneck_stream: str | None
    reason: str                          # why infeasible, or "ok"
    streams: list[StreamResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def aggregate(streams: list[Stream],
              per_frame_latency: dict[str, float | None],
              chosen_batch: dict[str, int | None] | None = None,
              *, deadline_s: float = DEFAULT_DEADLINE_S,
              util: float = DEFAULT_UTIL,
              max_gpus: int = DEFAULT_MAX_GPUS) -> WorkloadResult:
    """Combine per-stream best per-frame latencies into a sizing verdict.

    `per_frame_latency[name]` is the best per-frame seconds for that stream
    (None ⇒ couldn't be served — e.g. no batch fit VRAM). `chosen_batch[name]`
    is the batch that produced it (None for custom streams).
    """
    chosen_batch = chosen_batch or {}
    stream_results: list[StreamResult] = []
    total_raw = 0.0
    any_unserved = False
    bottleneck: str | None = None
    bottleneck_secs = -1.0

    for s in streams:
        lat = per_frame_latency.get(s.name)
        if lat is None or lat <= 0:
            any_unserved = True
            stream_results.append(StreamResult(
                name=s.name, frames=s.frames, img_size=s.img_size,
                model=s.predict_model, is_custom=not s.needs_prediction,
                batch=chosen_batch.get(s.name), per_frame_latency_s=None,
                fps_per_gpu=None, required_fps=s.frames / deadline_s if deadline_s else None,
                gpu_seconds=None, feasible=False,
            ))
            continue
        gpu_secs = s.frames * lat
        total_raw += gpu_secs
        if gpu_secs > bottleneck_secs:
            bottleneck_secs, bottleneck = gpu_secs, s.name
        stream_results.append(StreamResult(
            name=s.name, frames=s.frames, img_size=s.img_size,
            model=s.predict_model, is_custom=not s.needs_prediction,
            batch=chosen_batch.get(s.name), per_frame_latency_s=lat,
            fps_per_gpu=1.0 / lat,
            required_fps=s.frames / deadline_s if deadline_s else None,
            gpu_seconds=gpu_secs, feasible=True,
        ))

    required = total_raw / util if util else float("inf")

    if any_unserved:
        return WorkloadResult(
            deadline_s=deadline_s, util=util, max_gpus=max_gpus,
            required_gpu_seconds=required, gpus_needed=0, feasible=False,
            headroom_frac=0.0, bottleneck_stream=bottleneck,
            reason="one or more streams could not be served (no batch fit VRAM, "
                   "or unknown model without a fixed latency)",
            streams=stream_results,
        )

    gpus_needed = max(1, math.ceil(required / deadline_s)) if deadline_s > 0 else 0
    feasible = gpus_needed <= max_gpus
    capacity = gpus_needed * deadline_s
    headroom = (capacity - required) / capacity if capacity > 0 else 0.0
    reason = "ok" if feasible else (
        f"needs {gpus_needed} GPUs but max_gpus={max_gpus}; use a faster GPU, "
        f"raise --max-gpus, or relax the deadline"
    )
    return WorkloadResult(
        deadline_s=deadline_s, util=util, max_gpus=max_gpus,
        required_gpu_seconds=required, gpus_needed=gpus_needed, feasible=feasible,
        headroom_frac=headroom, bottleneck_stream=bottleneck, reason=reason,
        streams=stream_results,
    )
