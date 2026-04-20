"""Control script that orchestrates the YOLO CPU/GPU benchmark containers.

Instead of relying on docker-compose, this script uses the Docker SDK for
Python to: validate the daemon is reachable, build the three benchmark
images, and run them sequentially against a shared data volume. Results from
each container land in the host data directory as CSV files.

Usage:
    python src/run_benchmark.py \
        --data-dir ./tmp/data \
        --images-dir ./src/check_yolo_predict/images \
        --weights-dir ./tmp/weights \
        [--only cpu,gpu,yolo] [--skip-build]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import APIError, DockerException, ImageNotFound
from docker.types import DeviceRequest

LOG = logging.getLogger("run_benchmark")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent


@dataclass(frozen=True)
class Benchmark:
    name: str
    context: Path
    image: str
    needs_gpu: bool
    needs_images: bool = False
    needs_weights: bool = False


BENCHMARKS: dict[str, Benchmark] = {
    "cpu": Benchmark(
        name="check_cpu_config",
        context=SRC_DIR / "check_cpu_config",
        image="yolo-benchmark/check_cpu_config:latest",
        needs_gpu=False,
    ),
    "gpu": Benchmark(
        name="check_gpu_config",
        context=SRC_DIR / "check_gpu_config",
        image="yolo-benchmark/check_gpu_config:latest",
        needs_gpu=True,
    ),
    "yolo": Benchmark(
        name="check_yolo_predict",
        context=SRC_DIR / "check_yolo_predict",
        image="yolo-benchmark/check_yolo_predict:latest",
        needs_gpu=True,
        needs_images=True,
        needs_weights=True,
    ),
}


def _client() -> docker.DockerClient:
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:
        LOG.error("cannot reach docker daemon: %s", exc)
        raise SystemExit(2) from exc
    return client


def _build(client: docker.DockerClient, bench: Benchmark, pull: bool = False) -> None:
    LOG.info("building %s from %s", bench.image, bench.context)
    error: str | None = None
    for chunk in client.api.build(
        path=str(bench.context),
        tag=bench.image,
        rm=True,
        pull=pull,
        decode=True,
    ):
        if "stream" in chunk:
            sys.stdout.write(chunk["stream"])
            sys.stdout.flush()
        elif "error" in chunk:
            sys.stderr.write(chunk["error"])
            sys.stderr.flush()
            error = chunk["error"]
    if error:
        raise SystemExit(f"build failed for {bench.name}: {error}")


def _run(
    client: docker.DockerClient,
    bench: Benchmark,
    data_dir: Path,
    images_dir: Path,
    weights_dir: Path,
) -> None:
    volumes = {str(data_dir): {"bind": "/app/data", "mode": "rw"}}
    if bench.needs_images:
        volumes[str(images_dir)] = {"bind": "/app/data/images", "mode": "ro"}
    if bench.needs_weights:
        volumes[str(weights_dir)] = {"bind": "/app/weights", "mode": "rw"}

    device_requests = None
    if bench.needs_gpu:
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

    LOG.info("running %s", bench.name)
    try:
        container = client.containers.run(
            image=bench.image,
            name=bench.name,
            volumes=volumes,
            device_requests=device_requests,
            detach=True,
            remove=False,
        )
    except ImageNotFound as exc:
        raise SystemExit(f"image missing for {bench.name}; run without --skip-build") from exc
    except APIError as exc:
        raise SystemExit(f"docker refused to start {bench.name}: {exc}") from exc

    interrupted = False
    try:
        try:
            for chunk in container.logs(stream=True, follow=True):
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
            result = container.wait()
        except KeyboardInterrupt:
            interrupted = True
            LOG.warning(
                "interrupted; forwarding SIGINT to %s so Python's `with` blocks"
                " can flush CSV buffers before exit",
                bench.name,
            )
            try:
                container.kill(signal="SIGINT")
                result = container.wait(timeout=30)
            except APIError:
                LOG.warning("container did not stop on SIGINT; sending SIGTERM")
                container.stop(timeout=10)
                result = container.wait(timeout=10)
    finally:
        container.remove(force=True)

    if interrupted:
        raise SystemExit(130)

    status = result.get("StatusCode", 1)
    if status != 0:
        raise SystemExit(f"{bench.name} exited with status {status}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "data",
        help="host directory for collected CSV results (mounted at /app/data).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=SRC_DIR / "check_yolo_predict" / "images",
        help="host directory with images for YOLO prediction.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "weights",
        help="host directory used as a persistent cache for YOLO weights (.pt files).",
    )
    parser.add_argument(
        "--only",
        default="cpu,gpu,yolo",
        help="comma-separated subset of benchmarks to run.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="skip image build step and use existing images.",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="force pulling the base image from the registry on build"
        " (default: reuse local base image).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = [name for name in selected if name not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {', '.join(unknown)}")

    data_dir = args.data_dir.resolve()
    images_dir = args.images_dir.resolve()
    weights_dir = args.weights_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if "yolo" in selected and not images_dir.is_dir():
        raise SystemExit(f"images dir not found: {images_dir}")

    client = _client()

    for name in selected:
        bench = BENCHMARKS[name]
        if not args.skip_build:
            _build(client, bench, pull=args.pull)
        _run(client, bench, data_dir, images_dir, weights_dir)

    LOG.info("all benchmarks finished; results in %s", data_dir)


if __name__ == "__main__":
    main()
