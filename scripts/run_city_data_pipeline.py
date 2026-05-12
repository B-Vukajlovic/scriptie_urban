"""Run the city data-preparation pipeline across one or more cities.

The runner intentionally shells out to the single-city data-preparation scripts.
Model evaluation is handled by the separate multi-city evaluation scripts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.cities import CITY_CONFIGS, DEFAULT_CITIES, validate_cities  # noqa: E402

StageName = Literal[
    "backbone",
    "target",
    "global_targets",
    "features",
    "model_dataset",
    "diagnostics",
]
DistanceEngine = Literal["euclidean", "block_graph", "osm_network"]
StageStatus = Literal["passed", "failed", "timeout", "skipped", "dry_run"]

DEFAULT_STAGES: tuple[StageName, ...] = (
    "backbone",
    "target",
    "global_targets",
    "features",
    "model_dataset",
    "diagnostics",
)
AVAILABLE_STAGES: tuple[StageName, ...] = DEFAULT_STAGES


@dataclass(frozen=True)
class PipelineArgs:
    cities: list[str]
    stages: list[StageName]
    interim_root: str
    raw_root: str
    outputs_root: str
    distance_engine: DistanceEngine
    graph_batch_size: int
    skip_existing: bool
    dry_run: bool
    fail_fast: bool
    stage_timeout_seconds: int
    summary_path: str | None


@dataclass
class StageResult:
    city: str
    stage: StageName
    status: StageStatus
    seconds: float
    command: list[str]
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    skip_reason: str = ""


def parse_args() -> PipelineArgs:
    parser = argparse.ArgumentParser(
        description="Run city data-preparation and diagnostics stages."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=DEFAULT_CITIES,
        help="City slugs to run. Defaults to all supported cities.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=AVAILABLE_STAGES,
        default=list(DEFAULT_STAGES),
        help="Pipeline stages to run in order.",
    )
    parser.add_argument("--interim-root", default="data/interim")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--distance-engine",
        choices=["euclidean", "block_graph", "osm_network"],
        default="osm_network",
        help="Target reachability engine.",
    )
    parser.add_argument(
        "--graph-batch-size",
        type=int,
        default=128,
        help="Batch size passed to graph-based target construction.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages whose primary output artifact already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and record commands without executing them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed stage.",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=0,
        help="Per-stage timeout. 0 means no timeout.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for the JSON run summary.",
    )
    namespace = parser.parse_args()
    return PipelineArgs(
        cities=[str(city) for city in namespace.cities],
        stages=[cast(StageName, stage) for stage in namespace.stages],
        interim_root=str(namespace.interim_root),
        raw_root=str(namespace.raw_root),
        outputs_root=str(namespace.outputs_root),
        distance_engine=cast(DistanceEngine, namespace.distance_engine),
        graph_batch_size=int(namespace.graph_batch_size),
        skip_existing=bool(namespace.skip_existing),
        dry_run=bool(namespace.dry_run),
        fail_fast=bool(namespace.fail_fast),
        stage_timeout_seconds=int(namespace.stage_timeout_seconds),
        summary_path=(
            str(namespace.summary_path)
            if namespace.summary_path is not None
            else None
        ),
    )


def _script(name: str) -> str:
    return str(Path("scripts") / name)


def stage_command(city: str, stage: StageName, args: PipelineArgs) -> list[str]:
    meta = CITY_CONFIGS[city]
    raw_root = Path(args.raw_root)
    interim_root = Path(args.interim_root)
    common = [sys.executable]

    if stage == "backbone":
        return common + [
            _script("build_city_backbone.py"),
            "--city",
            city,
            "--state",
            meta.state,
            "--place-geoid",
            meta.place_geoid,
            "--tiger-root",
            str(raw_root / "tiger"),
            "--output-root",
            str(interim_root),
        ]
    if stage == "target":
        return common + [
            _script("build_city_target.py"),
            "--city",
            city,
            "--state",
            meta.state,
            "--backbone-root",
            str(interim_root),
            "--lehd-root",
            str(raw_root / "lehd"),
            "--osm-root",
            str(raw_root / "osm"),
            "--tiger-root",
            str(raw_root / "tiger"),
            "--output-root",
            str(interim_root),
            "--distance-engine",
            args.distance_engine,
            "--graph-batch-size",
            str(args.graph_batch_size),
        ]
    if stage == "global_targets":
        return common + [
            _script("build_global_targets.py"),
            "--cities",
            *args.cities,
            "--interim-root",
            str(interim_root),
        ]
    if stage == "features":
        return common + [
            _script("build_city_features.py"),
            "--city",
            city,
            "--state",
            meta.state,
            "--backbone-root",
            str(interim_root),
            "--gtfs-root",
            str(raw_root / "gtfs"),
            "--osm-root",
            str(raw_root / "osm"),
            "--acs-root",
            str(raw_root / "acs"),
            "--output-root",
            str(interim_root),
        ]
    if stage == "model_dataset":
        return common + [
            _script("build_city_model_dataset.py"),
            "--city",
            city,
            "--interim-root",
            str(interim_root),
            "--output-root",
            str(interim_root),
        ]
    if stage == "diagnostics":
        return common + [
            _script("diagnose_city_model_dataset.py"),
            "--city",
            city,
            "--interim-root",
            str(interim_root),
            "--outputs-root",
            str(args.outputs_root),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def primary_artifact(city: str, stage: StageName, args: PipelineArgs) -> Path:
    interim = Path(args.interim_root) / city
    outputs = Path(args.outputs_root)
    artifacts = {
        "backbone": interim / "backbone" / "qc_summary.json",
        "target": interim / "target" / "target_table.parquet",
        "global_targets": Path(args.interim_root) / "global_targets_metadata.json",
        "features": interim / "features" / "feature_table.parquet",
        "model_dataset": interim / "modeling" / "model_dataset.parquet",
        "diagnostics": outputs / "diagnostics" / f"{city}_split_diagnostics.json",
    }
    return artifacts[stage]


def tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def timeout_output_to_str(
    output: str | bytes | bytearray | memoryview | None,
) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, memoryview):
        output = output.tobytes()
    if isinstance(output, bytearray):
        output = bytes(output)
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def run_stage(city: str, stage: StageName, args: PipelineArgs) -> StageResult:
    command = stage_command(city, stage, args)
    artifact = primary_artifact(city, stage, args)
    if args.skip_existing and artifact.exists():
        return StageResult(
            city=city,
            stage=stage,
            status="skipped",
            seconds=0.0,
            command=command,
            skip_reason=f"Primary artifact exists: {artifact}",
        )

    if args.dry_run:
        return StageResult(
            city=city,
            stage=stage,
            status="dry_run",
            seconds=0.0,
            command=command,
        )

    start = time.perf_counter()
    timeout = args.stage_timeout_seconds if args.stage_timeout_seconds > 0 else None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        seconds = time.perf_counter() - start
        return StageResult(
            city=city,
            stage=stage,
            status="timeout",
            seconds=round(seconds, 3),
            command=command,
            stdout_tail=tail(timeout_output_to_str(exc.stdout)),
            stderr_tail=tail(timeout_output_to_str(exc.stderr)),
            skip_reason=f"Timed out after {timeout} seconds",
        )
    seconds = time.perf_counter() - start
    status = "passed" if completed.returncode == 0 else "failed"
    return StageResult(
        city=city,
        stage=stage,
        status=status,
        seconds=round(seconds, 3),
        command=command,
        returncode=completed.returncode,
        stdout_tail=tail(completed.stdout),
        stderr_tail=tail(completed.stderr),
    )


def write_summary(results: list[StageResult], args: PipelineArgs) -> Path:
    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = Path(args.outputs_root) / "pipeline_runs" / f"run_{stamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cities": validate_cities(args.cities),
        "stages": args.stages,
        "distance_engine": args.distance_engine,
        "skip_existing": bool(args.skip_existing),
        "dry_run": bool(args.dry_run),
        "stage_timeout_seconds": int(args.stage_timeout_seconds),
        "results": [asdict(result) for result in results],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    cities = validate_cities(args.cities)
    results: list[StageResult] = []
    blocked: dict[str, bool] = {city: False for city in cities}

    for stage in args.stages:
        if stage == "global_targets":
            if all(blocked.values()):
                result = StageResult(
                    city="__all__",
                    stage=stage,
                    status="skipped",
                    seconds=0.0,
                    command=stage_command(cities[0], stage, args),
                    skip_reason="All cities are blocked by previous failures.",
                )
            else:
                result = run_stage(cities[0], stage, args)
                result.city = "__all__"
            results.append(result)
            print(
                json.dumps(
                    {
                        "city": result.city,
                        "stage": result.stage,
                        "status": result.status,
                        "seconds": result.seconds,
                        "returncode": result.returncode,
                    }
                ),
                flush=True,
            )
            if result.status in {"failed", "timeout"} and args.fail_fast:
                summary_path = write_summary(results, args)
                raise SystemExit(f"Failed at {stage}; summary: {summary_path}")
            if result.status in {"failed", "timeout"}:
                for city in cities:
                    blocked[city] = True
            continue

        for city in cities:
            if blocked[city]:
                result = StageResult(
                    city=city,
                    stage=stage,
                    status="skipped",
                    seconds=0.0,
                    command=stage_command(city, stage, args),
                    skip_reason="Previous stage failed or timed out.",
                )
            else:
                result = run_stage(city, stage, args)
            results.append(result)
            print(
                json.dumps(
                    {
                        "city": result.city,
                        "stage": result.stage,
                        "status": result.status,
                        "seconds": result.seconds,
                        "returncode": result.returncode,
                    }
                ),
                flush=True,
            )
            if result.status in {"failed", "timeout"} and args.fail_fast:
                summary_path = write_summary(results, args)
                raise SystemExit(f"Failed at {city}/{stage}; summary: {summary_path}")
            if result.status in {"failed", "timeout"}:
                blocked[city] = True

    summary_path = write_summary(results, args)
    print(json.dumps({"summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
