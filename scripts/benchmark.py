#!/usr/bin/env python3
"""Reproducible local benchmark. Run with uv; standard library only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA_VERSION = 2
ENVIRONMENT_KEYS = {
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_TARGET",
    "CARGO_TARGET_DIR",
    "CARGO_INCREMENTAL",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "ORT_DYLIB_PATH",
    "ORT_LOG_LEVEL",
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "RUST_LOG",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
}
ENVIRONMENT_PREFIXES = (
    "CARGO_PROFILE_RELEASE_",
    "ORT_",
    "OMP_",
    "RAYON_",
    "TOKENIZERS_",
    "COREML_",
    "OPENBLAS_",
    "MKL_",
)
COMPARABILITY_KEYS = (
    "cargo_version",
    "cargo_lock_sha256",
    "profile_sha256",
    "model_hashes",
    "fixtures",
    "rustc",
    "rustc_target",
    "os",
    "hardware",
    "build_settings",
    "environment",
    "registered_provider",
    "thread_settings",
    "samples",
    "warmup",
    "stage_samples",
    "repeats",
)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def json_sha(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_hashes():
    candidates = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for pattern in (
        "src/**/*.rs",
        "examples/**/*.rs",
        "scripts/**/*.py",
        "benches/**/*.rs",
        ".cargo/config",
        ".cargo/config.toml",
    ):
        candidates.extend(ROOT.glob(pattern))
    for path in (ROOT / "build.rs", ROOT / "rust-toolchain", ROOT / "rust-toolchain.toml"):
        if path.is_file():
            candidates.append(path)
    paths = sorted(set(path for path in candidates if path.is_file()))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def relevant_environment():
    keys = set(ENVIRONMENT_KEYS)
    keys.update(
        key
        for key in os.environ
        if key.startswith(ENVIRONMENT_PREFIXES)
    )
    return {key: os.environ.get(key) for key in sorted(keys)}


def trace_summary(path):
    events = json.loads(Path(path).read_text())
    groups = {}
    for event in events:
        args = event.get("args", {})
        event_name = event.get("name")
        if event.get("cat") != "Node" or not isinstance(event_name, str) or not event_name.endswith("_kernel_time"):
            continue
        provider = args.get("provider", "unknown") or "unknown"
        operator = args.get("op_name", "unknown") or "unknown"
        key = (provider, operator)
        row = groups.setdefault(
            key,
            {"provider": provider, "operator": operator, "calls": 0, "duration_us": 0},
        )
        duration = event.get("dur", 0)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration):
            raise ValueError(f"invalid duration in ORT trace {path}")
        row["calls"] += 1
        row["duration_us"] += duration
    return sorted(groups.values(), key=lambda row: row["duration_us"], reverse=True)


def percentile95(samples):
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def session_summary(run):
    samples = run["warm_ms"]
    return {
        "n": len(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile95(samples),
        "initialization_ms": run["initialization"]["total_ms"],
        "peak_rss_bytes": run["peak_rss_bytes"],
        "measurement_diagnostic": run["measurement_diagnostic"],
    }


def validate_result(result, path, allow_trace=False):
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported result schema in {path}")
    if result.get("status") != "complete":
        raise ValueError(f"incomplete result: {path}")
    if result.get("native_profiling") and not allow_trace:
        raise ValueError(f"traced result cannot be compared: {path}")
    fixture_names = result.get("fixtures")
    runs = result.get("runs")
    repeats = result.get("repeats")
    if not isinstance(fixture_names, dict) or not fixture_names or not isinstance(runs, list):
        raise ValueError(f"invalid fixture or run list in {path}")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"invalid repeat count in {path}")
    missing_metadata = [
        key
        for key in COMPARABILITY_KEYS
        + ("source_snapshot_sha256", "binary_sha256", "native_profiling")
        if key not in result
    ]
    if missing_metadata:
        raise ValueError(f"missing comparison metadata in {path}: {missing_metadata}")
    if not isinstance(result["native_profiling"], bool):
        raise ValueError(f"invalid native profiling flag in {path}")
    for key in ("samples", "warmup", "stage_samples"):
        if not isinstance(result.get(key), int) or result[key] < 1:
            raise ValueError(f"invalid {key} count in {path}")
    expected = {(name, repeat) for name in fixture_names for repeat in range(repeats)}
    indexed = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"invalid run entry in {path}")
        key = (run.get("fixture"), run.get("repeat"))
        if key not in expected:
            raise ValueError(f"unexpected run {key!r} in {path}")
        if key in indexed:
            raise ValueError(f"duplicate run {key!r} in {path}")
        if len(run.get("warm_ms", [])) != result.get("samples"):
            raise ValueError(f"partial warm samples for {key!r} in {path}")
        if len(run.get("stage_samples", [])) != result.get("stage_samples"):
            raise ValueError(f"partial stage samples for {key!r} in {path}")
        initialization = run.get("initialization")
        if not isinstance(initialization, dict) or any(
            not isinstance(initialization.get(field), (int, float))
            or not math.isfinite(initialization[field])
            or initialization[field] < 0
            for field in ("tokenizer_ms", "session_ms", "total_ms")
        ):
            raise ValueError(f"invalid initialization timings for {key!r} in {path}")
        if initialization["total_ms"] <= 0:
            raise ValueError(f"missing constructor timing for {key!r} in {path}")
        if (
            not isinstance(run.get("first_call_ms"), (int, float))
            or not math.isfinite(run["first_call_ms"])
            or run["first_call_ms"] <= 0
        ):
            raise ValueError(f"invalid first-call timing for {key!r} in {path}")
        if not isinstance(run.get("sample_count"), int) or run["sample_count"] != result["samples"]:
            raise ValueError(f"invalid sample count for {key!r} in {path}")
        if not isinstance(run.get("warmup_count"), int) or run["warmup_count"] != result["warmup"]:
            raise ValueError(f"invalid warmup count for {key!r} in {path}")
        for sample in run["warm_ms"]:
            if not isinstance(sample, (int, float)) or not math.isfinite(sample) or sample <= 0:
                raise ValueError(f"invalid warm sample for {key!r} in {path}")
        for stage in run["stage_samples"]:
            if not isinstance(stage, dict) or not all(
                field in stage
                for field in (
                    "tokenization_ms",
                    "window_preparation_ms",
                    "tensor_preparation_ms",
                    "model_execution_ms",
                    "decoding_ms",
                    "stitching_ms",
                    "recognizers_ms",
                    "masking_ms",
                    "total_ms",
                )
            ) or any(
                not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in stage.values()
            ):
                raise ValueError(f"invalid stage sample for {key!r} in {path}")
        if not isinstance(run.get("peak_rss_bytes"), int) or run["peak_rss_bytes"] <= 0:
            raise ValueError(f"missing peak RSS for {key!r} in {path}")
        if not isinstance(run.get("semantic"), dict) or not isinstance(run.get("confidence"), list):
            raise ValueError(f"missing semantic or confidence output for {key!r} in {path}")
        semantic = run["semantic"]
        if not all(
            field in semantic
            for field in (
                "replacements",
                "placeholder_map_sha256",
                "anonymized_text_sha256",
                "sequence_len",
                "windows",
            )
        ):
            raise ValueError(f"incomplete semantic output for {key!r} in {path}")
        if not isinstance(semantic["replacements"], list) or not isinstance(semantic["placeholder_map_sha256"], dict):
            raise ValueError(f"invalid semantic output for {key!r} in {path}")
        for replacement in semantic["replacements"]:
            if not isinstance(replacement, dict) or not all(
                field in replacement
                for field in ("start", "end", "family", "placeholder", "original_sha256")
            ):
                raise ValueError(f"invalid semantic replacement for {key!r} in {path}")
        for score in run["confidence"]:
            if (
                not isinstance(score, dict)
                or not isinstance(score.get("score"), (int, float))
                or not math.isfinite(score["score"])
                or not all(field in score for field in ("start", "end", "family", "placeholder"))
            ):
                raise ValueError(f"invalid confidence output for {key!r} in {path}")
        if not isinstance(run.get("quality"), dict):
            raise ValueError(f"missing fixture quality report for {key!r} in {path}")
        if not isinstance(run.get("measurement_diagnostic"), dict):
            raise ValueError(f"missing timing-gate diagnostic for {key!r} in {path}")
        if "ort_build_info" not in run or "window_config" not in run:
            raise ValueError(f"missing runtime configuration for {key!r} in {path}")
        throughput = run.get("throughput")
        if not isinstance(throughput, dict) or not all(
            isinstance(throughput.get(field), (int, float))
            and math.isfinite(throughput[field])
            and throughput[field] > 0
            for field in (
                "documents_per_second",
                "input_bytes_per_second",
                "input_tokens_per_second",
            )
        ):
            raise ValueError(f"invalid throughput values for {key!r} in {path}")
        if not isinstance(run.get("input_tokens_including_special"), int) or run["input_tokens_including_special"] < 1:
            raise ValueError(f"invalid input token count for {key!r} in {path}")
        if not isinstance(run.get("windows"), int) or run["windows"] < 1:
            raise ValueError(f"invalid window count for {key!r} in {path}")
        indexed[key] = run
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"missing runs in {path}: {missing}")
    return indexed


def confidence_changes(before, after):
    before_scores = before["confidence"]
    after_scores = after["confidence"]
    if len(before_scores) != len(after_scores):
        return [{"reason": "different score count", "before": len(before_scores), "after": len(after_scores)}]
    changes = []
    keys = ("start", "end", "family", "placeholder")
    for left, right in zip(before_scores, after_scores):
        if any(left.get(key) != right.get(key) for key in keys):
            return [{"reason": "scores cannot be aligned by semantic span"}]
        delta = right["score"] - left["score"]
        if delta != 0:
            changes.append(
                {
                    **{key: left[key] for key in keys},
                    "before": left["score"],
                    "after": right["score"],
                    "delta": delta,
                }
            )
    return changes


def compare(before_path, after_path):
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    before_runs = validate_result(before, before_path)
    after_runs = validate_result(after, after_path)
    for key in COMPARABILITY_KEYS:
        if before.get(key) != after.get(key):
            raise ValueError(f"incomparable {key}")

    for name in before["fixtures"]:
        for repeat in range(before["repeats"]):
            left = before_runs[(name, repeat)]
            right = after_runs[(name, repeat)]
            if left["ort_build_info"] != right["ort_build_info"]:
                raise ValueError(f"incomparable ORT build for {name} session {repeat}")
            if left["window_config"] != right["window_config"]:
                raise ValueError(f"incomparable window config for {name} session {repeat}")

    rows = []
    all_semantics_preserved = True
    all_confidence_changes = []
    all_regressions = []
    for name in before["fixtures"]:
        left_runs = [before_runs[(name, repeat)] for repeat in range(before["repeats"])]
        right_runs = [after_runs[(name, repeat)] for repeat in range(after["repeats"])]
        left_samples = [sample for run in left_runs for sample in run["warm_ms"]]
        right_samples = [sample for run in right_runs for sample in run["warm_ms"]]
        left_sessions = [session_summary(run) for run in left_runs]
        right_sessions = [session_summary(run) for run in right_runs]
        pooled_before = {
            "n": len(left_samples),
            "p50_ms": statistics.median(left_samples),
            "p95_ms": percentile95(left_samples),
            "initialization_p50_ms": statistics.median(
                run["initialization"]["total_ms"] for run in left_runs
            ),
            "peak_rss_max_bytes": max(run["peak_rss_bytes"] for run in left_runs),
        }
        pooled_after = {
            "n": len(right_samples),
            "p50_ms": statistics.median(right_samples),
            "p95_ms": percentile95(right_samples),
            "initialization_p50_ms": statistics.median(
                run["initialization"]["total_ms"] for run in right_runs
            ),
            "peak_rss_max_bytes": max(run["peak_rss_bytes"] for run in right_runs),
        }
        pooled_median_change = 100 * (pooled_after["p50_ms"] / pooled_before["p50_ms"] - 1)
        sessions = []
        for repeat, (left_run, right_run, left, right) in enumerate(
            zip(left_runs, right_runs, left_sessions, right_sessions)
        ):
            median_change = 100 * (right["p50_ms"] / left["p50_ms"] - 1)
            p95_change = 100 * (right["p95_ms"] / left["p95_ms"] - 1)
            init_change = 100 * (right["initialization_ms"] / left["initialization_ms"] - 1)
            rss_change = 100 * (right["peak_rss_bytes"] / left["peak_rss_bytes"] - 1)
            sessions.append(
                {
                    "repeat": repeat,
                    "before": left,
                    "after": right,
                    "median_change_percent": median_change,
                    "p95_change_percent": p95_change,
                    "initialization_change_percent": init_change,
                    "peak_rss_change_percent": rss_change,
                }
            )
            if median_change > 5:
                all_regressions.append(f"{name} session {repeat} median +{median_change:.2f}%")
            if p95_change > 10:
                all_regressions.append(f"{name} session {repeat} p95 +{p95_change:.2f}%")
            if init_change > 10:
                all_regressions.append(f"{name} session {repeat} initialization +{init_change:.2f}%")
            if rss_change > 10:
                all_regressions.append(f"{name} session {repeat} peak RSS +{rss_change:.2f}%")
            confidence_delta = confidence_changes(left_run, right_run)
            if confidence_delta:
                all_confidence_changes.append(
                    {"fixture": name, "repeat": repeat, "changes": confidence_delta}
                )

        behavior_preserved = all(run["semantic"] == other["semantic"] for run, other in zip(left_runs, right_runs))
        all_semantics_preserved = all_semantics_preserved and behavior_preserved
        if pooled_median_change > 5:
            all_regressions.append(f"{name} pooled median +{pooled_median_change:.2f}%")
        p95_change = 100 * (pooled_after["p95_ms"] / pooled_before["p95_ms"] - 1)
        init_change = 100 * (
            pooled_after["initialization_p50_ms"] / pooled_before["initialization_p50_ms"] - 1
        )
        rss_change = 100 * (
            pooled_after["peak_rss_max_bytes"] / pooled_before["peak_rss_max_bytes"] - 1
        )
        if p95_change > 10:
            all_regressions.append(f"{name} pooled p95 +{p95_change:.2f}%")
        if init_change > 10:
            all_regressions.append(f"{name} initialization median +{init_change:.2f}%")
        if rss_change > 10:
            all_regressions.append(f"{name} max peak RSS +{rss_change:.2f}%")
        rows.append(
            {
                "fixture": name,
                "before": {"pooled": pooled_before, "sessions": left_sessions},
                "after": {"pooled": pooled_after, "sessions": right_sessions},
                "pooled_median_change_percent": pooled_median_change,
                "pooled_p95_change_percent": p95_change,
                "sessions": sessions,
                "behavior_preserved": behavior_preserved,
                "label_quality": {
                    "before": left_runs[0]["quality"],
                    "after": right_runs[0]["quality"],
                },
            }
        )
    report = {
        "comparison_schema_version": 1,
        "provenance": {
            "before": {
                "source_snapshot_sha256": before["source_snapshot_sha256"],
                "binary_sha256": before["binary_sha256"],
            },
            "after": {
                "source_snapshot_sha256": after["source_snapshot_sha256"],
                "binary_sha256": after["binary_sha256"],
            },
        },
        "rows": rows,
        "confidence_drift": all_confidence_changes,
        "regressions_over_practical_thresholds": all_regressions,
        "behavior_preserved": all_semantics_preserved,
    }
    print(json.dumps(report, indent=2))
    if not all_semantics_preserved:
        raise SystemExit("semantic behavior regression; see comparison report")


def build_benchmark(stderr_path):
    build_command = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--features",
        "profiling",
        "--example",
        "pipeline_bench",
        "--message-format=json-render-diagnostics",
    ]
    result = subprocess.run(build_command, cwd=ROOT, text=True, capture_output=True)
    stderr_path.write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"benchmark build failed; inspect {stderr_path}")
    executable = None
    for line in result.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        target = message.get("target", {})
        if (
            message.get("reason") == "compiler-artifact"
            and target.get("name") == "pipeline_bench"
            and "example" in target.get("kind", [])
        ):
            executable = message.get("executable")
    if not executable or not Path(executable).is_file():
        raise RuntimeError(f"Cargo did not report the benchmark executable; inspect {stderr_path}")
    return Path(executable).resolve(), build_command


def parse_peak_rss(stderr):
    patterns = {
        "Darwin": r"^\s*(\d+)\s+maximum resident set size\s*$",
        "Linux": r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$",
    }
    pattern = re.compile(patterns[platform.system()], re.MULTILINE)
    match = pattern.search(stderr)
    if not match:
        raise ValueError("/usr/bin/time did not report maximum resident set size")
    value = int(match.group(1))
    return value if platform.system() == "Darwin" else value * 1024


def validate_child_row(row, fixture_name, samples, warmup, stage_samples):
    if row.get("fixture") != fixture_name:
        raise ValueError(f"child returned unexpected fixture {row.get('fixture')!r}")
    if row.get("sample_count") != samples or len(row.get("warm_ms", [])) != samples:
        raise ValueError("child returned partial warm samples")
    if row.get("warmup_count") != warmup:
        raise ValueError("child warmup count does not match request")
    if len(row.get("stage_samples", [])) != stage_samples:
        raise ValueError("child returned partial stage samples")
    if not isinstance(row.get("semantic"), dict) or not isinstance(row.get("confidence"), list):
        raise ValueError("child omitted semantic or confidence output")
    for sample in row["warm_ms"]:
        if not isinstance(sample, (int, float)) or not math.isfinite(sample) or sample <= 0:
            raise ValueError("child returned invalid warm sample")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--stage-samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixture", action="append")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not args.output or min(args.samples, args.warmup, args.stage_samples, args.repeats) < 1:
        parser.error("--output and positive sample, warmup, stage-sample and repeat counts required")
    if platform.system() not in ("Darwin", "Linux"):
        parser.error("peak RSS supported on macOS and Linux only")
    args.output = args.output.resolve()
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    build_stderr = args.output.parent / f"{args.output.stem}.build.stderr.txt"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or partial_path.exists():
        parser.error("output or partial output already exists; use a new path")

    profile_path = ROOT / "models/profiles.toml"
    profile = tomllib.loads(profile_path.read_text())
    model_dir = profile_path.parent / profile["model_dir"]
    fixtures = sorted((ROOT / "performance/fixtures").glob("*.json"))
    fixtures = [fixture for fixture in fixtures if fixture.name != "behavior_spans.json"]
    if args.fixture:
        requested = args.fixture
        if len(requested) != len(set(requested)):
            parser.error("--fixture values must be unique")
        fixture_by_name = {fixture.stem: fixture for fixture in fixtures}
        unknown = sorted(set(requested) - set(fixture_by_name))
        if unknown:
            parser.error(f"unknown fixture: {', '.join(unknown)}")
        fixtures = [fixture_by_name[name] for name in requested]
    if not fixtures:
        parser.error("no fixtures selected")

    sidecars = [build_stderr]
    for repeat in range(args.repeats):
        for fixture in fixtures:
            stem = f"{args.output.stem}-{repeat}-{fixture.stem}"
            sidecars.append(args.output.parent / f"{stem}.stderr.txt")
            if args.trace:
                sidecars.extend(args.output.parent.glob(f"{stem}*.json"))
    if any(path.exists() for path in sidecars):
        parser.error("one or more raw sidecar paths already exist; use a new output stem")
    binary, build_command = build_benchmark(build_stderr)

    rustc = command("rustc", "-Vv")
    rustc_target = next(
        (line.partition(":")[2].strip() for line in rustc.splitlines() if line.startswith("host:")),
        "unknown",
    )
    hardware = (
        command("sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu")
        if sys.platform == "darwin"
        else command("lscpu")
    )
    sources = source_hashes()
    metadata = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "in_progress",
        "label": args.label,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "git_commit": command("git", "rev-parse", "HEAD"),
        "source_hashes": sources,
        "source_snapshot_sha256": json_sha(sources),
        "binary_sha256": sha(binary),
        "binary_path": str(binary),
        "rustc": rustc,
        "rustc_target": rustc_target,
        "cargo_version": command("cargo", "-V"),
        "os": platform.platform(),
        "hardware": hardware,
        "build_settings": {
            "command": build_command[:-1],
            "profile": "release",
            "features": ["profiling"],
            "target_override": os.environ.get("CARGO_BUILD_TARGET"),
            "cargo_target_dir": os.environ.get("CARGO_TARGET_DIR"),
        },
        "build_stderr_file": build_stderr.name,
        "cargo_lock_sha256": sha(ROOT / "Cargo.lock"),
        "profile_sha256": sha(profile_path),
        "model_hashes": {
            relative: sha(model_dir / relative)
            for relative in ("config.json", "tokenizer.json", "onnx/model_quantized.onnx")
        },
        "model_bytes": {
            relative: (model_dir / relative).stat().st_size
            for relative in ("config.json", "tokenizer.json", "onnx/model_quantized.onnx")
        },
        "registered_provider": "CoreML with CPU fallback" if sys.platform == "darwin" else "CPU",
        "thread_settings": {
            "intra_op": "ORT default (0)",
            "inter_op": "ORT default",
            "execution": "sequential",
            "graph_optimization": "Level3",
        },
        "environment": relevant_environment(),
        "rss_method": "/usr/bin/time child high-water RSS; ps snapshots are current RSS; bytes",
        "fixtures": {fixture.stem: sha(fixture) for fixture in fixtures},
        "samples": args.samples,
        "warmup": args.warmup,
        "stage_samples": args.stage_samples,
        "repeats": args.repeats,
        "native_profiling": args.trace,
        "runs": [],
    }
    partial_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # Run one child at a time. Rotate fixture order to reduce fixed-order bias.
    for repeat in range(args.repeats):
        order = fixtures[repeat % len(fixtures) :] + fixtures[: repeat % len(fixtures)]
        for fixture in order:
            if sha(binary) != metadata["binary_sha256"]:
                raise RuntimeError("benchmark executable changed after build")
            stem = f"{args.output.stem}-{repeat}-{fixture.stem}"
            stderr_path = args.output.parent / f"{stem}.stderr.txt"
            call = [
                str(binary),
                "--fixture",
                str(fixture),
                "--profiles",
                str(profile_path),
                "--samples",
                str(args.samples),
                "--warmup",
                str(args.warmup),
                "--stage-samples",
                str(args.stage_samples),
            ]
            trace_prefix = args.output.parent / stem
            if args.trace:
                call += ["--trace", str(trace_prefix)]
            timer = ["/usr/bin/time", "-l"] if sys.platform == "darwin" else ["/usr/bin/time", "-v"]
            print(
                f"{args.label}: repeat {repeat + 1}/{args.repeats} {fixture.stem}",
                file=sys.stderr,
                flush=True,
            )
            result = subprocess.run(timer + call, cwd=ROOT, text=True, capture_output=True)
            stderr_path.write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"benchmark child failed; inspect {stderr_path}")
            try:
                row = json.loads(result.stdout)
                validate_child_row(row, fixture.stem, args.samples, args.warmup, args.stage_samples)
                row["peak_rss_bytes"] = parse_peak_rss(result.stderr)
                ordinary_p50 = statistics.median(row["warm_ms"])
                stage_total_median = statistics.median(
                    stage["total_ms"] for stage in row["stage_samples"]
                )
                row["measurement_diagnostic"] = {
                    "ordinary_warm_p50_ms": ordinary_p50,
                    "measured_stage_total_median_ms": stage_total_median,
                    "delta_ms": stage_total_median - ordinary_p50,
                    "delta_percent": 100 * (stage_total_median / ordinary_p50 - 1),
                    "interpretation": (
                        "diagnostic only; does not isolate disabled-gate overhead"
                    ),
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid benchmark child result; inspect {stderr_path}") from error
            row["repeat"] = repeat
            row["stderr_file"] = stderr_path.name
            if args.trace:
                trace_path = Path(row.get("trace_path", ""))
                if not trace_path.is_file():
                    raise RuntimeError("ORT trace file missing; inspect child stderr")
                row["operator_summary"] = trace_summary(trace_path)
            metadata["runs"].append(row)
            partial_path.write_text(json.dumps(metadata, indent=2) + "\n")

    metadata["status"] = "complete"
    validate_result(metadata, partial_path, allow_trace=args.trace)
    partial_path.write_text(json.dumps(metadata, indent=2) + "\n")
    partial_path.replace(args.output)


if __name__ == "__main__":
    main()
