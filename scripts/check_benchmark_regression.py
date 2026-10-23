"""Compare explicit benchmark artifacts; never sort commit hashes as chronology."""

import argparse
import json
import math
from pathlib import Path


def load_results(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    values = {b.get("fullname", b["name"]): float(b["stats"]["mean"]) for b in data["benchmarks"]}
    if not values or any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("Benchmark means must be finite, positive and nonempty")
    return values


def regressions(
    previous: dict[str, float], current: dict[str, float], threshold: float
) -> list[str]:
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative")
    shared = previous.keys() & current.keys()
    if not shared:
        raise ValueError("No common benchmarks to compare")
    return [
        f"{name}: {previous[name] * 1e3:.3f}ms -> {current[name] * 1e3:.3f}ms"
        for name in sorted(shared)
        if current[name] > previous[name] * (1 + threshold)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    result = regressions(load_results(args.previous), load_results(args.current), args.threshold)
    for item in result:
        print(item)
    if result:
        raise SystemExit(1)
    print("No regressions above the configured threshold")


if __name__ == "__main__":
    main()
