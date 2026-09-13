"""
Compares the most recent benchmark result against the previous one.
Fails if any benchmark has degraded by more than --threshold (default 20%).
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {b["name"]: b["stats"]["mean"] for b in data["benchmarks"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()

    results_dir = Path("benchmarks/results")
    result_files = sorted(results_dir.glob("*.json"))

    if len(result_files) < 2:
        print("Fewer than 2 result files — nothing to compare")
        sys.exit(0)

    previous = load_results(result_files[-2])
    current = load_results(result_files[-1])

    regressions: list[str] = []
    for name, current_mean in current.items():
        if name not in previous:
            continue
        prev_mean = previous[name]
        if prev_mean == 0:
            continue
        change = (current_mean - prev_mean) / prev_mean
        if change > args.threshold:
            regressions.append(
                f"{name}: {prev_mean*1e3:.3f}ms -> {current_mean*1e3:.3f}ms "
                f"(+{change*100:.1f}%, threshold {args.threshold*100:.0f}%)"
            )

    if regressions:
        print(f"Performance regressions detected ({len(regressions)}):")
        for r in regressions:
            print(f"  {r}")
        sys.exit(1)

    print(f"No regressions above {args.threshold*100:.0f}% threshold.")


if __name__ == "__main__":
    main()
