# Benchmarks

Run `pytest benchmarks/ --benchmark-only --benchmark-json=benchmarks/results/current.json`.
The four workloads exercise vector CF evaluation, canonical clock composition, path simulation
and FFT pricing. They are separate from the mathematical correctness suite.

CI uploads artifacts and caches historical benchmark files rather than committing timing noise
back into the repository. A comparison runs only when a previous successful result exists on
the same CI platform. Hosted-runner timing is noisy: the 20% alert threshold is a diagnostic,
not an accuracy test or a hardware-independent performance promise.

For an explicit comparison:

```bash
python scripts/check_benchmark_regression.py --previous old.json --current current.json --threshold .20
```

Only matching workload fullnames are compared. Missing/empty/invalid artifacts and no shared
workloads fail explicitly. Commit hashes are never treated as chronological order.
