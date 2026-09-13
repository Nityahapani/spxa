---
name: Bug report
about: Incorrect result, wrong exactness level, or crash
labels: bug
---

## Description

<!-- What went wrong? -->

## Minimal reproducer

```python
import spxa
# minimal code that demonstrates the bug
```

## Expected behaviour

<!-- What should happen? If this is a mathematical correctness issue, cite the theorem or formula. -->

## Actual behaviour

<!-- What actually happens? Include the full traceback if there is one. -->

## Is this an exactness violation?

If spxa returned a wrong closed-form result (e.g. wrong cumulant, wrong characteristic function value), this is an **exactness violation** — the highest priority bug. Please add the label `exactness-violation` and cite the published formula that contradicts the result.

## Environment

- spxa version:
- Python version:
- OS:
