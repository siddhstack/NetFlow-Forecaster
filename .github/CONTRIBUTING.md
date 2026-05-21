# Contributing

Thanks for improving NetFlow-Forecaster. This project is an applied ML and
networking toolkit, so useful contributions should preserve reproducibility,
chronological time-series evaluation, and clear evidence.

## Development Setup

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Use the cross-platform runner for local smoke checks:

```powershell
python runners\run.py synthetic --samples 300 --epochs 3 --no-auto-benchmark --skip-install
```

## Contribution Guidelines

- Keep telemetry rows in chronological order.
- Do not fit scalers, calibrators, or feature transforms on validation or test
  labels.
- Preserve existing CLI flags when adding new runner or trainer options.
- Prefer small, focused changes over broad refactors.
- Add or update tests when changing shared evaluation, training, or runner
  behavior.
- Keep generated model artifacts out of git unless they are intentionally copied
  into `docs/` as curated evidence.
- Report benchmark results honestly, including failed gates.

## Pull Request Checklist

- [ ] The change is scoped and described clearly.
- [ ] Tests or smoke checks were run.
- [ ] README/docs were updated if user-facing behavior changed.
- [ ] New results are backed by files in `docs/results` or clearly described as
      local-only.
- [ ] No generated caches, virtual environments, or large run folders are staged.

## Useful Commands

```powershell
python -m py_compile ml\*.py scripts\*.py runners\run.py
python -m pytest
python runners\run.py benchmark --target-quality 90 --max-attempts 2 --skip-install
```
