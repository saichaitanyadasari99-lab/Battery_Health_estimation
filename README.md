# Battery Health Estimation Pipeline

SOH (State of Health) + RUL (Remaining Useful Life) estimation for commercial EV fleets.

## Quick start

```bash
python soh_rul_12062026.py
```

## File map

| File | Purpose |
|------|---------|
| `soh_rul_12062026.py` | **Current working version** — use this |
| `win_test.py` | Original baseline (v1.0) — kept for reference |
| `win_test_backup.py` | Backup of original |
| `codex_ev_pipeline_combined.py` | Separate fleet (different dataset) |
| `CHANGELOG.md` | Full revision history with explanations |

## Revision history

See `CHANGELOG.md` or use git tags:

```bash
git tag                        # list all versions
git checkout v3.0-confirmation-gate   # jump to a version
git checkout main              # return to latest
```

## Key design decisions (current v3.0)

- **`soh_xgb`** — raw XGBoost prediction, stored for debugging
- **`soh_display`** — customer-facing SOH with 30-session confirmation window for drops,
  5-session for rises. Customer never sees a one-off sensor noise reading.
- XGBoost runs without `monotone_constraints` — monotone behaviour comes from the
  confirmation gate, not from forcing the model.

## What is NOT in this repo

- `data/` — vehicle CSV files (private fleet data, gitignored)
- `*.pkl` — pipeline state cache (gitignored)
- `soh_rul_results_*/` — generated outputs/images (gitignored)
