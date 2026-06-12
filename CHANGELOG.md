# Changelog — Battery Health Estimation Pipeline

All revisions to the SOH / RUL pipeline are recorded here.
Each entry maps to a git tag so you can `git checkout <tag>` to get that exact code.

---

## v3.0 — Confirmation Gate + Free XGBoost (2026-06-12)
**File:** `soh_rul_12062026.py`
**Tag:** `v3.0-confirmation-gate`

### What changed
- **Removed `monotone_constraints=-1`** from XGBoost. The constraint permanently locked
  predictions at the historical SOH minimum whenever the training data had a sustained
  low-reading period. XGBoost is now free to learn the true shape.
- **Added `_apply_confirmation_gate`** (replaces `_apply_1pct_hold_gate`).
  Customer-facing SOH (`soh_display`) uses an asymmetric confirmation window:
  - Downward change (lower SOH): requires **30 consecutive sessions** before showing
    the customer a lower value. Protects against sensor noise / bad charging events.
  - Upward change (higher SOH): requires **5 consecutive sessions**.
  - Within 3 pp of current confirmed value: accepted immediately.
- **Two SOH columns**: `soh_xgb` = raw model output (for debugging);
  `soh_display` = customer-facing stable value.
- `compute_all_rul` now reads `soh_display` for `soh_now` (fleet summary SOH%).

### Why
Three vehicles (383543, 415931, 468807) were reading 14–17 pp below BMS SOH.
The XGBoost monotone constraint locked them at the minimum ever seen in training data.
The confirmation gate keeps the display stable without needing artificial constraints on the model.

---

## v2.0 — 1% Hold Gate + Clean Training Labels (2026-06-12)
**File:** `soh_rul_12062026.py` (earlier commit) / `win_test.py`
**Tag:** `v2.0-hold-gate`

### What changed
- **`_apply_1pct_hold_gate`**: asymmetric gate on `soh_xgb` — downward changes pass
  through freely; upward jumps > 1 pp are held. Replaced strict global-minimum monotone.
- **Training label filter**: sessions where `|soh_label(n) - soh_label(n-1)| > 1 pp`
  are NaN-interpolated before becoming `soh_smooth`. XGBoost never trains on sudden spikes.
- **Removed `_soft_monotone_curve` from `soh_smooth`**: it was locking the training
  target at the historical minimum, anchoring XGBoost permanently.

### Why
Triple monotone application (soh_smooth, XGBoost internal, post-processing) meant a single
noise-induced dip locked the reported SOH forever. 1% gate allows gradual real degradation
through while suppressing one-off sensor glitches.

---

## v1.0 — Original Pipeline (pre 2026-06-12)
**File:** `win_test.py`
**Tag:** `v1.0-original`

### State
Baseline pipeline as received. Strict `_soft_monotone_curve` applied at 3 levels.
Known issue: vehicles with any historical low-reading period are permanently locked at that SOH.
