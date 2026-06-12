# Changelog — Battery Health Estimation Pipeline

All revisions to the SOH / RUL pipeline are recorded here.
Each entry maps to a git tag so you can `git checkout <tag>` to get that exact code.

---

## v5.0 — Sensor Cal Factor Cap 1.15 → 1.30 (2026-06-12)
**File:** `soh_rul_12062026.py`
**Tag:** `v5.0-sensor-cal-cap`

### What changed
- **`sensor_cal_factor` cap raised from 1.15 to 1.30** (~line 2801).
  The correction factor `q_base / q_ref_for_soh` was hard-capped at 1.15 (15% max correction).
  Vehicles 415931, 383543, 468807 have sensors that under-read by ~18–19%, so their true
  correction factor is ~1.18–1.22 — hitting the old cap every time. Every `soh_label` for
  these vehicles was systematically 3–7pp too low; XGBoost learned that floor.

### Why 1.30 and not uncapped
The cap protects against noisy early sessions (tiny SOC swings) producing a falsely low
`q_ref_for_soh` (e.g., 200 Ah instead of 515 Ah), which would give a factor of 3.0 and
clip every `soh_label` to 100%. The IQR filter in `_estimate_initial_capacity_ah` handles
most cases; 1.30 is the final backstop. Maximum observed real-world underread in this fleet
is ~19%, so 1.30 gives 11pp of safety margin.

### Expected impact
- 415931, 383543, 468807: `sensor_cal_factor` now ~1.18–1.22 instead of 1.15 → `soh_label`
  increases by 3–7pp → XGBoost trains on corrected labels → `soh_display` moves toward BMS.
- All other vehicles: unaffected (`if q_base_for_soh > q_ref_for_soh` guard; vehicles that
  over-read or read accurately stay at `sensor_cal_factor = 1.0`).

---

## v4.0 — Weighted Training + Chassis ID + Device Tracking (2026-06-12)
**File:** `soh_rul_12062026.py`
**Tag:** `v4.0-weighted-chassis`

### What changed
- **delta_soc weighted XGBoost**: Sessions with large SOC swing (30%+) get full weight;
  small-swing sessions (5–10%) get weight 0.1–0.3. Noisy implied_Q from small-swing
  sessions no longer anchors XGBoost at a false low. Combined with recency weighting
  (recent sessions get up to 1.5× weight vs oldest).
- **Chassis number as vehicle_id**: `vehicle_id` is now the CSV filename stem
  (e.g. `MC2V7SRT0TF131176`) instead of the IMEI. IMEI stored in `imei_from_file` column.
- **Upward confirmation window raised 5 → 20 sessions**: Reduces overcorrection upward
  for healthy vehicles (was causing 99%+ readings on 94% BMS vehicles).
- **`device_tracking.json`**: Audit trail mapping chassis → vehicle name + IMEI history.
  Update `imei_history` whenever a telematics unit is replaced.
- **4 new data files added**: MC2V2HRT0PH228159/160/163/171 (vehicle names TBD).

### Why
After v3.0, three vehicles still showed 78–82% vs 91–94% BMS (training anchored by
noisy short-session labels). Three others showed 96–99% vs 92–94% BMS (free XGBoost
overcorrecting upward, 5-session window too short). delta_soc weighting addresses the
first; raising window_up to 20 addresses the second.

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
