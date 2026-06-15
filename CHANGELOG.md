# Changelog — Battery Health Estimation Pipeline

All revisions to the SOH / RUL pipeline are recorded here.
Each entry maps to a git tag so you can `git checkout <tag>` to get that exact code.

---

## v9.0 — RUL: Minimum Data Span Gate + Max Slope Cap (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v9.0-rul-data-span-gate`

### What changed
- **`RUL_MIN_DATA_SPAN_DAYS = 120.0`** (new constant): Vehicles with fewer than 4 months of
  session history are no longer trusted for WLS slope estimation. The slope is zeroed and they
  fall through to the floor rate (0.3%/yr). Basis shows `floor_rate_insufficient_data`.
- **`RUL_MAX_SLOPE_PCT_PER_YEAR = 20.0`** (new constant): After computing the WLS slope, if
  the implied annual degradation rate exceeds 20%/yr, the slope is clamped to -20%/yr. Real-world
  commercial EV fleet max is ~8-10%/yr; 20% gives headroom without allowing noise artifacts.
- **`floor_basis` distinction**: `slope_basis` now reports `floor_rate_insufficient_data` vs
  `floor_rate_no_degradation` so you can tell which vehicles lack data vs which are genuinely flat.

### Why
v8.0 switched slope input to `soh_xgb`, which is correct. But MC2R9SRT0TG132697 had only 71 days
of session data — WLS over ~35 days of noisy soh_xgb produced a -0.06/10k slope that implied
12%/yr degradation, giving a fake EOL of Oct 2027 for a 96% SOH vehicle. Guard 1 (120-day gate)
fixes this directly. Guard 2 (20%/yr cap) catches any remaining cases where a short noisy tail
slips through.

---

## v8.0 — RUL: Use soh_xgb for Slope + Fix Floor Rate Dates (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v8.0-rul-soh-xgb-slope`

### What changed
- **RUL slope now uses `soh_xgb`** instead of `soh_display` (~line 3325).
  `soh_display` has the confirmation gate applied — it makes artificial step changes
  (gate accepts a new level after 30 sessions) and flat periods (while waiting to confirm).
  The WLS slope on a step-change reads as extremely steep → EOL date too near (e.g. 440 days
  for a 96% SOH vehicle). The flat periods give slope≈0 → NA for most vehicles.
  `soh_xgb` is the raw model output without any gate — it shows the genuine gradual trend.
- **Floor rate branch now returns p50 date** instead of NaN.
  When no confirmed degradation is detected, `p50 = p90 = floor_rate estimate (0.3%/yr)`.
  `p10` stays NaN (degradation not confirmed). Every vehicle now shows an EOL date in p50.

### Why
v6.0 introduced the WLS tail slope. v7.0 summary showed most vehicles as NA (flat soh_display
→ zero slope) and two vehicles with unrealistically near EOL dates (confirmation gate step
changes amplified by WLS). Both issues traced to the same root: wrong input column for slope.

---

## v7.0 — SOH Label Quality Gate: Minimum delta_soc = 10% (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v7.0-soh-label-dsoc-gate`

### What changed
- **`SOH_LABEL_MIN_DELTA_SOC = 10.0`** (new constant, ~line 120).
  Sessions with `delta_soc_pct < 10%` no longer contribute their own `soh_label` to training.
  They are NaN'd and replaced by linear interpolation from neighbouring high-quality sessions.
  The existing 1pp session-to-session jump filter runs after, as before.

### Why
`implied_Q_Ah = ah_total / (delta_soc / 100)`. With BMS SOC at 1% integer resolution:
  - delta_soc = 4%: ±1% rounding = ±25% error in implied_Q → ±25pp noise in soh_label
  - delta_soc = 10%: ±1% rounding = ±10% error → ±10pp noise

For a 608 Ah pack these small-swing sessions can give soh_label anywhere from 75% to 120%
(clipped to 100%). The p10 of implied_Q for vehicle 383543 was 445 Ah vs median 567 Ah —
entirely explained by small delta_soc amplification. These bad labels dragged XGBoost
predictions down to 78% even though the vehicle's true SOH is ~91%.

### pkl must be deleted before re-running
soh_label changes → soh_smooth changes → XGBoost retrains.

---

## v6.0 — Improved RUL: Recency-Weighted Slope + Slope-Sampling Uncertainty (2026-06-12)
**File:** `soh_rul_12062026.py`
**Tag:** `v6.0-rul-weighted-slope`

### What changed
- **`extrapolate_rul()` fully rewritten**: replaced knee detection + multiple-window selection
  with a single recency-weighted Weighted Least Squares (WLS) fit on the most recent 50% of
  sessions (min 8). Exponential weights give the newest session weight=1.0, oldest ~0.14.
  One stable slope, no window flip-flopping.
- **Uncertainty via slope sampling** (replaces Gaussian noise on SOH values): WLS gives a
  standard error for the slope. Sample 500 slopes from `Normal(slope, slope_se)` → P10/P50/P90
  of RUL. Tighter P10/P90 for vehicles with a clear trend; wider for noisy/short data.
- **Floor degradation rate** (0.3%/year): when no negative slope is detected, report a
  conservative upper-bound RUL as `rul_days_p90` instead of ∞.
- **Recent km/day** (last 60 days): `km_to_eol` now uses recent usage rate instead of all-time
  average. Falls back to historical if <5 data points in the window.
- **`km_per_day_recent`** added to fleet_summary.csv for transparency.

### Why
Old MC approach added noise to SOH readings (sensor noise model). WLS slope-sampling models
uncertainty in the degradation *trend* — physically correct. Old knee detection on noisy SOH
was unreliable; recency-weighted tail WLS is stable and reflects current behavior.

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
