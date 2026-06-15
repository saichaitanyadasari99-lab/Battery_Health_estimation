# Changelog — Battery Health Estimation Pipeline

All revisions to the SOH / RUL pipeline are recorded here.
Each entry maps to a git tag so you can `git checkout <tag>` to get that exact code.

---

## v12.3 — Distance: Time-Sort + Rolling-Median Spike Removal (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v12.3-odometer-spike-removal`

### What changed
- Added `_remove_odometer_spikes(dist_series, spike_factor=5.0)`:
  uses a centered rolling median (window=21, `center=True`) to detect transient jumps.
  If a value exceeds 5× its local rolling median it is dropped. The centered window
  naturally captures the "settled afterwards" property — a spike surrounded by normal
  values has a low rolling median and gets flagged; a genuine sustained increase has a
  proportionally rising median and is kept.
  Hard-cap (>= TOTAL_DISTANCE_MAX_KM) is applied first as a backstop.
- `compute_all_rul` now **sorts `raw_v` by `_utc_num` before distance extraction**.
  Without time-sorting, `_cumulative_odometer` saw values in arbitrary order, detected
  hundreds of false "resets" (each adding a large delta), and summed them into 777M km.
- Both the full-history and recent-60-day distance paths sort + spike-clean before
  calling `_cumulative_odometer`.

### Why
v12.2 added the 500k absolute filter but not the time-sort. With unsorted data,
`_cumulative_odometer` misidentified ordering artefacts as resets and accumulated
the deltas, producing wildly incorrect km totals (e.g. 777,706,096 km for 228171).
The spike-removal step also handles any fault-code values that fall below 500k.

---

## v12.2 — Distance: Filter INT32_MAX Fault Codes + Remove Meters Heuristic (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v12.2-distance-fault-filter`

### What changed
- Added constant `TOTAL_DISTANCE_MAX_KM = 500_000.0`.
- `_cumulative_odometer` now drops all values `>= TOTAL_DISTANCE_MAX_KM` before processing.
  This removes telematics sensor fault codes (e.g. `INT32_MAX / 100 ≈ 21,474,836`) which
  appear as isolated spikes in `totalDistance`.
- Removed the `/1000` meters auto-detection heuristic introduced in v11.1. Units throughout
  this fleet are km; the 21M spike was a sensor fault code, not a unit issue.

### Why
`totalDistance` for 228171 shows a spike to exactly `21,474,836` — which equals
`2^31 / 100 = 21,474,836.48`, a standard telematics overflow sentinel emitted when the
sensor malfunctions or the counter overflows a 32-bit integer. The v11.1 "/1000 if >1500
km/day" heuristic happened to produce a plausible-looking number (21,475 km) by accident,
but was conceptually wrong. The correct fix is to discard the fault value entirely and
compute cumulative distance from the valid readings on either side.

---

## v12.1 — Distance: Cumulative Odometer Across Resets (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v12.1-odometer-reset`

### What changed
- Added `_cumulative_odometer(dist_arr)` helper that sums monotonically increasing segments
  of `totalDistance`, detecting resets when the value drops by more than 5% of the previous
  reading or by more than 1000 units. Returns `(total_delta, last_raw_value)`.
- `km_run_till_date` now uses cumulative total distance across all resets instead of
  `max(totalDistance)`. If an odometer reset occurred mid-life, the lifetime km is correctly
  accumulated as segment1_delta + segment2_delta + …
- `km_per_day` (recent 60-day window) also uses `_cumulative_odometer` so a reset inside the
  window doesn't falsely show near-zero or near-max km/day.
- Removed the now-redundant `km_odometer` (max) variable.

### Why
MC2V2HRT0PH228171's `totalDistance` resets from ~21,474,836 back to ~0.69 between Oct 2025
and Mar 2026. The old `max - min` approach happened to give approximately the right delta in
this case (since min ≈ 0), but `km_run_till_date = max` would read the old pre-reset peak
rather than the true cumulative total. Any future vehicle where the data window starts after
a prior partial run (min > 0) would get an undercount. The cumulative approach is correct
regardless of how many resets occur or where in the data window the reset falls.

---

## v12.0 — RUL: Fill LSTM Trailing NaN with XGBoost Values (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v12.0-lstm-nan-tail-fill`

### What changed
- In `compute_all_rul`, when a vehicle has LSTM results, the LSTM `soh_pred` array now has its
  trailing NaN values filled with the corresponding `soh_xgb` values before being passed to
  `extrapolate_rul`. Fix: `soh_seq[:n_fill] = np.where(nan_mask, xgb_vals[:n_fill], soh_seq[:n_fill])`.
- Only trailing NaN slots are affected; finite LSTM predictions are preserved as-is.

### Why
LSTM models have a lookback gap — the last `lb` sessions cannot produce a prediction because
the model needs `lb` preceding sessions. These slots are NaN in `soh_pred`. For vehicle
MC2V7SRT0TG132661 the last 5 LSTM predictions were NaN; the most recent finite LSTM value was
75.14% (from an old session when the vehicle genuinely read low). After `_finite_series` NaN
filtering, `soh_seq[-1]` = 75.14% which is below the EOL threshold of 80%, triggering
`already_at_eol` and reporting 0 days RUL / an EOL date in the past.
XGBoost predictions for the same recent sessions show 93–97% SOH — the vehicle is healthy.
Filling the LSTM tail NaN with XGBoost values gives `soh_now` ≈ 94.5%, correctly clearing
the already_at_eol flag.

---

## v11.0 — RUL: Floor Rate p90 Spread + Min SE 20% + Manager Table Redesign (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v11.0-rul-spread-manager-table`

### What changed
- **Floor rate p90 = 2× p50**: Previously `rul_days_p90 = rul_days_p50` for all floor-rate vehicles,
  making "likely" and "best" dates identical. p90 now represents the best case (half the floor rate),
  so the Best EOL date is ~2× further out than the Likely date.
- **Min slope SE raised 5% → 20%**: WLS slope SE was floored at 5% of slope magnitude. This was too
  tight — P10/P90 bands were almost indistinguishable even for noisy data. 20% gives a more realistic
  spread (model uncertainty + future conditions).
- **Manager table redesign**: `[MANAGER TABLE]` now shows exactly:
  Vehicle | KM_Run | Init_kWh | Curr_kWh | Init_Ah | Curr_Ah | RUL_Likely | RUL_Best | EOL_Likely | EOL_Best

### Why
"Likely" and "Best" dates were near-identical for most vehicles, giving false precision.
Floor rate had no uncertainty band at all. WLS uncertainty was unrealistically tight.
Manager table had too many columns (status, action, distance remaining, daily km) that obscured
the key numbers managers actually need.

---

## v10.0 — RUL: Lifetime Slope Fallback for Floor Rate (2026-06-15)
**File:** `soh_rul_12062026.py`
**Tag:** `v10.0-rul-lifetime-slope-fallback`

### What changed
- When floor rate is used (WLS tail flat or data < 120 days), the code now checks whether
  the vehicle has historically degraded > 2% (first vs last decile of soh_xgb).
  If so, half the lifetime degradation rate is used as the floor instead of the fixed 0.3%/yr.
- `slope_basis` now reports `floor_rate_lifetime_slope` for this path.
- The 0.5× factor is conservative: it assumes future degradation will slow compared to history.

### Why
v9.0 correctly suppressed fake-steep slopes for data-sparse vehicles. But the fixed 0.3%/yr
floor rate is too conservative for vehicles that have genuinely degraded — e.g.:
  - MC2V2HRT0PH228160: soh_xgb dropped 10.2% over 534 days (7%/yr lifetime), recent tail flat
    at ~88%. At 0.3%/yr → 25 years → "Beyond 5y horizon". Misleading for an 85.6% SOH vehicle.
  - MC2V7SRT0TF131176: 6.9% drop over 146 days. Same issue.
Using half the lifetime rate gives a realistic EOL estimate that reflects observed degradation
while remaining conservative about future behavior.

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
