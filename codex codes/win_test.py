"""
SOH + RUL Pipeline for Commercial EV Fleet  (Combined)
=======================================================
Creator  : Chaitanya Dasari

Combined pipeline: Data Cleaning notebook + SOH/RUL pipeline in one single run.

Flow:
  Raw CSVs (folder)
    -> [Step 0] load_and_clean()
         |-- Read all CSVs, resolve vehicle_id
         |-- Cast numerics, fix hrlfc counter resets
         |-- State-machine session segmentation
         |     (CHARGING / DRIVING / CRANKING / OTHERS)
         |-- GPS-UTC -> IST DateTime conversion
         |-- Compute dt_sec, charge_calc, chg_power_calc
         |-- Derived signals (volt_spread, temp_spread)
    -> [Step 1] build_session_table()   â€” aggregate per charging session
    -> [Step 2] compute_soh_labels()    â€” energy-based pseudo-labels
    -> [Step 3] train_xgboost_soh()     â€” XGBoost SOH per session
    -> [Step 4] train_lstm_trajectory() â€” LSTM SOH trajectory
    -> [Step 5] compute_all_rul()       â€” RUL (days to 80% EOL)
    -> [Step 6] detect_battery_replacements()
    -> Plots + CSV exports

v2 fixes (original pipeline):
  - Dynamic per-vehicle Q bounds (no hardcoded limits)
  - hrlfc counter reset repair + overflow cap
  - Monotone XGBoost + post-processing
  - IQR label cleaning per vehicle
  - Min sessions lowered to 8
  - CSV search handles subfolders + uppercase extensions

Integration notes:
  - If input CSVs are already pre-cleaned by the notebook (bucket/session_id columns
    present), segmentation is skipped automatically and existing labels are used.
  - The notebook's EDA plots (histograms, scatter) are intentionally excluded here;
    run the notebook separately if EDA visuals are needed.
"""

import re
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from scipy import stats
from itertools import chain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, UnivariateSpline

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# USER INPUTS (edit these for normal operation)
# ------------------------------------------------------------------------------
SOH_EOL       = 80.0                  # End-of-life SOH threshold %
MIN_DELTA_SOC = 2.0                   # Minimum SOC swing % to use a session
MIN_AH        = 1.0                   # Minimum Ah delivered in a session
EXPECTED_CAPACITY_OPTIONS_AH = (104.5, 150, 300, 600)  # Fleet pack capacities
REPORT_RUL_CAP_DAYS = 1825.0          # RUL reporting horizon cap (days)
VEHICLE_ID_ALIASES = {}               # Optional: {"actual_vehicle": ["old_device_id", "new_device_id"]}
VEHICLE_ID_ALIAS_PATH = None          # Optional JSON path; auto-detects vehicle_id_aliases.json if None
VEHICLE_ID_MODE = "latest_device_per_file"  # auto | latest_device_per_file

# ------------------------------------------------------------------------------
# OPTIONAL ADVANCED INPUTS (edit only when tuning behavior)
# ------------------------------------------------------------------------------
SOFT_MIN_DROP_XGB   = 0.6             # Min total drop enforced on XGBoost trend
SOFT_MIN_DROP_LSTM  = 0.6             # Min total drop enforced on LSTM trend
REPL_Q_JUMP_PCT     = 20.0            # Replacement detection: capacity jump %
REPL_Q_JUMP_AH      = 35.0            # Replacement detection: absolute Ah jump
REPL_SOH_JUMP_PCT   = 10.0            # Replacement detection: SOH jump %

# ------------------------------------------------------------------------------
# INTERNAL DEFAULTS (normally do not edit)
# ------------------------------------------------------------------------------
Q_RATED_AH    = 304.0                 # Fallback nominal capacity floor
MAX_TAIL_STRIP = 2                    # Tail-drop correction strips
SOFT_MIN_DROP_LABEL = 0.4             # Min total drop on smoothed pseudo-label
INIT_CAPACITY_CYCLES = 25             # Auto-estimate init capacity from first N charging sessions
HRLFC_WRAP_MOD      = 65536.0         # 16-bit counter wrap value (2^16)
HRLFC_VALID_MAX     = 65600.0         # Valid range + sensor slack
CAPACITY_SCALE_CANDIDATES = (0.5, 1.0, 2.0)  # x0.5/x1/x2 telemetry scaling fix
CAPACITY_CAL_MAX_ERR_PCT = 15.0      # Use calibrated baseline only when match error is within this bound
CELL_VOLT_MIN = 2.8                  # Cell minimum voltage (V)
CELL_VOLT_MAX = 3.6                  # Cell maximum voltage (V)
CELL_VOLT_NOM = 3.2                  # Cell nominal voltage (V) for series estimation
PACK_SERIES_OPTIONS = (96, 120, 208) # Known series cell configurations
PACK_CAPACITY_OPTIONS_BY_SERIES = {
    96: (104.5,),
    120: (104.5,),
    208: (150.0, 300.0, 600.0),      # 208s1p / 208s2p / 208s4p
}
PACK_CLASSIFY_W_SERIES = 0.45
PACK_CLASSIFY_W_VOLT = 0.20
PACK_CLASSIFY_W_CAP = 0.30
PACK_CLASSIFY_W_CONT = 0.05
PACK_CLASSIFY_LOCK_CONF = 0.08
PACK_CONFIG_SWITCH_MIN_CONF = 0.12
PACK_208_FORCE_2P_Q_THRESHOLD_AH = 170.0
PACK_208_FORCE_1P_Q_THRESHOLD_AH = 150.0
BASELINE_MAX_DRIFT_PCT = 5.0         # Per-run cap on baseline drift unless replacement signal is strong
STRICT_PACK_BASELINE_ENABLED = True   # Force baseline capacity from inferred pack config
PACK_USABLE_FRACTION = 1.00          # Usable fraction applied to nominal pack capacity
BMS_CALIBRATION_ENABLED = True        # Scale implied_Q_Ah by per-vehicle BMS/MY ratio when BMS columns present
BMS_INIT_CAP_OVERRIDE   = True        # Use BMS initial capacity as q_base when bms_init_cap column present
PACK_FIXED_BASELINE_AH = {
    '96s1p': 104.5,
    '120s1p': 104.5,
    '208s1p': 150.0,
    '208s2p': 300.0,
}
RUL_MIN_NEG_SLOPE = -1e-6             # Min negative slope treated as degrading
RUL_GLOBAL_DROP_TRIGGER_PCT = 2.0     # Allow global slope if total drop is meaningful
RUL_SLOPE_DISPLAY_AXIS_SCALE = 10000.0  # Show slope as % per 10k axis units
REPL_WINDOW = 8                       # Sessions per side for replacement medians
REPL_PERSIST_M = 6                    # Lookahead window for persistence
REPL_PERSIST_K = 3                    # Required confirmations in lookahead

# ------------------------------------------------------------------------------
# CONFIG PROFILES
# ------------------------------------------------------------------------------
PROFILE_DEFAULT_NAME = "conservative"
PROFILE_DEFAULT_PATH = "ev_pipeline_profiles.json"
STATE_DEFAULT_PATH = "ev_pipeline_state.pkl"
INCREMENTAL_OVERLAP_HOURS = 24.0

USER_INPUT_KEYS = (
    'SOH_EOL',
    'MIN_DELTA_SOC',
    'MIN_AH',
    'EXPECTED_CAPACITY_OPTIONS_AH',
    'REPORT_RUL_CAP_DAYS',
)

ID_CONFIG_KEYS = (
    'VEHICLE_ID_ALIASES',
    'VEHICLE_ID_ALIAS_PATH',
    'VEHICLE_ID_MODE',
)

ADVANCED_INPUT_KEYS = (
    'SOFT_MIN_DROP_XGB',
    'SOFT_MIN_DROP_LSTM',
    'REPL_Q_JUMP_PCT',
    'REPL_Q_JUMP_AH',
    'REPL_SOH_JUMP_PCT',
)

TUNABLE_CONFIG_KEYS = USER_INPUT_KEYS + ADVANCED_INPUT_KEYS

CONFIG_PROFILES = {
    'default': {
        'SOH_EOL': 80.0,
        'MIN_DELTA_SOC': 1.0,
        'MIN_AH': 1.0,
        'EXPECTED_CAPACITY_OPTIONS_AH': [104.5, 152.0, 304.0, 600.0],
        'REPORT_RUL_CAP_DAYS': 1825.0,
        'SOFT_MIN_DROP_XGB': 0.8,
        'SOFT_MIN_DROP_LSTM': 0.8,
        'REPL_Q_JUMP_PCT': 15.0,
        'REPL_Q_JUMP_AH': 35.0,
        'REPL_SOH_JUMP_PCT': 8.0,
    },
    'conservative': {
        'SOH_EOL': 80.0,
        'MIN_DELTA_SOC': 2.0,
        'MIN_AH': 1.0,
        'EXPECTED_CAPACITY_OPTIONS_AH': [104.5, 152.0, 304.0, 600.0],
        'REPORT_RUL_CAP_DAYS': 1825.0,
        'SOFT_MIN_DROP_XGB': 0.6,
        'SOFT_MIN_DROP_LSTM': 0.6,
        'REPL_Q_JUMP_PCT': 20.0,
        'REPL_Q_JUMP_AH': 35.0,
        'REPL_SOH_JUMP_PCT': 10.0,
    },
    'aggressive': {
        'SOH_EOL': 80.0,
        'MIN_DELTA_SOC': 1.0,
        'MIN_AH': 0.5,
        'EXPECTED_CAPACITY_OPTIONS_AH': [104.5, 152.0, 304.0, 600.0],
        'REPORT_RUL_CAP_DAYS': 1825.0,
        'SOFT_MIN_DROP_XGB': 1.0,
        'SOFT_MIN_DROP_LSTM': 1.0,
        'REPL_Q_JUMP_PCT': 12.0,
        'REPL_Q_JUMP_AH': 25.0,
        'REPL_SOH_JUMP_PCT': 6.0,
    },
}

# ------------------------------------------------------------------------------
# SEGMENTATION CONSTANTS  (from data-cleaning notebook)
# ------------------------------------------------------------------------------
# State-machine thresholds
SEG_EPS_CURR_ENTRY = 0.2   # Min |chargingCurrent| A to enter CHARGING state
SEG_EPS_CURR_EXIT  = 2.0   # Max |chargingCurrent| A to leave CHARGING state
SEG_EPS_SPEED      = 1.0   # Min vehicleSpeed km/h to classify as DRIVING
SEG_MAX_GAP_MIN    = 10.0  # UTC gap (minutes) that resets state to OTHERS

# Columns used by state machine
SEG_CHG_STATUS  = 'chargingStatus'
SEG_CRANK_STATUS= 'crankStatus'
SEG_CHG_CURR    = 'chargingCurrent'
SEG_VEH_SPEED   = 'vehicleSpeed'


def _coerce_profile_value(key, value):
    if key == 'EXPECTED_CAPACITY_OPTIONS_AH':
        if isinstance(value, (list, tuple)):
            return tuple(float(v) for v in value)
        raise ValueError("EXPECTED_CAPACITY_OPTIONS_AH must be a list/tuple of numbers")
    if key == 'REPORT_RUL_CAP_DAYS':
        return float(value)
    return float(value)


def _coerce_id_config_value(key, value):
    if key == 'VEHICLE_ID_ALIASES':
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("VEHICLE_ID_ALIASES must be a JSON object/dict")
        return value
    if key == 'VEHICLE_ID_ALIAS_PATH':
        return None if value in (None, '') else str(value)
    if key == 'VEHICLE_ID_MODE':
        mode = str(value).strip().lower()
        valid = {'auto', 'latest_device_per_file'}
        if mode not in valid:
            raise ValueError(f"VEHICLE_ID_MODE must be one of {sorted(valid)}")
        return mode
    return value


def _validate_runtime_config():
    if not (0.0 < float(SOH_EOL) < 100.0):
        raise ValueError(f"Invalid SOH_EOL={SOH_EOL}. Expected 0 < SOH_EOL < 100.")
    if float(MIN_DELTA_SOC) < 0.0:
        raise ValueError(f"Invalid MIN_DELTA_SOC={MIN_DELTA_SOC}. Expected >= 0.")
    if float(MIN_AH) <= 0.0:
        raise ValueError(f"Invalid MIN_AH={MIN_AH}. Expected > 0.")
    if float(REPORT_RUL_CAP_DAYS) < 365.0:
        raise ValueError(f"Invalid REPORT_RUL_CAP_DAYS={REPORT_RUL_CAP_DAYS}. Expected >= 365.")

    caps = tuple(float(v) for v in EXPECTED_CAPACITY_OPTIONS_AH)
    if len(caps) == 0 or any((not np.isfinite(v)) or (v <= 0) for v in caps):
        raise ValueError("EXPECTED_CAPACITY_OPTIONS_AH must contain positive finite values.")

    for k in ['SOFT_MIN_DROP_XGB', 'SOFT_MIN_DROP_LSTM']:
        v = float(globals()[k])
        if (not np.isfinite(v)) or (v < 0.0):
            raise ValueError(f"Invalid {k}={v}. Expected finite >= 0.")

    for k in ['REPL_Q_JUMP_PCT', 'REPL_Q_JUMP_AH', 'REPL_SOH_JUMP_PCT']:
        v = float(globals()[k])
        if (not np.isfinite(v)) or (v <= 0.0):
            raise ValueError(f"Invalid {k}={v}. Expected finite > 0.")


def _load_profiles_from_json(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    # Handle UTF-8 BOM transparently (common when JSON is edited in Windows tools).
    with p.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Profile file must be a JSON object: {path}")

    # Supports either:
    #   1) {"conservative": {...}, "aggressive": {...}}
    #   2) {"SOH_EOL": 80, ...}  -> treated as "custom"
    if data and all(isinstance(v, dict) for v in data.values()):
        return data
    return {'custom': data}


def apply_config_profile(profile_name: str = PROFILE_DEFAULT_NAME, profile_path: str = None):
    profiles = {k: dict(v) for k, v in CONFIG_PROFILES.items()}

    if profile_path:
        external = _load_profiles_from_json(profile_path)
        for pn, cfg in external.items():
            if isinstance(cfg, dict):
                profiles[pn] = cfg

    if profile_name not in profiles:
        choices = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown profile '{profile_name}'. Available: {choices}")

    cfg = profiles[profile_name]
    for k in TUNABLE_CONFIG_KEYS:
        if k in cfg:
            globals()[k] = _coerce_profile_value(k, cfg[k])
    for k in ID_CONFIG_KEYS:
        if k in cfg:
            globals()[k] = _coerce_id_config_value(k, cfg[k])
        elif k.lower() in cfg:
            globals()[k] = _coerce_id_config_value(k, cfg[k.lower()])

    _validate_runtime_config()

    print(f"  Config profile: {profile_name}")
    if profile_path:
        print(f"  Profile file  : {profile_path}")
    for k in USER_INPUT_KEYS:
        print(f"    {k} = {globals()[k]}")
    for k in ADVANCED_INPUT_KEYS:
        print(f"    {k} = {globals()[k]}")
    if VEHICLE_ID_ALIASES:
        print(f"    VEHICLE_ID_ALIASES = {len(VEHICLE_ID_ALIASES)} configured entries")
    if VEHICLE_ID_ALIAS_PATH:
        print(f"    VEHICLE_ID_ALIAS_PATH = {VEHICLE_ID_ALIAS_PATH}")
    print(f"    VEHICLE_ID_MODE = {VEHICLE_ID_MODE}")


def _resolve_state_path(plot_path: str = None, state_path: str = None) -> Path:
    if state_path:
        return Path(state_path)
    if plot_path:
        p = Path(plot_path)
        return p.parent / STATE_DEFAULT_PATH
    return Path(__file__).resolve().parent / STATE_DEFAULT_PATH


def _load_pipeline_state(state_path: Path):
    if not state_path.exists():
        return None
    try:
        state = pd.read_pickle(state_path)
        if isinstance(state, dict):
            return state
    except Exception as e:
        print(f"    [WARN] Could not load state file {state_path}: {e}")
    return None


def _save_pipeline_state(state_path: Path, state: dict):
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(state, state_path)
        print(f"    State saved -> {state_path}")
    except Exception as e:
        print(f"    [WARN] Could not save state file {state_path}: {e}")


def _merge_sessions_cached(old_sessions: pd.DataFrame, new_sessions: pd.DataFrame) -> pd.DataFrame:
    if old_sessions is None or len(old_sessions) == 0:
        out = new_sessions.copy()
    elif new_sessions is None or len(new_sessions) == 0:
        out = old_sessions.copy()
    else:
        out = pd.concat([old_sessions, new_sessions], ignore_index=True, sort=False)

    if len(out) == 0:
        return out

    keys = [c for c in ['vehicle_id', 'session_id', 'start_utc', 'end_utc'] if c in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys, keep='last')

    sort_cols = [c for c in ['vehicle_id', 'start_utc', 'hrlfc_mid', 'session_id'] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _state_safe_xgb_results(xgb_results: dict) -> dict:
    out = {}
    for vid, res in (xgb_results or {}).items():
        if not isinstance(res, dict):
            continue
        rr = dict(res)
        rr['model'] = None
        rr['scaler'] = None
        out[vid] = rr
    return out


def _state_safe_lstm_results(lstm_results: dict) -> dict:
    out = {}
    keep = {'lookback', 'soh_seq', 'hrlfc_seq', 'soh_pred'}
    for vid, res in (lstm_results or {}).items():
        if not isinstance(res, dict):
            continue
        rr = {k: res.get(k) for k in keep if k in res}
        rr['model'] = None
        rr['scaler'] = None
        out[vid] = rr
    return out


def _extract_cached_init_capacity_map(rul_all_cached: dict) -> dict:
    out = {}
    if not isinstance(rul_all_cached, dict):
        return out
    for vid, rr in rul_all_cached.items():
        if not isinstance(rr, dict):
            continue
        v = rr.get('q_base_for_soh_ah', rr.get('init_capacity_ah', np.nan))
        v = pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0]
        if np.isfinite(v) and (v > 0):
            out[vid] = float(v)
    return out


def _parse_pack_config_guess(cfg_text: str):
    s = str(cfg_text).strip().lower()
    m = re.match(r'^(\d+)s(\d+)p$', s)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))


def _capacity_options_from_pack_config(cfg_text: str):
    s_cells, p_count = _parse_pack_config_guess(cfg_text)
    if np.isfinite(s_cells) and np.isfinite(p_count) and int(s_cells) == 208:
        return (304.0,) if int(p_count) >= 2 else (152.0,)
    if np.isfinite(s_cells):
        return tuple(float(v) for v in PACK_CAPACITY_OPTIONS_BY_SERIES.get(int(s_cells), EXPECTED_CAPACITY_OPTIONS_AH))
    return tuple(float(v) for v in EXPECTED_CAPACITY_OPTIONS_AH)


def _strict_baseline_from_pack_config(cfg_text: str):
    cfg = str(cfg_text).strip().lower()
    nom = pd.to_numeric(pd.Series([PACK_FIXED_BASELINE_AH.get(cfg, np.nan)]), errors='coerce').iloc[0]
    if (not np.isfinite(nom)) or (nom <= 0):
        return np.nan, np.nan
    usable = float(nom) * float(PACK_USABLE_FRACTION)
    return float(nom), float(usable)


def _extract_cached_pack_context_map(rul_all_cached: dict) -> dict:
    out = {}
    if not isinstance(rul_all_cached, dict):
        return out
    for vid, rr in rul_all_cached.items():
        if not isinstance(rr, dict):
            continue
        out[vid] = {
            'pack_config_guess': str(rr.get('pack_config_guess', 'unknown')),
            'pack_score_confidence': pd.to_numeric(pd.Series([rr.get('pack_score_confidence', np.nan)]), errors='coerce').iloc[0],
            'config_epoch_id': int(pd.to_numeric(pd.Series([rr.get('config_epoch_id', 0)]), errors='coerce').fillna(0).iloc[0]),
        }
    return out


def _normalize_vehicle_id_text(value) -> str:
    s = str(value).strip()
    s = re.sub(r'^(\d+)\.0+$', r'\1', s)
    return s


def _vehicle_id_lookup_keys(value) -> list:
    s = _normalize_vehicle_id_text(value)
    keys = [s] if s else []
    m = re.fullmatch(r'IMEI[_-]?(\d{10,17})', s, flags=re.IGNORECASE)
    if m:
        keys.append(m.group(1))
    elif re.fullmatch(r'\d{10,17}', s):
        keys.append(f"IMEI_{s}")
    return list(dict.fromkeys(k for k in keys if k))


def _normalize_vehicle_alias_config(raw_aliases) -> dict:
    """
    Build alias->canonical map.
    Supported formats:
      {"BUS_001": ["old_device", "new_device"]}
      {"old_device": "BUS_001", "new_device": "BUS_001"}
      {"VEHICLE_ID_ALIASES": {...}}
    """
    if not raw_aliases:
        return {}
    if not isinstance(raw_aliases, dict):
        raise ValueError("Vehicle ID aliases must be a JSON object/dict")

    for wrapper_key in ('VEHICLE_ID_ALIASES', 'vehicle_id_aliases', 'aliases'):
        if wrapper_key in raw_aliases and isinstance(raw_aliases[wrapper_key], dict):
            raw_aliases = raw_aliases[wrapper_key]
            break

    alias_map = {}
    for key, value in raw_aliases.items():
        key_norm = _normalize_vehicle_id_text(key)
        if not key_norm:
            continue

        if isinstance(value, (list, tuple, set)):
            canonical = key_norm
            for key_variant in _vehicle_id_lookup_keys(canonical):
                alias_map[key_variant] = canonical
            for alias in value:
                for key_variant in _vehicle_id_lookup_keys(alias):
                    alias_map[key_variant] = canonical
        elif isinstance(value, dict):
            canonical = _normalize_vehicle_id_text(
                value.get('vehicle_id') or value.get('canonical') or value.get('target') or key_norm
            )
            for key_variant in _vehicle_id_lookup_keys(canonical):
                alias_map[key_variant] = canonical
            aliases = value.get('aliases') or value.get('device_ids') or value.get('deviceIds') or []
            for alias in aliases:
                for key_variant in _vehicle_id_lookup_keys(alias):
                    alias_map[key_variant] = canonical
        elif value is not None:
            canonical = _normalize_vehicle_id_text(value)
            if canonical:
                for key_variant in _vehicle_id_lookup_keys(key_norm):
                    alias_map[key_variant] = canonical
                for key_variant in _vehicle_id_lookup_keys(canonical):
                    alias_map[key_variant] = canonical

    return alias_map


def _load_vehicle_id_aliases(data_path: str = None) -> dict:
    alias_map = _normalize_vehicle_alias_config(VEHICLE_ID_ALIASES)

    candidate_paths = []
    if VEHICLE_ID_ALIAS_PATH:
        candidate_paths.append(Path(VEHICLE_ID_ALIAS_PATH))
    else:
        if data_path:
            p = Path(data_path)
            base_dir = p if p.is_dir() else p.parent
            candidate_paths.append(base_dir / "vehicle_id_aliases.json")
        candidate_paths.append(Path(__file__).resolve().parent / "vehicle_id_aliases.json")

    seen = set()
    for candidate in candidate_paths:
        key = str(candidate.resolve()).lower() if candidate.exists() else str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        with candidate.open('r', encoding='utf-8-sig') as f:
            file_aliases = json.load(f)
        file_map = _normalize_vehicle_alias_config(file_aliases)
        alias_map.update(file_map)
        print(f"    Vehicle ID alias file loaded: {candidate} ({len(file_map)} aliases)")

    return alias_map


def _apply_vehicle_id_aliases(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    if df is None or 'vehicle_id' not in df.columns or not alias_map:
        return df

    before = df['vehicle_id'].astype(str).map(_normalize_vehicle_id_text)
    after = before.map(lambda v: next((alias_map[k] for k in _vehicle_id_lookup_keys(v) if k in alias_map), v))
    changed = before.ne(after)
    if changed.any():
        df_attrs = dict(getattr(df, 'attrs', {}) or {})
        df = df.copy()
        df.attrs.update(df_attrs)
        df['vehicle_id_raw'] = before
        df['vehicle_id'] = after
        print(
            "    Vehicle ID aliases applied: "
            f"{int(changed.sum()):,} rows remapped | "
            f"{before.nunique(dropna=True)} raw IDs -> {after.nunique(dropna=True)} vehicle IDs"
        )
        preview = (
            pd.DataFrame({'raw_id': before[changed], 'vehicle_id': after[changed]})
            .drop_duplicates()
            .head(10)
        )
        for _, row in preview.iterrows():
            print(f"      {row['raw_id']} -> {row['vehicle_id']}")
    return df


def _canonical_vehicle_id(value, alias_map: dict) -> str:
    if not alias_map:
        return _normalize_vehicle_id_text(value)
    for key in _vehicle_id_lookup_keys(value):
        if key in alias_map:
            return alias_map[key]
    return _normalize_vehicle_id_text(value)


def _select_telematics_device_id_column(df: pd.DataFrame):
    if df is None or len(df.columns) == 0:
        return None

    exact_priority = [
        'deviceId', 'device_id', 'device_imei', 'deviceimei',
        'IMEI', 'imei',
    ]
    for c in exact_priority:
        if c in df.columns:
            return c

    candidates = []
    for c in df.columns:
        n = _norm_col(c)
        if n in {'deviceid', 'deviceimei', 'imei'} or ('imei' in n):
            candidates.append(c)
    return candidates[0] if candidates else None


def _latest_device_id_for_group(g: pd.DataFrame, device_col: str):
    device_ids = _clean_id_series(g[device_col])
    valid = device_ids.notna()
    if not valid.any():
        return np.nan

    work = pd.DataFrame({
        '_device_id': device_ids,
        '_row_order': np.arange(len(g), dtype=float),
    }, index=g.index)

    if '_utc_num' in g.columns:
        work['_sort_time'] = pd.to_numeric(g['_utc_num'], errors='coerce')
    elif 'DateTime' in g.columns:
        dt = pd.to_datetime(g['DateTime'], errors='coerce')
        work['_sort_time'] = dt.view('int64').where(dt.notna(), np.nan)
    else:
        work['_sort_time'] = np.nan

    work = work.loc[valid].copy()
    if work['_sort_time'].notna().any():
        work['_sort_time'] = work['_sort_time'].fillna(-np.inf)
        work = work.sort_values(['_sort_time', '_row_order'], kind='mergesort')
    else:
        work = work.sort_values('_row_order', kind='mergesort')
    return _normalize_vehicle_id_text(work['_device_id'].iloc[-1])


def _apply_latest_device_per_file_identity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat each CSV file as one physical vehicle and use the latest deviceId in
    that file as the stable vehicle_id for all rows from that file.
    """
    if VEHICLE_ID_MODE != 'latest_device_per_file':
        return df

    device_col = _select_telematics_device_id_column(df)
    if not device_col:
        print("    [WARN] VEHICLE_ID_MODE=latest_device_per_file but no device ID column was found.")
        return df

    group_col = '__source_path' if '__source_path' in df.columns else (
        '__source_file' if '__source_file' in df.columns else None
    )
    if group_col is None:
        group_keys = pd.Series('__single_input__', index=df.index)
    else:
        group_keys = df[group_col].astype(str)

    df = df.copy()
    df['device_id_raw'] = _clean_id_series(df[device_col])

    latest_by_file = {}
    for source_key, idx in group_keys.groupby(group_keys).groups.items():
        latest = _latest_device_id_for_group(df.loc[idx], device_col)
        if pd.notna(latest) and str(latest).strip():
            latest_by_file[source_key] = latest

    if not latest_by_file:
        print("    [WARN] No usable latest device IDs found; keeping existing vehicle ID resolution.")
        return df

    file_vehicle_id = group_keys.map(latest_by_file)
    fallback = df['vehicle_id'] if 'vehicle_id' in df.columns else df['device_id_raw']
    df['vehicle_id'] = file_vehicle_id.combine_first(fallback).fillna('unknown_vehicle').astype(str)

    runtime_alias_map = {}
    alias_pairs = pd.DataFrame({
        'source_key': group_keys,
        'raw_id': df['device_id_raw'],
        'vehicle_id': df['vehicle_id'],
    }).dropna().drop_duplicates()
    for _, row in alias_pairs.iterrows():
        raw_id = _normalize_vehicle_id_text(row['raw_id'])
        canonical_id = _normalize_vehicle_id_text(row['vehicle_id'])
        if raw_id and canonical_id:
            for key in _vehicle_id_lookup_keys(raw_id):
                runtime_alias_map[key] = canonical_id
            for key in _vehicle_id_lookup_keys(canonical_id):
                runtime_alias_map[key] = canonical_id
    df.attrs['vehicle_id_alias_map'] = runtime_alias_map

    print(
        "    Vehicle ID mode: latest_device_per_file | "
        f"{len(latest_by_file):,} file(s), {df['device_id_raw'].nunique(dropna=True):,} raw device ID(s), "
        f"{df['vehicle_id'].nunique(dropna=True):,} vehicle ID(s)"
    )
    for source_key, latest in list(latest_by_file.items())[:10]:
        print(f"      {Path(str(source_key)).name} -> vehicle_id={latest}")
    if len(latest_by_file) > 10:
        print(f"      ... and {len(latest_by_file) - 10} more")
    return df


def _extract_runtime_vehicle_alias_map(df: pd.DataFrame) -> dict:
    if df is None or 'vehicle_id' not in df.columns:
        return {}

    alias_map = {}
    if isinstance(getattr(df, 'attrs', None), dict):
        alias_map.update(df.attrs.get('vehicle_id_alias_map') or {})

    raw_cols = [c for c in ['device_id_raw', 'vehicle_id_raw'] if c in df.columns]
    if not raw_cols:
        return alias_map

    canonical = df['vehicle_id'].astype(str).map(_normalize_vehicle_id_text)
    for raw_col in raw_cols:
        raw = _clean_id_series(df[raw_col])
        pairs = pd.DataFrame({'raw': raw, 'canonical': canonical}).dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            raw_id = _normalize_vehicle_id_text(row['raw'])
            canonical_id = _normalize_vehicle_id_text(row['canonical'])
            if raw_id and canonical_id:
                for key in _vehicle_id_lookup_keys(raw_id):
                    alias_map[key] = canonical_id
                for key in _vehicle_id_lookup_keys(canonical_id):
                    alias_map[key] = canonical_id
    return alias_map


def _remap_vehicle_ids_in_frame(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or 'vehicle_id' not in df.columns or not alias_map:
        return df
    out = df.copy()
    out['vehicle_id'] = out['vehicle_id'].astype(str).map(lambda v: _canonical_vehicle_id(v, alias_map))
    return out


def _payload_vehicle_score(payload) -> int:
    if isinstance(payload, dict):
        sessions = payload.get('sessions')
        if isinstance(sessions, pd.DataFrame):
            return len(sessions)
        n_sessions = payload.get('n_sessions')
        if pd.notna(n_sessions):
            try:
                return int(n_sessions)
            except Exception:
                return 0
    return 0


def _remap_vehicle_keyed_dict(data: dict, alias_map: dict) -> dict:
    if not isinstance(data, dict) or not alias_map:
        return data
    out = {}
    for key, payload in data.items():
        new_key = _canonical_vehicle_id(key, alias_map)
        new_payload = payload
        if isinstance(payload, dict):
            new_payload = dict(payload)
            if isinstance(new_payload.get('sessions'), pd.DataFrame):
                new_payload['sessions'] = _remap_vehicle_ids_in_frame(new_payload['sessions'], alias_map)
            if 'vehicle_id' in new_payload:
                new_payload['vehicle_id'] = new_key

        if new_key not in out or _payload_vehicle_score(new_payload) >= _payload_vehicle_score(out[new_key]):
            out[new_key] = new_payload
    return out


def _remap_pipeline_state_vehicle_ids(state: dict, alias_map: dict) -> dict:
    if not isinstance(state, dict) or not alias_map:
        return state

    remapped = dict(state)
    if isinstance(remapped.get('sessions'), pd.DataFrame):
        remapped['sessions'] = _remap_vehicle_ids_in_frame(remapped['sessions'], alias_map)
    for key in ['xgb_results', 'lstm_results', 'rul_all']:
        if isinstance(remapped.get(key), dict):
            remapped[key] = _remap_vehicle_keyed_dict(remapped[key], alias_map)
    if isinstance(remapped.get('replacement_events'), pd.DataFrame):
        remapped['replacement_events'] = _remap_vehicle_ids_in_frame(remapped['replacement_events'], alias_map)
    return remapped


def _get_cached_vehicle_ids_from_state(state: dict) -> set:
    """Collect cached vehicle IDs from state payload."""
    out = set()
    if not isinstance(state, dict):
        return out

    sessions = state.get('sessions')
    if isinstance(sessions, pd.DataFrame) and 'vehicle_id' in sessions.columns:
        vals = sessions['vehicle_id'].dropna().astype(str).map(_normalize_vehicle_id_text)
        out.update(v for v in vals.tolist() if v)

    rul_all = state.get('rul_all', {})
    if isinstance(rul_all, dict):
        out.update(_normalize_vehicle_id_text(k) for k in rul_all.keys())

    xgb_results = state.get('xgb_results', {})
    if isinstance(xgb_results, dict):
        out.update(_normalize_vehicle_id_text(k) for k in xgb_results.keys())

    return {v for v in out if v}


def _scan_vehicle_ids_from_input_path(path_str: str) -> set:
    """
    Best-effort vehicle-id scan from input file/folder before full load.
    Reads only ID-like columns from CSVs.
    """
    if not path_str:
        return set()

    p = Path(path_str)
    if not p.exists():
        return set()

    csv_files = []
    if p.is_file():
        csv_files = [p]
    elif p.is_dir():
        csv_files = sorted(p.glob("**/*.csv")) + sorted(p.glob("**/*.CSV"))
        dedup = {}
        for f in csv_files:
            dedup[str(f.resolve()).lower()] = f
        csv_files = sorted(dedup.values(), key=lambda x: str(x).lower())

    ids = set()
    id_alias_norm = {
        'vehicleid', 'imei', 'deviceimei', 'deviceid', 'vin'
    }

    for f in csv_files:
        try:
            header = list(pd.read_csv(f, nrows=0).columns)
        except Exception:
            header = []

        use_cols = []
        for c in header:
            n = _norm_col(c)
            if n in id_alias_norm or ('imei' in n) or ('vehicle' in n and 'id' in n) or ('vin' in n):
                use_cols.append(c)

        if use_cols:
            try:
                tmp = pd.read_csv(f, usecols=use_cols, dtype=str, low_memory=True)
                for c in use_cols:
                    if c in tmp.columns:
                        vv = _clean_id_series(tmp[c]).dropna().astype(str).map(_normalize_vehicle_id_text).unique().tolist()
                        ids.update(v for v in vv if v)
            except Exception:
                pass

        if not ids:
            m = re.search(r'IMEI[_-]?(\d{10,17})', f.name, flags=re.IGNORECASE)
            if m:
                ids.add(_normalize_vehicle_id_text(m.group(1)))
            else:
                for tok in re.findall(r'\d{10,17}', f.stem):
                    ids.add(_normalize_vehicle_id_text(tok))

    return {v for v in ids if v}


def _filter_outputs_by_vehicle_scope(
    xgb_results: dict,
    lstm_results: dict,
    rul_all: dict,
    replacement_events: pd.DataFrame,
    vehicle_ids
):
    """Return reporting views filtered to a vehicle-id scope, leaving originals untouched."""
    if not vehicle_ids:
        return xgb_results, lstm_results, rul_all, replacement_events

    scope = {_normalize_vehicle_id_text(v) for v in vehicle_ids if pd.notna(v)}
    if not scope:
        return xgb_results, lstm_results, rul_all, replacement_events

    xgb_view = {k: v for k, v in (xgb_results or {}).items() if _normalize_vehicle_id_text(k) in scope}
    lstm_view = {k: v for k, v in (lstm_results or {}).items() if _normalize_vehicle_id_text(k) in scope}
    rul_view = {k: v for k, v in (rul_all or {}).items() if _normalize_vehicle_id_text(k) in scope}

    repl_view = replacement_events
    if isinstance(replacement_events, pd.DataFrame) and len(replacement_events) > 0 and 'vehicle_id' in replacement_events.columns:
        repl_tmp = replacement_events.copy()
        repl_view = repl_tmp[repl_tmp['vehicle_id'].map(_normalize_vehicle_id_text).isin(scope)].reset_index(drop=True)

    return xgb_view, lstm_view, rul_view, repl_view


def _guess_vehicle_scope_from_input_path(path_str: str, known_vehicle_ids=None):
    """
    Best-effort scope inference from input file name/path for cached-only incremental runs.
    """
    if not path_str:
        return set()

    known_norm = {_normalize_vehicle_id_text(v) for v in (known_vehicle_ids or [])}
    p = Path(path_str)
    text = str(p)
    stem = p.stem

    tokens = set()
    if re.fullmatch(r'\d{10,17}(?:\.0+)?', stem):
        tokens.add(_normalize_vehicle_id_text(stem))
    for m in re.findall(r'\d{10,17}', text):
        tokens.add(_normalize_vehicle_id_text(m))
    for m in re.findall(r'IMEI[_-]?(\d{10,17})', text, flags=re.IGNORECASE):
        tokens.add(_normalize_vehicle_id_text(m))

    if not known_norm:
        return tokens

    return known_norm.intersection(tokens)


def _finite_series(series: pd.Series) -> pd.Series:
    """Coerce to numeric and remove +/-inf."""
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _pick_axis_col(df: pd.DataFrame) -> str:
    """
    Prefer hrlfc_mid only when it has enough finite variation.
    Fall back to start_utc, then synthetic session index.
    """
    if 'hrlfc_mid' in df.columns:
        h = _finite_series(df['hrlfc_mid'])
        min_needed = max(5, int(0.6 * max(len(df), 1)))
        if h.notna().sum() >= min_needed and np.isfinite(h.max() - h.min()) and (h.max() - h.min()) > 0:
            return 'hrlfc_mid'
    if 'start_utc' in df.columns:
        t = _finite_series(df['start_utc'])
        min_needed = max(5, int(0.6 * max(len(df), 1)))
        if t.notna().sum() >= min_needed and np.isfinite(t.max() - t.min()) and (t.max() - t.min()) > 0:
            return 'start_utc'
    return '__session_idx'


def _soft_monotone_curve(values, x=None, min_total_drop=0.0, smooth_window=9):
    """
    Build a descending curve without staircase artifacts:
      1) centered rolling mean smoothing,
      2) monotone projection (non-increasing),
      3) optional weak linear drift floor to avoid fully flat trajectories.
    """
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return np.zeros(len(s), dtype=float)
    s = s.interpolate(limit_direction='both').bfill().ffill()

    n = len(s)
    if n <= 1:
        return s.values.astype(float)

    # odd window, bounded by sequence length
    w = min(int(smooth_window), n if n % 2 == 1 else max(1, n - 1))
    w = max(3, w)
    smooth = s.rolling(window=w, min_periods=1, center=True).mean().values.astype(float)

    if x is not None:
        xv = _finite_series(pd.Series(x)).values.astype(float)
        if len(xv) != n or np.isfinite(xv).sum() < 2:
            x_norm = np.linspace(0.0, 1.0, n, dtype=float)
        else:
            xmin = np.nanmin(xv)
            xmax = np.nanmax(xv)
            span = xmax - xmin
            x_norm = (xv - xmin) / span if np.isfinite(span) and span > 0 else np.linspace(0.0, 1.0, n, dtype=float)
            x_norm = np.clip(x_norm, 0.0, 1.0)
    else:
        x_norm = np.linspace(0.0, 1.0, n, dtype=float)

    mono = smooth.copy()
    for i in range(1, n):
        if mono[i] > mono[i - 1]:
            mono[i] = mono[i - 1]

    drop_now = float(mono[0] - mono[-1]) if np.isfinite(mono[0] - mono[-1]) else 0.0
    target_drop = max(float(min_total_drop), drop_now)
    if target_drop > 0:
        drift_line = mono[0] - target_drop * x_norm
        mono = np.minimum(mono, drift_line)
        for i in range(1, n):
            if mono[i] > mono[i - 1]:
                mono[i] = mono[i - 1]

    return mono


def _unwrap_hrlfc_counter(values, wrap_mod=HRLFC_WRAP_MOD, reset_margin=500.0):
    """
    Unwrap a modulo counter (e.g., 16-bit 0..65535) into a monotone series.
    Adds +wrap_mod on true wrap events instead of compounding by prior cumulative value.
    """
    arr = pd.to_numeric(pd.Series(values), errors='coerce').values.astype(float)
    out = np.full(len(arr), np.nan, dtype=float)

    wraps = 0.0
    last_raw = np.nan
    for i, v in enumerate(arr):
        if np.isnan(v):
            continue

        if np.isfinite(last_raw):
            # Wrap when current raw value drops sharply from the previous raw value.
            if v + reset_margin < last_raw:
                wraps += wrap_mod

        out[i] = v + wraps
        last_raw = v

    return out


def _calibrate_capacity_ah(q_raw, options=None, scales=None):
    """
    Map raw implied capacity to nearest expected nominal capacity under common
    telemetry scaling factors (e.g., x2 or x0.5).
    Returns: (q_calibrated, nominal_capacity, chosen_scale, rel_error_pct)
    """
    options = tuple(float(v) for v in (options if options is not None else EXPECTED_CAPACITY_OPTIONS_AH))
    scales = tuple(float(v) for v in (scales if scales is not None else CAPACITY_SCALE_CANDIDATES))
    q_raw = float(q_raw)
    if (not np.isfinite(q_raw)) or (q_raw <= 0) or (len(options) == 0) or (len(scales) == 0):
        return np.nan, np.nan, np.nan, np.inf
    best = None
    for s in scales:
        q_scaled = q_raw * s
        for nom in options:
            err = abs(q_scaled - nom) / max(nom, 1e-9)
            # Prefer lower relative error; on ties prefer scale closest to 1.
            tie_penalty = abs(np.log2(max(s, 1e-12)))
            score = err + 1e-6 * tie_penalty
            cand = (score, err, q_scaled, float(nom), float(s))
            if best is None or cand[0] < best[0]:
                best = cand
    _, err, q_cal, nom, scale = best
    # When within error threshold, snap to nominal spec (avoids ~4% systematic sensor underread)
    use_nom = np.isfinite(err) and (err <= CAPACITY_CAL_MAX_ERR_PCT / 100.0)
    return (float(nom) if use_nom else q_cal), nom, scale, err * 100.0


def _infer_pack_config_options(g: pd.DataFrame, q_anchor=np.nan, q_prev=np.nan):
    """
    Score candidate pack configurations using voltage, cell-ratio series estimate,
    capacity fit, and continuity against previous stable baseline.
    """
    default_opts = tuple(float(v) for v in EXPECTED_CAPACITY_OPTIONS_AH)
    info = {
        'series_cells': np.nan,
        'parallel_count': np.nan,
        'voltage_ref_v': np.nan,
        'series_est_cellratio': np.nan,
        'series_est_nominal': np.nan,
        'capacity_options': default_opts,
        'series_source': 'unknown',
        'config_family': 'unknown',
        'config_guess': 'unknown',
        'chosen_nominal_ah': np.nan,
        'score_confidence': np.nan,
        'options_source': 'default',
        'q_data_median_ah': np.nan,
        'q_data_count': 0,
    }

    if g is None or len(g) == 0:
        return info

    quality = pd.Series(True, index=g.index, dtype=bool)
    if 'delta_soc_pct' in g.columns:
        quality &= _finite_series(g['delta_soc_pct']) >= 5.0
    if 'soc_end' in g.columns:
        quality &= _finite_series(g['soc_end']) >= 70.0

    v_pack = None
    v_source = 'none'
    for col in ['pack_v_max', 'pack_v_mean', 'pack_v_min']:
        if col not in g.columns:
            continue
        vals = _finite_series(g[col])
        vals_q = vals.where(quality)
        use = vals_q if vals_q.notna().sum() >= 5 else vals
        use = _finite_series(use)
        if use.notna().sum() == 0:
            continue
        v_pack = use
        v_source = col
        break
    if v_pack is None:
        return info

    v_ref = float(v_pack.quantile(0.90))
    if (not np.isfinite(v_ref)) or (v_ref <= 0):
        return info

    s_nom_est = v_ref / float(CELL_VOLT_NOM)
    s_cell_est = np.nan
    if ('cell_v_max_peak' in g.columns) and ('pack_v_max' in g.columns):
        p = _finite_series(g['pack_v_max'].where(quality))
        c = _finite_series(g['cell_v_max_peak'].where(quality))
        m = (
            p.notna() &
            c.notna() &
            (c >= 0.90 * float(CELL_VOLT_MIN)) &
            (c <= 1.15 * float(CELL_VOLT_MAX))
        )
        if int(m.sum()) >= 5:
            s_cell_est = float(np.nanmedian((p[m] / c[m]).values.astype(float)))
    if not np.isfinite(s_cell_est):
        if ('cell_v_max_mean' in g.columns) and ('cell_v_min_mean' in g.columns) and ('pack_v_mean' in g.columns):
            p = _finite_series(g['pack_v_mean'].where(quality))
            cmx = _finite_series(g['cell_v_max_mean'].where(quality))
            cmn = _finite_series(g['cell_v_min_mean'].where(quality))
            cavg = (cmx + cmn) / 2.0
            m = (
                p.notna() &
                cavg.notna() &
                (cavg >= 0.90 * float(CELL_VOLT_MIN)) &
                (cavg <= 1.15 * float(CELL_VOLT_MAX))
            )
            if int(m.sum()) >= 5:
                s_cell_est = float(np.nanmedian((p[m] / cavg[m]).values.astype(float)))

    q_q = _finite_series(g['implied_Q_Ah'].where(quality)) if 'implied_Q_Ah' in g.columns else pd.Series(dtype=float)
    q_q = q_q.where(q_q > 0)
    q_q_count = int(q_q.notna().sum())
    q_data_med = float(q_q.median()) if q_q_count > 0 else np.nan
    # Prefer measured quality-session evidence over prior anchors.
    q_evidence = q_data_med if np.isfinite(q_data_med) else (float(q_anchor) if np.isfinite(q_anchor) and (float(q_anchor) > 0) else np.nan)

    candidates = [
        {'series': 96, 'parallel': 1, 'nom_ah': 104.5},
        {'series': 120, 'parallel': 1, 'nom_ah': 104.5},
        {'series': 208, 'parallel': 1, 'nom_ah': 152.0},
        {'series': 208, 'parallel': 2, 'nom_ah': 304.0},
    ]
    scored = []
    for cand in candidates:
        series = float(cand['series'])
        nom = float(cand['nom_ah'])

        errs_series = []
        if np.isfinite(s_cell_est):
            errs_series.append(abs(s_cell_est - series) / max(series, 1e-9))
        if np.isfinite(s_nom_est):
            errs_series.append(abs(s_nom_est - series) / max(series, 1e-9))
        series_err = min(errs_series) if errs_series else 1.0

        v_lo = float(CELL_VOLT_MIN) * series
        v_hi = float(CELL_VOLT_MAX) * series
        if v_lo <= v_ref <= v_hi:
            v_err = 0.0
        elif v_ref < v_lo:
            v_err = (v_lo - v_ref) / max(v_lo, 1e-9)
        else:
            v_err = (v_ref - v_hi) / max(v_hi, 1e-9)

        cap_err = 0.5
        if np.isfinite(q_evidence) and (q_evidence > 0):
            cap_err = min(
                abs((q_evidence * float(s)) - nom) / max(nom, 1e-9)
                for s in CAPACITY_SCALE_CANDIDATES
            )

        cont_err = 0.0
        if np.isfinite(q_prev) and (float(q_prev) > 0):
            cont_err = abs(float(q_prev) - nom) / max(nom, 1e-9)

        total_err = (
            float(PACK_CLASSIFY_W_SERIES) * series_err +
            float(PACK_CLASSIFY_W_VOLT) * v_err +
            float(PACK_CLASSIFY_W_CAP) * cap_err +
            float(PACK_CLASSIFY_W_CONT) * cont_err
        )
        scored.append((total_err, cand, series_err, v_err, cap_err, cont_err))

    scored.sort(key=lambda x: x[0])
    best = scored[0]
    second = scored[1] if len(scored) > 1 else None
    conf = (second[0] - best[0]) if second is not None else np.nan
    chosen = best[1]
    options_source = 'scored_best'

    if np.isfinite(conf) and (conf < float(PACK_CLASSIFY_LOCK_CONF)) and np.isfinite(q_prev) and (float(q_prev) > 0):
        chosen = min(candidates, key=lambda c: abs(float(q_prev) - float(c['nom_ah'])))
        options_source = 'continuity_low_conf'

    # Strong 208s split from measured Ah evidence (prevents sticky 1p lock on true 2p packs).
    chosen_series_pre = int(chosen['series'])
    if chosen_series_pre == 208 and q_q_count >= 8 and np.isfinite(q_data_med):
        if q_data_med >= float(PACK_208_FORCE_2P_Q_THRESHOLD_AH):
            chosen = {'series': 208, 'parallel': 2, 'nom_ah': 300.0}
            options_source = 'force_208_2p_from_q'
        elif q_data_med <= float(PACK_208_FORCE_1P_Q_THRESHOLD_AH):
            chosen = {'series': 208, 'parallel': 1, 'nom_ah': 150}
            options_source = 'force_208_1p_from_q'

    chosen_series = int(chosen['series'])
    chosen_nom = float(chosen['nom_ah'])
    series_opts = tuple(float(v) for v in PACK_CAPACITY_OPTIONS_BY_SERIES.get(chosen_series, default_opts))

    if options_source == 'continuity_low_conf':
        cap_opts = (chosen_nom,)
    elif np.isfinite(conf) and (conf >= float(PACK_CLASSIFY_LOCK_CONF)):
        cap_opts = (chosen_nom,)
    else:
        cap_opts = series_opts if len(series_opts) > 0 else default_opts

    info['series_cells'] = chosen_series
    info['parallel_count'] = int(chosen.get('parallel', 1))
    info['voltage_ref_v'] = v_ref
    info['series_est_cellratio'] = s_cell_est
    info['series_est_nominal'] = s_nom_est
    info['capacity_options'] = cap_opts
    info['series_source'] = v_source
    info['config_family'] = f"{chosen_series}s"
    info['config_guess'] = f"{chosen_series}s{int(chosen.get('parallel', 1))}p"
    info['chosen_nominal_ah'] = chosen_nom
    info['score_confidence'] = conf
    info['options_source'] = options_source
    info['q_data_median_ah'] = q_data_med
    info['q_data_count'] = int(q_q_count)
    return info


def _adaptive_min_drop(values, configured_drop):
    """
    Avoid injecting extra SOH drop from post-processing.
    Uses observed head-to-tail decline as an upper bound for enforced min drop.
    """
    y = _finite_series(pd.Series(values)).values.astype(float)
    y = y[np.isfinite(y)]
    if len(y) < 4:
        return 0.0
    k = max(2, int(np.ceil(0.2 * len(y))))
    head = np.nanmedian(y[:k])
    tail = np.nanmedian(y[-k:])
    if (not np.isfinite(head)) or (not np.isfinite(tail)):
        return 0.0
    observed_drop = max(0.0, float(head - tail))
    conf_drop = max(0.0, float(configured_drop))
    return min(conf_drop, observed_drop)


def _estimate_initial_capacity_ah(g: pd.DataFrame, first_cycles: int = INIT_CAPACITY_CYCLES) -> dict:
    """
    Robust initial capacity estimator from early life sessions.
    Uses first N sessions with quality filters + outlier removal, then caps against
    an early stabilization window to avoid inflated baseline-induced SOH cliffs.
    """
    out = {
        'q_base_ah': np.nan,
        'q_init_first_ah': np.nan,
        'q_ref_robust_ah': np.nan,
        'q_settle_ah': np.nan,
        'source': 'unknown',
    }

    if g is None or len(g) == 0 or ('implied_Q_Ah' not in g.columns):
        out['source'] = 'no_data'
        return out

    q_all = _finite_series(g['implied_Q_Ah'])
    q_all = q_all.where(q_all > 0)
    q_ref_robust = q_all.quantile(0.70) if q_all.notna().any() else np.nan
    q_med = q_all.median() if q_all.notna().any() else np.nan
    if np.isfinite(q_ref_robust) and np.isfinite(q_med):
        q_ref_robust = max(q_ref_robust, q_med * 0.8)
    out['q_ref_robust_ah'] = q_ref_robust

    n0 = max(int(first_cycles), 1)
    early = g.head(n0).copy()
    if len(early) == 0:
        out['q_base_ah'] = q_ref_robust
        out['source'] = 'fallback_no_early'
        return out

    mask = pd.Series(True, index=early.index, dtype=bool)
    if 'delta_soc_pct' in early.columns:
        mask &= _finite_series(early['delta_soc_pct']) >= 5.0
    if 'duration_min' in early.columns:
        mask &= _finite_series(early['duration_min']) >= 10.0
    if 'ah_total' in early.columns:
        mask &= _finite_series(early['ah_total']) >= max(float(MIN_AH), 2.0)

    q_early = _finite_series(early.loc[mask, 'implied_Q_Ah'])
    if q_early.notna().sum() < 4:
        # Relax only slightly; avoid anchoring on tiny-SOC/noisy sessions.
        relaxed = pd.Series(True, index=early.index, dtype=bool)
        if 'delta_soc_pct' in early.columns:
            relaxed &= _finite_series(early['delta_soc_pct']) >= 5.0
        if 'duration_min' in early.columns:
            relaxed &= _finite_series(early['duration_min']) >= 6.0
        if 'ah_total' in early.columns:
            relaxed &= _finite_series(early['ah_total']) >= max(float(MIN_AH), 1.5)
        q_early = _finite_series(early.loc[relaxed, 'implied_Q_Ah'])

    q_vals = q_early.dropna().values.astype(float)
    q_vals = q_vals[np.isfinite(q_vals) & (q_vals > 0)]
    if len(q_vals) >= 4:
        q1, q3 = np.percentile(q_vals, [25, 75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            q_trim = q_vals[(q_vals >= lo) & (q_vals <= hi)]
            if len(q_trim) >= 3:
                q_vals = q_trim

    q_init = float(np.nanmedian(q_vals)) if len(q_vals) > 0 else np.nan
    out['q_init_first_ah'] = q_init

    settle_start = min(n0, max(len(g) - 1, 0))
    settle_end = min(len(g), max(n0 + 1, 40))
    if settle_end > settle_start:
        q_settle = _finite_series(g.iloc[settle_start:settle_end]['implied_Q_Ah']).median()
    else:
        q_settle = np.nan
    out['q_settle_ah'] = q_settle

    q_base = q_init if np.isfinite(q_init) and q_init > 0 else np.nan
    source = 'first_cycles'

    # Cap early baseline inflation against stabilization window (+2% headroom).
    if np.isfinite(q_base) and np.isfinite(q_settle) and q_settle > 0:
        cap = 1.02 * float(q_settle)
        if q_base > cap:
            q_base = cap
            source = 'first_cycles_capped_to_settle'

    if (not np.isfinite(q_base)) or (q_base <= 0):
        q_base = q_ref_robust
        source = 'fallback_robust'

    out['q_base_ah'] = q_base
    out['source'] = source
    return out


def _compute_bms_correction_factor(g: pd.DataFrame) -> float:
    """Return median(bms_ah / implied_Q_Ah) over high-quality sessions, or 1.0 if unavailable."""
    if 'bms_ah' not in g.columns:
        return 1.0
    valid = g[
        g['bms_ah'].notna() & (g['bms_ah'] > 0) &
        g['implied_Q_Ah'].notna() & (g['implied_Q_Ah'] > 0) &
        (g['delta_soc_pct'] >= MIN_DELTA_SOC * 2)
    ]
    if len(valid) < 3:
        return 1.0
    ratios = valid['bms_ah'] / valid['implied_Q_Ah']
    lo, hi = ratios.quantile(0.10), ratios.quantile(0.90)
    trimmed = ratios[(ratios >= lo) & (ratios <= hi)]
    return float(trimmed.median()) if len(trimmed) > 0 else 1.0


def _fmt_days(v, cap_days=REPORT_RUL_CAP_DAYS):
    if np.isinf(v):
        return f"> {cap_days:.0f} days (no EOL trend)"
    if np.isfinite(v):
        if v > cap_days:
            return f"> {cap_days:.0f} days"
        return f"{v:.0f} days"
    return "NA"


def _fmt_kwh(v):
    if np.isinf(v):
        return "inf kWh (no finite EOL crossing)"
    if np.isfinite(v):
        return f"{v:.1f} kWh"
    return "NA"


def _fmt_ah(v):
    if np.isinf(v):
        return "inf Ah"
    if np.isfinite(v):
        return f"{v:.1f} Ah"
    return "NA"


def _fmt_date(ts):
    if ts is None or (isinstance(ts, float) and not np.isfinite(ts)):
        return "NA"
    try:
        if pd.isna(ts):
            return "NA"
        t = pd.to_datetime(ts, errors='coerce')
        if pd.isna(t):
            return "NA"
        return t.strftime('%Y-%m-%d')
    except Exception:
        return "NA"


def _fmt_km(v, approx=False):
    if np.isinf(v):
        return "inf km"
    if np.isfinite(v):
        return f"{'~' if approx else ''}{int(round(v)):,} km"
    return "NA"


def _imei_text(vehicle_id):
    s = str(vehicle_id)
    return s.replace("IMEI_", "") if s.startswith("IMEI_") else s


def _life_remaining_text(p50_days, p10_days, p90_days=np.nan):
    p50 = _fmt_days(p50_days)
    p10 = _fmt_days(p10_days)
    p90 = _fmt_days(p90_days) if np.isfinite(p90_days) or np.isinf(p90_days) else "NA"
    # If both finite regular day counts, prefer compact customer style.
    if (
        np.isfinite(p50_days) and np.isfinite(p10_days) and np.isfinite(p90_days) and
        p50_days <= REPORT_RUL_CAP_DAYS and p10_days <= REPORT_RUL_CAP_DAYS and p90_days <= REPORT_RUL_CAP_DAYS
    ):
        return (
            f"{int(round(p50_days))} days "
            f"(worst: {int(round(p10_days))} | best: {int(round(p90_days))})"
        )
    if np.isfinite(p90_days) or np.isinf(p90_days):
        return f"{p50}  (worst: {p10} | best: {p90})"
    return f"{p50}  (worst case: {p10})"


def _safe_project_date(base_ts, days, horizon_days=REPORT_RUL_CAP_DAYS):
    """
    Safely project a date by N days.
    Returns NaT when projection is non-finite, negative-invalid, or beyond horizon.
    Prevents pandas OutOfBoundsDatetime on huge tails.
    """
    try:
        if pd.isna(base_ts) or (not np.isfinite(days)):
            return pd.NaT
        d = float(days)
        if d < 0:
            d = 0.0
        if d > float(horizon_days):
            return pd.NaT
        return pd.to_datetime(base_ts, errors='coerce') + pd.to_timedelta(d, unit='D')
    except Exception:
        return pd.NaT


def _fmt_eol_date(ts, rul_days, horizon_days=REPORT_RUL_CAP_DAYS):
    if np.isfinite(rul_days) and (rul_days > horizon_days):
        return f"Beyond {int(horizon_days/365)}y horizon"
    return _fmt_date(ts)


def _utc_to_ist_datetime(utc_num):
    """Convert GPS-based numeric UTC column used in telemetry to IST datetime."""
    if not np.isfinite(utc_num):
        return pd.NaT
    # gps_seconds + unix_offset + IST offset
    return pd.to_datetime(float(utc_num) + 946684800 + 19800, unit='s', origin='unix', errors='coerce')


def detect_battery_replacements(xgb_results: dict) -> pd.DataFrame:
    """
    Detect likely battery replacement events from positive capacity/SOH jumps
    with persistence checks. Does not use hrlfc resets as a primary signal.
    """
    events = []
    w = int(REPL_WINDOW)
    m = int(REPL_PERSIST_M)
    k = int(REPL_PERSIST_K)

    for vid, res in xgb_results.items():
        g = res['sessions'].copy()
        if len(g) < (2 * w + 3):
            continue

        if 'start_utc' in g.columns and _finite_series(g['start_utc']).notna().sum() >= max(5, int(0.6 * len(g))):
            g = g.sort_values('start_utc').reset_index(drop=True)
        else:
            sort_col = _pick_axis_col(g)
            if sort_col == '__session_idx':
                g = g.copy()
                g['__session_idx'] = np.arange(len(g), dtype=float)
            g = g.sort_values(sort_col).reset_index(drop=True)

        q = _finite_series(g['implied_Q_Ah']) if 'implied_Q_Ah' in g.columns else pd.Series(np.nan, index=g.index)
        soh = _finite_series(g['soh_label']) if 'soh_label' in g.columns else _finite_series(g.get('soh_xgb', pd.Series(np.nan, index=g.index)))

        i = w
        while i < len(g) - w - 1:
            pre_q = float(np.nanmedian(q.iloc[i - w:i])) if np.isfinite(np.nanmedian(q.iloc[i - w:i])) else np.nan
            post_q = float(np.nanmedian(q.iloc[i + 1:i + 1 + w])) if np.isfinite(np.nanmedian(q.iloc[i + 1:i + 1 + w])) else np.nan
            pre_soh = float(np.nanmedian(soh.iloc[i - w:i])) if np.isfinite(np.nanmedian(soh.iloc[i - w:i])) else np.nan
            post_soh = float(np.nanmedian(soh.iloc[i + 1:i + 1 + w])) if np.isfinite(np.nanmedian(soh.iloc[i + 1:i + 1 + w])) else np.nan

            q_jump_ah = (post_q - pre_q) if np.isfinite(pre_q) and np.isfinite(post_q) else np.nan
            q_jump_pct = (100.0 * q_jump_ah / pre_q) if np.isfinite(q_jump_ah) and np.isfinite(pre_q) and pre_q > 0 else np.nan
            soh_jump = (post_soh - pre_soh) if np.isfinite(pre_soh) and np.isfinite(post_soh) else np.nan

            cond_q = np.isfinite(q_jump_pct) and np.isfinite(q_jump_ah) and (q_jump_pct >= REPL_Q_JUMP_PCT) and (q_jump_ah >= REPL_Q_JUMP_AH)
            cond_soh = np.isfinite(soh_jump) and (soh_jump >= REPL_SOH_JUMP_PCT)
            if not (cond_q or cond_soh):
                i += 1
                continue

            end = min(len(g), i + 1 + m)
            next_q = q.iloc[i + 1:end]
            next_soh = soh.iloc[i + 1:end]

            q_thresh = pre_q * (1.0 + 0.6 * REPL_Q_JUMP_PCT / 100.0) if np.isfinite(pre_q) else np.nan
            soh_thresh = pre_soh + 0.6 * REPL_SOH_JUMP_PCT if np.isfinite(pre_soh) else np.nan
            keep_q = np.isfinite(q_thresh) and int((next_q >= q_thresh).sum()) >= k
            keep_soh = np.isfinite(soh_thresh) and int((next_soh >= soh_thresh).sum()) >= k

            if not (keep_q or keep_soh):
                i += 1
                continue

            event_row = g.iloc[i + 1]
            utc_ev = float(event_row['start_utc']) if ('start_utc' in g.columns and np.isfinite(pd.to_numeric(event_row['start_utc'], errors='coerce'))) else np.nan
            dt_ev = _utc_to_ist_datetime(utc_ev)

            q_score = 0.0
            if np.isfinite(q_jump_pct):
                q_score += min(1.0, q_jump_pct / max(REPL_Q_JUMP_PCT, 1e-9))
            if np.isfinite(q_jump_ah):
                q_score += min(1.0, q_jump_ah / max(REPL_Q_JUMP_AH, 1e-9))
            q_score = min(1.0, q_score / 2.0)
            soh_score = min(1.0, max(0.0, soh_jump / max(REPL_SOH_JUMP_PCT, 1e-9))) if np.isfinite(soh_jump) else 0.0
            pers_score = min(1.0, max(int((next_q >= q_thresh).sum()) if np.isfinite(q_thresh) else 0,
                                      int((next_soh >= soh_thresh).sum()) if np.isfinite(soh_thresh) else 0) / max(k, 1))
            conf = 0.45 * q_score + 0.35 * soh_score + 0.20 * pers_score

            events.append({
                'vehicle_id': vid,
                'event_session_idx': int(i + 1),
                'event_utc': utc_ev,
                'event_datetime_ist': dt_ev,
                'pre_q_ah': pre_q,
                'post_q_ah': post_q,
                'q_jump_ah': q_jump_ah,
                'q_jump_pct': q_jump_pct,
                'pre_soh_pct': pre_soh,
                'post_soh_pct': post_soh,
                'soh_jump_pct': soh_jump,
                'confidence': float(conf),
            })

            i += w

    if not events:
        return pd.DataFrame(columns=[
            'vehicle_id', 'event_session_idx', 'event_utc', 'event_datetime_ist',
            'pre_q_ah', 'post_q_ah', 'q_jump_ah', 'q_jump_pct',
            'pre_soh_pct', 'post_soh_pct', 'soh_jump_pct', 'confidence'
        ])

    ev = pd.DataFrame(events).sort_values(['vehicle_id', 'event_session_idx']).reset_index(drop=True)
    return ev


def _print_summary_tables(rul_all: dict, replacement_events: pd.DataFrame):
    """Print readable in-console tables for fleet RUL and replacement events."""
    def _render_table(df: pd.DataFrame):
        if df is None or len(df) == 0:
            print("(no rows)")
            return
        try:
            from tabulate import tabulate
            print(tabulate(df, headers='keys', tablefmt='rounded_grid', showindex=False, numalign='right', stralign='left'))
        except Exception:
            print(df.to_string(index=False, justify='left'))

    def _mgr_status(soh):
        if not np.isfinite(soh):
            return "Unknown"
        if soh > 85:
            return "Green"
        if soh >= 75:
            return "Amber"
        return "Red"

    def _mgr_action(soh, months):
        if (not np.isfinite(soh)) and (not np.isfinite(months)):
            return "Collect more data"
        if (np.isfinite(soh) and soh < 75) or (np.isfinite(months) and months <= 6):
            return "Replace soon"
        if (np.isfinite(soh) and soh < 85) or (np.isfinite(months) and months <= 12):
            return "Monitor monthly"
        return "Healthy"

    rows = []
    for vid, r in rul_all.items():
        rows.append({
            'Vehicle': vid,
            'SOH_now_%': r.get('soh_now', np.nan),
            'InitCap_100%_Ah': r.get('init_capacity_ah', np.nan),
            'InitCap_100%_kWh': r.get('init_capacity_kwh', np.nan),
            'CurrentCap_Ah': r.get('current_capacity_ah', np.nan),
            'CurrentCap_kWh': r.get('current_capacity_kwh', np.nan),
            'EOLCap_80%_Ah': r.get('eol_capacity_ah', np.nan),
            'Charging_events': r.get('charging_events_count', np.nan),
            'Eq_full_cycles': r.get('equivalent_full_cycles', np.nan),
            'KM_run_to_date': r.get('km_run_till_date', np.nan),
            'KM_per_day_hist': r.get('km_per_day_hist', np.nan),
            'RUL_P10_days': r.get('rul_days_p10', np.nan),
            'RUL_P50_days': r.get('rul_days_p50', np.nan),
            'RUL_P90_days': r.get('rul_days_p90', np.nan),
            'KM_to_EOL_P10': r.get('km_to_eol_p10', np.nan),
            'KM_to_EOL_P50': r.get('km_to_eol_p50', np.nan),
            'KM_to_EOL_P90': r.get('km_to_eol_p90', np.nan),
            'EOL_date_P10': r.get('eol_date_p10_ist', pd.NaT),
            'EOL_date_P50': r.get('eol_date_p50_ist', pd.NaT),
            'EOL_date_P90': r.get('eol_date_p90_ist', pd.NaT),
            'E_to_EOL_P10_kWh': r.get('energy_to_eol_kwh_p10', np.nan),
            'E_to_EOL_P50_kWh': r.get('energy_to_eol_kwh_p50', np.nan),
            'E_to_EOL_P90_kWh': r.get('energy_to_eol_kwh_p90', np.nan),
            'Slope_%/10k': (r.get('phase2_slope', np.nan) * RUL_SLOPE_DISPLAY_AXIS_SCALE) if np.isfinite(r.get('phase2_slope', np.nan)) else np.nan,
            'SlopeBasis': r.get('slope_basis', ''),
        })

    if rows:
        df = pd.DataFrame(rows).sort_values('Vehicle').reset_index(drop=True)
        df['Vehicle'] = df['Vehicle'].map(_normalize_vehicle_id_text)

        # Manager-facing compact table first (easy to read in terminal).
        m = df.copy()
        m['SOH_now_%'] = pd.to_numeric(m['SOH_now_%'], errors='coerce')
        m['RUL_P50_days_num'] = pd.to_numeric(m['RUL_P50_days'], errors='coerce')
        m['Months_to_EOL'] = m['RUL_P50_days_num'] / 30.44
        m['Status'] = m['SOH_now_%'].apply(_mgr_status)
        m['Action'] = [
            _mgr_action(s, mo) for s, mo in zip(m['SOH_now_%'].tolist(), m['Months_to_EOL'].tolist())
        ]
        m['SOH_%'] = m['SOH_now_%'].apply(lambda v: f"{v:.1f}%" if np.isfinite(v) else "NA")
        m['Months_to_EOL'] = m['Months_to_EOL'].apply(lambda v: f"{v:.1f}" if np.isfinite(v) else "NA")
        m['RUL_Worst'] = pd.to_numeric(m['RUL_P10_days'], errors='coerce').apply(lambda v: _fmt_days(v))
        m['RUL_Likely'] = m['RUL_P50_days_num'].apply(lambda v: _fmt_days(v))
        m['RUL_Best'] = pd.to_numeric(m['RUL_P90_days'], errors='coerce').apply(lambda v: _fmt_days(v))
        m['Distance_Travelled'] = m['KM_run_to_date'].apply(
            lambda v: f"{v:,.0f} km" if np.isfinite(v) else "NA"
        )
        m['Dist_Rem_Worst'] = m['KM_to_EOL_P10'].apply(
            lambda v: ("inf km" if np.isinf(v) else (f"~{v:,.0f} km" if np.isfinite(v) else "NA"))
        )
        m['Dist_Rem_Likely'] = m['KM_to_EOL_P50'].apply(
            lambda v: ("inf km" if np.isinf(v) else (f"~{v:,.0f} km" if np.isfinite(v) else "NA"))
        )
        m['Dist_Rem_Best'] = m['KM_to_EOL_P90'].apply(
            lambda v: ("inf km" if np.isinf(v) else (f"~{v:,.0f} km" if np.isfinite(v) else "NA"))
        )
        m['Daily_KM_Run'] = pd.to_numeric(m['KM_per_day_hist'], errors='coerce').apply(
            lambda v: f"{v:.1f} km/day" if np.isfinite(v) else "NA"
        )
        m['EOL_date_P50'] = [
            _fmt_eol_date(ts, d)
            for ts, d in zip(df['EOL_date_P50'].tolist(), m['RUL_P50_days_num'].tolist())
        ]
        m['EOL_Worst'] = [
            _fmt_eol_date(ts, d)
            for ts, d in zip(df['EOL_date_P10'].tolist(), pd.to_numeric(m['RUL_P10_days'], errors='coerce').tolist())
        ]
        m['EOL_Best'] = [
            _fmt_eol_date(ts, d)
            for ts, d in zip(df['EOL_date_P90'].tolist(), pd.to_numeric(m['RUL_P90_days'], errors='coerce').tolist())
        ]
        m_show = m[
            [
                'Vehicle', 'SOH_%', 'Status', 'Months_to_EOL',
                'Distance_Travelled', 'Dist_Rem_Worst', 'Dist_Rem_Likely', 'Dist_Rem_Best', 'Daily_KM_Run',
                'RUL_Worst', 'RUL_Likely', 'RUL_Best',
                'EOL_Worst', 'EOL_date_P50', 'EOL_Best',
                'Action'
            ]
        ]
        m_show = m_show.sort_values(['Status', 'Vehicle'], key=lambda s: s.map({'Red': 0, 'Amber': 1, 'Green': 2, 'Unknown': 3}) if s.name == 'Status' else s).reset_index(drop=True)
        print("\n[MANAGER TABLE] Vehicle SOH / Status / Timeline")
        _render_table(m_show)

        df_show = df.copy()
        for c in [
            'SOH_now_%', 'InitCap_100%_Ah', 'InitCap_100%_kWh', 'CurrentCap_Ah', 'CurrentCap_kWh', 'EOLCap_80%_Ah',
            'Charging_events', 'Eq_full_cycles',
            'KM_run_to_date',
            'RUL_P10_days', 'RUL_P50_days', 'RUL_P90_days',
            'KM_to_EOL_P10', 'KM_to_EOL_P50', 'KM_to_EOL_P90',
            'E_to_EOL_P10_kWh', 'E_to_EOL_P50_kWh', 'E_to_EOL_P90_kWh',
            'Slope_%/10k'
        ]:
            df_show[c] = pd.to_numeric(df_show[c], errors='coerce').round(2)
        df_show['RUL_P10_days'] = df['RUL_P10_days'].apply(lambda v: _fmt_days(v))
        df_show['RUL_P50_days'] = df['RUL_P50_days'].apply(lambda v: _fmt_days(v))
        df_show['RUL_P90_days'] = df['RUL_P90_days'].apply(lambda v: _fmt_days(v))
        df_show['E_to_EOL_P10_kWh'] = df['E_to_EOL_P10_kWh'].apply(lambda v: _fmt_kwh(v))
        df_show['E_to_EOL_P50_kWh'] = df['E_to_EOL_P50_kWh'].apply(lambda v: _fmt_kwh(v))
        df_show['E_to_EOL_P90_kWh'] = df['E_to_EOL_P90_kWh'].apply(lambda v: _fmt_kwh(v))
        df_show['KM_to_EOL_P10'] = df['KM_to_EOL_P10'].apply(lambda v: "inf km" if np.isinf(v) else (f"{v:.0f} km" if np.isfinite(v) else "NA"))
        df_show['KM_to_EOL_P50'] = df['KM_to_EOL_P50'].apply(lambda v: "inf km" if np.isinf(v) else (f"{v:.0f} km" if np.isfinite(v) else "NA"))
        df_show['KM_to_EOL_P90'] = df['KM_to_EOL_P90'].apply(lambda v: "inf km" if np.isinf(v) else (f"{v:.0f} km" if np.isfinite(v) else "NA"))
        df_show['EOL_date_P50'] = [
            _fmt_eol_date(ts, d)
            for ts, d in zip(df['EOL_date_P50'].tolist(), pd.to_numeric(df['RUL_P50_days'], errors='coerce').tolist())
        ]

        # Cleaner technical summary for terminal readability.
        tech_show = pd.DataFrame({
            'Vehicle': df_show['Vehicle'].astype(str),
            'SOH_%': df_show['SOH_now_%'].apply(lambda v: f"{v:.1f}%" if np.isfinite(v) else "NA"),
            'Init_Ah': df_show['InitCap_100%_Ah'].apply(lambda v: f"{v:.1f}" if np.isfinite(v) else "NA"),
            'Curr_Ah': df_show['CurrentCap_Ah'].apply(lambda v: f"{v:.1f}" if np.isfinite(v) else "NA"),
            'Init_kWh': df_show['InitCap_100%_kWh'].apply(lambda v: f"{v:.1f}" if np.isfinite(v) else "NA"),
            'Curr_kWh': df_show['CurrentCap_kWh'].apply(lambda v: f"{v:.1f}" if np.isfinite(v) else "NA"),
            'Events': df_show['Charging_events'].apply(lambda v: f"{int(v)}" if np.isfinite(v) else "NA"),
            'RUL_P50': df_show['RUL_P50_days'].astype(str),
            'EOL_P50': df_show['EOL_date_P50'].astype(str),
            'Slope': df_show['Slope_%/10k'].apply(lambda v: f"{v:.2f}" if np.isfinite(v) else "NA"),
            'Basis': df_show['SlopeBasis'].astype(str),
        })
        print("\n[SUMMARY TABLE] Fleet Capacity / SOH / RUL")
        _render_table(tech_show)

    if replacement_events is None or replacement_events.empty:
        print("\n[REPLACEMENT TABLE] No likely battery replacement events detected.")
        return

    ev = replacement_events.copy()
    ev_show = pd.DataFrame({
        'Vehicle': ev['vehicle_id'].map(_normalize_vehicle_id_text),
        'SessionIdx': ev['event_session_idx'],
        'DateTime_IST': ev['event_datetime_ist'].astype(str),
        'Q_pre_Ah': pd.to_numeric(ev['pre_q_ah'], errors='coerce').round(1),
        'Q_post_Ah': pd.to_numeric(ev['post_q_ah'], errors='coerce').round(1),
        'Q_jump_%': pd.to_numeric(ev['q_jump_pct'], errors='coerce').round(1),
        'SOH_pre_%': pd.to_numeric(ev['pre_soh_pct'], errors='coerce').round(1),
        'SOH_post_%': pd.to_numeric(ev['post_soh_pct'], errors='coerce').round(1),
        'SOH_jump_%': pd.to_numeric(ev['soh_jump_pct'], errors='coerce').round(1),
        'Confidence': pd.to_numeric(ev['confidence'], errors='coerce').round(2),
    })
    print("\n[REPLACEMENT TABLE] Likely Battery Replacement Events")
    _render_table(ev_show)


def _clean_id_series(series: pd.Series) -> pd.Series:
    """Normalize candidate ID values and map empties to NaN."""
    s = series.astype(str).str.strip()
    # CSV numeric parsing can turn integer-like IDs into strings like "352914000000000.0".
    s = s.str.replace(r'^(\d+)\.0+$', r'\1', regex=True)
    return s.replace({
        '': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan, 'none': np.nan, 'null': np.nan
    })


def _norm_col(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


CANONICAL_COLUMN_ALIASES = {
    # raw SOC variants
    'fuelLevel': (
        'fuellevel', 'soc', 'socpct', 'socpercentage', 'stateofcharge',
        'stateofchargepct', 'batterysoc', 'batterypercentage',
    ),
    # row-level charging current variants
    'chargingCurrent': (
        'chargingcurrent', 'chargecurrent', 'chargingcurrenta', 'chargecurrenta',
        'chargingamp', 'chargingamps', 'currentcharge',
    ),
    # row-level voltage variants
    'battPackVoltage': (
        'battpackvoltage', 'batteryvoltage', 'packvoltage', 'packv', 'batteryv',
    ),
    # utc/time variants often found in exports
    '_utc_num': (
        'utcnum', 'utc', 'gpstime', 'gpsutc', 'utctime', 'epochtime',
    ),
    # derived/session helper variants
    'charge_calc': (
        'chargecalc', 'chargeah', 'chargecalcah',
    ),
    'dt_sec': (
        'dtsec', 'deltatsec', 'timedeltasec',
    ),
    # BMS-reported current capacity (Ah)
    'bms_ah': (
        'bmsah', 'bms_ah', 'bmscapacity', 'bms_capacity_ah', 'packah',
        'bmscurrentcap', 'bms_current_capacity',
    ),
    # BMS-reported state of health (%)
    'bms_soh': (
        'bmssoh', 'bms_soh', 'packsoh', 'stateofhealth', 'batteryhealth',
        'bmshealth', 'bms_health_pct',
    ),
    # BMS-reported initial / rated capacity (Ah)
    'bms_init_cap': (
        'bmsinitcap', 'bms_initial_capacity', 'bmsratedcap', 'bms_rated_cap',
        'bmsratecap', 'bmsnominalcap', 'bms_nominal_capacity',
    ),
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common telemetry aliases to the canonical column names used by this pipeline.
    Keeps existing canonical names untouched.
    """
    if df is None or len(df.columns) == 0:
        return df

    norm_to_cols = {}
    for c in df.columns:
        norm_to_cols.setdefault(_norm_col(c), []).append(c)

    rename_map = {}
    for canonical, alias_norms in CANONICAL_COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias_norm in alias_norms:
            matches = norm_to_cols.get(alias_norm, [])
            if not matches:
                continue
            source_col = matches[0]
            if source_col in rename_map:
                continue
            rename_map[source_col] = canonical
            break

    if rename_map:
        df = df.rename(columns=rename_map)
        mapped = ", ".join(f"{src}->{dst}" for src, dst in sorted(rename_map.items()))
        print(f"    Canonicalized columns: {mapped}")

    return df


def _normalize_bucket_series(series: pd.Series) -> pd.Series:
    """
    Normalize bucket labels to canonical values used by this pipeline:
    CHARGING / DRIVING / CRANKING / OTHERS.
    """
    s = series.astype(str).str.strip()
    n = s.map(_norm_col)

    out = pd.Series('OTHERS', index=series.index, dtype=object)
    out.loc[n.isin({'charging', 'charge', 'chg'})] = 'CHARGING'
    out.loc[n.isin({'driving', 'drive', 'drv'})] = 'DRIVING'
    out.loc[n.isin({'cranking', 'crank'})] = 'CRANKING'
    return out


def _select_vehicle_id_column(df: pd.DataFrame):
    """
    Pick the best row-level ID column.
    Prefer stable vehicle identity columns (VIN/vehicle_id).
    Fall back to replaceable telematics IDs (IMEI/deviceId), then filename-derived fallback.
    """
    cols = list(df.columns)
    norm_map = {c: _norm_col(c) for c in cols}

    imei_like = []
    vin_like = []
    vehicle_like = []
    device_like = []
    for c, n in norm_map.items():
        if n in {'sessionid', 'tripid', 'bucketid'}:
            continue
        if 'imei' in n:
            imei_like.append(c)
        elif n == 'vin' or n.endswith('vin') or n.startswith('vin'):
            vin_like.append(c)
        elif 'vehicle' in n and 'id' in n:
            vehicle_like.append(c)
        elif n in {'deviceid', 'deviceimei', 'device_id', 'deviceimeiid'}:
            device_like.append(c)

    exact_priority = [
        'VIN', 'vin',
        'vehicle_id', 'vehicleId',
        'IMEI', 'imei', 'device_imei', 'deviceId', 'device_id',
    ]
    for c in exact_priority:
        if c in df.columns and c not in imei_like + vin_like + vehicle_like + device_like:
            n = _norm_col(c)
            if 'imei' in n:
                imei_like.append(c)
            elif 'vin' in n:
                vin_like.append(c)
            elif n in {'deviceid', 'deviceimei', 'deviceid', 'deviceimeiid'}:
                device_like.append(c)
            else:
                vehicle_like.append(c)

    families = [('vin', vin_like), ('vehicle', vehicle_like), ('imei', imei_like), ('device', device_like)]
    stats = []
    for fam, cand_cols in families:
        for c in cand_cols:
            s = _clean_id_series(df[c])
            nn = int(s.notna().sum())
            nu = int(s.nunique(dropna=True))
            stats.append((fam, c, nn, nu))

    # Prefer stable identity families that clearly separate multiple vehicles.
    for fam in ['vin', 'vehicle', 'imei', 'device']:
        fam_stats = [t for t in stats if t[0] == fam and t[3] > 1]
        if fam_stats:
            # more unique IDs first, then more non-null coverage
            fam_stats.sort(key=lambda t: (t[3], t[2]), reverse=True)
            return fam_stats[0][1], stats

    # Fallback: best available non-empty candidate, with stable families first.
    family_rank = {'vin': 0, 'vehicle': 1, 'imei': 2, 'device': 3}
    non_empty = [t for t in stats if t[3] >= 1]
    if non_empty:
        non_empty.sort(key=lambda t: (family_rank.get(t[0], 99), -t[3], -t[2], t[1]))
        return non_empty[0][1], stats

    return None, stats


def _resolve_vehicle_id(df: pd.DataFrame, fallback_col: str = '__file_vehicle_id') -> pd.DataFrame:
    """
    Build row-level vehicle_id from existing columns (IMEI/vehicle fields first),
    then fallback to filename-derived ID only where needed.
    """
    chosen_col, stats = _select_vehicle_id_column(df)
    if stats:
        print("    Vehicle-ID candidates:")
        for fam, c, nn, nu in sorted(stats, key=lambda t: (t[0], -t[3], -t[2], t[1])):
            print(f"      [{fam}] {c}: non-null={nn:,}, unique={nu:,}")
    if chosen_col:
        print(f"    Using vehicle ID column: {chosen_col}")
        resolved = _clean_id_series(df[chosen_col])
    else:
        print("    [INFO] No row-level vehicle ID column found; using filename fallback.")
        resolved = pd.Series(np.nan, index=df.index, dtype=object)

    if fallback_col in df.columns:
        resolved = resolved.combine_first(_clean_id_series(df[fallback_col]))

    resolved = resolved.fillna('unknown_vehicle')
    df['vehicle_id'] = resolved.astype(str)
    return df


def _read_csv_resilient(path: Path) -> pd.DataFrame:
    """
    Read very large CSVs with memory-aware fallbacks.
    1) normal read with selected columns
    2) chunked C-engine read
    3) chunked Python-engine read
    """
    # Keep only columns used by this pipeline to reduce RAM.
    keep_cols = {
        # ID candidates
        'vehicle_id', 'vehicleId', 'IMEI', 'imei', 'device_imei', 'deviceId', 'device_id', 'vin', 'VIN',
        # Core telemetry/features
        'utc', '_utc_num', 'DateTime',
        'fuelLevel', 'chargingCurrent', 'battPackVoltage',
        'maxCellVoltage', 'minCellVoltage', 'maxCellTemp', 'minCellTemp',
        'battPowerIn', 'regenerationPower', 'hrlfc', 'totalDistance',
        'charge_calc', 'dt_sec', 'vehicleSpeed', 'chargingStatus', 'crankStatus',
        'motorCurrent', 'chg_power_calc',
        # Session / engineered fields from pre-cleaned files
        'bucket', 'session_id',
        'delta_soc_pct', 'ah_total', 'volt_spread_mean', 'volt_spread_eoc',
        'temp_max', 'dT_per_crate', 'avg_c_rate', 'pack_v_norm',
        'ah_per_min', 'charge_efficiency', 'hrlfc_mid', 'duration_min',
        'temp_spread_mean', 'implied_Q_Ah', 'soh_label', 'soh_smooth',
        'start_utc', 'end_utc', 'packets',
        # Optional datetime columns from notebook variants
        'timestamp', 'datetime', 'date', 'time',
    }

    # Resolve available columns from header.
    header_cols = None
    try:
        header_cols = list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        header_cols = None

    keep_norm = {_norm_col(c) for c in keep_cols}
    for alias_norms in CANONICAL_COLUMN_ALIASES.values():
        keep_norm.update(alias_norms)

    use_cols = None
    if header_cols is not None and len(header_cols) > 0:
        use_cols = [c for c in header_cols if _norm_col(c) in keep_norm]
        if len(use_cols) == 0:
            use_cols = None

    # Avoid dtype=object for all columns; that explodes memory on large CSVs.
    read_kwargs = dict(low_memory=True, on_bad_lines='skip', memory_map=True)
    if use_cols is not None:
        read_kwargs['usecols'] = use_cols

    # Attempt 1: normal read.
    try:
        return pd.read_csv(path, **read_kwargs)
    except Exception as e1:
        print(f"    [WARN] Normal read failed for {path.name}: {e1}")

    # Attempt 2: chunked C-engine.
    for cs in [100_000, 50_000]:
        try:
            parts = []
            for chunk in pd.read_csv(path, chunksize=cs, **read_kwargs):
                parts.append(chunk)
            if parts:
                return pd.concat(parts, ignore_index=True, sort=False)
        except Exception as e2:
            print(f"    [WARN] Chunked C-engine read failed ({cs}) for {path.name}: {e2}")

    # Attempt 3: chunked Python-engine.
    for cs in [50_000, 20_000]:
        try:
            parts = []
            for chunk in pd.read_csv(path, chunksize=cs, engine='python', **read_kwargs):
                parts.append(chunk)
            if parts:
                return pd.concat(parts, ignore_index=True, sort=False)
        except Exception as e3:
            print(f"    [WARN] Chunked Python-engine read failed ({cs}) for {path.name}: {e3}")

    raise RuntimeError(f"Unable to read CSV: {path}")


# ------------------------------------------------------------------------------
# SESSION SEGMENTATION  (ported from Data_cleaning_improved.ipynb)
# ------------------------------------------------------------------------------
def segment_sessions(
        g: pd.DataFrame,
        max_gap_min: float = SEG_MAX_GAP_MIN,
        eps_curr_entry: float = SEG_EPS_CURR_ENTRY,
        eps_curr_exit: float  = SEG_EPS_CURR_EXIT,
        eps_speed: float      = SEG_EPS_SPEED,
) -> pd.DataFrame:
    """
    Vectorized state machine that labels every row as one of:
      CHARGING / DRIVING / CRANKING / OTHERS
    and assigns a monotone session_id within each vehicle group.

    Operates on a single vehicle's sorted DataFrame.
    Logic is identical to the notebook â€” only the loop is index-based
    (numpy array access) for ~20Ã— speed-up vs iterrows().
    """
    g = g.sort_values('_utc_num').reset_index(drop=True)
    n = len(g)

    utc      = g['_utc_num'].to_numpy(dtype=float)
    chg_st   = pd.to_numeric(g[SEG_CHG_STATUS]   if SEG_CHG_STATUS   in g.columns else pd.Series(0, index=g.index), errors='coerce').fillna(0).to_numpy()
    crank_st = pd.to_numeric(g[SEG_CRANK_STATUS] if SEG_CRANK_STATUS in g.columns else pd.Series(0, index=g.index), errors='coerce').fillna(0).to_numpy()
    chg_curr = pd.to_numeric(g[SEG_CHG_CURR]     if SEG_CHG_CURR     in g.columns else pd.Series(0, index=g.index), errors='coerce').fillna(0).to_numpy()
    veh_spd  = pd.to_numeric(g[SEG_VEH_SPEED]    if SEG_VEH_SPEED    in g.columns else pd.Series(0, index=g.index), errors='coerce').fillna(0).to_numpy()

    states   = ['OTHERS'] * n
    sess_ids = [np.nan]   * n
    state    = 'OTHERS'
    sess_id  = -1

    for i in range(n):
        gap = (utc[i] - utc[i - 1]) / 60.0 if i > 0 else 0.0

        # Gap reset
        if gap > max_gap_min and chg_st[i] != 1:
            state = 'OTHERS'

        # Entry
        if state == 'OTHERS':
            if chg_st[i] == 1 and crank_st[i] == 0 and abs(chg_curr[i]) > eps_curr_entry:
                state = 'CHARGING'; sess_id += 1
            elif chg_st[i] == 0 and crank_st[i] == 1 and veh_spd[i] > eps_speed:
                state = 'DRIVING';  sess_id += 1
            elif chg_st[i] == 0 and crank_st[i] == 1 and veh_spd[i] <= eps_speed:
                state = 'CRANKING'; sess_id += 1

        # Exit
        elif state == 'CHARGING':
            if chg_st[i] == 0 and abs(chg_curr[i]) <= eps_curr_exit:
                state = 'OTHERS'
        elif state == 'DRIVING':
            if veh_spd[i] <= eps_speed and crank_st[i] == 0:
                state = 'OTHERS'
        elif state == 'CRANKING':
            if veh_spd[i] > eps_speed:
                state = 'DRIVING'   # CRANKING -> DRIVING (no id increment)

        states[i]   = state
        sess_ids[i] = sess_id if state != 'OTHERS' else np.nan

    g['bucket']     = states
    g['session_id'] = sess_ids
    return g


# ------------------------------------------------------------------------------
# STEP 0 - LOAD AND CLEAN
# ------------------------------------------------------------------------------
def load_and_clean(path: str, since_utc: float = None, overlap_sec: float = 0.0) -> pd.DataFrame:
    """
    Accepts a folder (reads all CSVs recursively) or a single CSV file.
    Handles: vehicle_id extraction, hrlfc counter resets, type casting.
    """
    from tqdm import tqdm

    print("[1/6] Loading data...")
    p = Path(path)

    if p.is_dir():
        # Search recursively, case-insensitive extension
        csv_files = sorted(p.glob("**/*.csv")) + sorted(p.glob("**/*.CSV"))
        # Windows paths are case-insensitive; dedupe overlap from dual globs.
        dedup = {}
        for f in csv_files:
            key = str(f.resolve()).lower()
            dedup[key] = f
        csv_files = sorted(dedup.values(), key=lambda x: str(x).lower())
        if not csv_files:
            # Flat search fallback
            csv_files = sorted(p.glob("*.csv")) + sorted(p.glob("*.CSV"))
            dedup = {}
            for f in csv_files:
                key = str(f.resolve()).lower()
                dedup[key] = f
            csv_files = sorted(dedup.values(), key=lambda x: str(x).lower())
        if not csv_files:
            all_items = list(p.iterdir())
            print(f"    [DEBUG] Nothing found. Folder contains {len(all_items)} items:")
            for f in all_items[:20]:
                print(f"      {f.name}")
            raise FileNotFoundError(f"No CSV files found in: {path}")

        print(f"    Found {len(csv_files)} CSV files:")
        for f in csv_files[:5]:
            print(f"      {f}")
        if len(csv_files) > 5:
            print(f"      ... and {len(csv_files)-5} more")

        all_dfs = []
        for f in tqdm(csv_files, desc='    Reading files'):
            try:
                tmp = _read_csv_resilient(f)
                tmp['__source_file'] = f.name
                tmp['__source_path'] = str(f.resolve())
                # Filename fallback only; real row-level IDs are resolved later.
                match = re.search(r'IMEI_\d+', f.name)
                tmp['__file_vehicle_id'] = match.group(0) if match else f.stem
                all_dfs.append(tmp)
            except Exception as e:
                print(f"    [WARN] Could not read {f.name}: {e}")

        if not all_dfs:
            raise RuntimeError("No readable CSV files found.")
        df = pd.concat(all_dfs, ignore_index=True, sort=False)
        print(f"    Combined: {len(df):,} rows")

    elif p.is_file():
        df = _read_csv_resilient(p)
        print(f"    Single file: {len(df):,} rows")
        df['__source_file'] = p.name
        df['__source_path'] = str(p.resolve())
        df['__file_vehicle_id'] = p.stem

    else:
        raise FileNotFoundError(f"Path not found: {path}")

    df = _canonicalize_columns(df)
    df = df.dropna(how='all')
    df = _resolve_vehicle_id(df)
    df = _apply_latest_device_per_file_identity(df)
    df = _apply_vehicle_id_aliases(df, _load_vehicle_id_aliases(path))
    # Reduce memory footprint on very large files.
    if 'vehicle_id' in df.columns and df['vehicle_id'].dtype == object:
        df['vehicle_id'] = df['vehicle_id'].astype('category')

    # â”€â”€ Numeric casting (must run before segmentation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    num_cols = [
        'fuelLevel', 'chargingCurrent', 'battPackVoltage',
        'maxCellVoltage', 'minCellVoltage', 'maxCellTemp', 'minCellTemp',
        'battPowerIn', 'regenerationPower', 'hrlfc', '_utc_num',
        'totalDistance', 'charge_calc', 'dt_sec', 'vehicleSpeed',
        'chargingStatus', 'crankStatus', 'motorCurrent',
        'chg_power_calc',
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if '_utc_num' not in df.columns:
        if 'utc' in df.columns:
            df['_utc_num'] = pd.to_numeric(df['utc'], errors='coerce')
        elif 'DateTime' in df.columns:
            dt = pd.to_datetime(df['DateTime'], errors='coerce')
            df['_utc_num'] = (dt.view('int64') // 10**9).where(dt.notna(), np.nan)
        else:
            print("    [WARN] Missing time column (_utc_num/utc/DateTime). Using row index as fallback.")
            df['_utc_num'] = np.arange(len(df), dtype=float)

    # Optional incremental pre-filter: keep only rows newer than watermark
    if since_utc is not None and np.isfinite(float(since_utc)):
        cutoff = float(since_utc) - max(0.0, float(overlap_sec))
        if '_utc_num' in df.columns:
            keep_mask = _finite_series(df['_utc_num']) >= cutoff
            kept = int(keep_mask.sum())
            print(f"    Incremental cutoff: utc >= {cutoff:.0f} (watermark={float(since_utc):.0f}, overlap={float(overlap_sec):.0f}s)")
            print(f"    Incremental rows kept: {kept:,} / {len(df):,}")
            df_attrs = dict(getattr(df, 'attrs', {}) or {})
            df = df.loc[keep_mask].copy()
            df.attrs.update(df_attrs)
            if len(df) == 0:
                raise RuntimeError("[INCREMENTAL] No rows found newer than the saved watermark.")

    # â”€â”€ State-machine segmentation (from Data_cleaning_improved notebook) â”€â”€â”€â”€â”€â”€
    # Only runs when the CSVs are raw (not pre-processed by the notebook).
    if 'bucket' not in df.columns or 'session_id' not in df.columns:
        print("    Running state-machine session segmentation (CHARGING/DRIVING/CRANKING)...")
        from tqdm import tqdm as _tqdm
        vehicle_groups = list(df.groupby('vehicle_id', observed=True))
        seg_parts = [
            segment_sessions(g.copy())
            for _, g in _tqdm(vehicle_groups, desc='    Segmenting vehicles')
        ]
        df = pd.concat(seg_parts, ignore_index=True)
        vc = df['bucket'].value_counts()
        print(f"    Segmentation done  -> {vc.to_dict()}")
    else:
        df['bucket'] = _normalize_bucket_series(df['bucket'])
        vc = df['bucket'].value_counts(dropna=False)
        print("    [INFO] 'bucket' column found - using pre-cleaned sessions after bucket normalization.")
        print(f"    Bucket distribution -> {vc.to_dict()}")

    # â”€â”€ GPS-UTC â†’ IST DateTime conversion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Formula: (GPS_seconds + Unix_offset + IST_offset) / 86400 + Excel_epoch
    if 'DateTime' not in df.columns:
        print("    Converting GPS-UTC to IST DateTime...")
        df['DateTime'] = ((df['_utc_num'] + 946684800 + 19800) / 86400) + 25569
        df['DateTime'] = pd.to_datetime(
            df['DateTime'], unit='D', origin='1899-12-30', errors='coerce'
        )
        df = df.dropna(subset=['DateTime']).reset_index(drop=True)
        print(f"    Shape after DateTime conversion: {len(df):,} rows")

    # â”€â”€ Session-level derived signals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # dt_sec: time delta within each session (seconds)
    if 'dt_sec' not in df.columns:
        df = df.sort_values(['vehicle_id', 'session_id', 'DateTime']).reset_index(drop=True)
        df['dt_sec'] = (
            df.groupby(['vehicle_id', 'session_id'])['DateTime']
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

    # charge_calc: Coulomb counting (Ah = |I| * dt / 3600)
    if 'charge_calc' not in df.columns and 'chargingCurrent' in df.columns:
        df['charge_calc'] = (df['chargingCurrent'].abs() * df['dt_sec']) / 3600.0

    # chg_power_calc: instantaneous charging power (W), only during CHARGING rows
    if 'chg_power_calc' not in df.columns:
        if 'battPackVoltage' in df.columns and 'chargingCurrent' in df.columns:
            df['chg_power_calc'] = np.where(
                df['bucket'] == 'CHARGING',
                df['battPackVoltage'] * df['chargingCurrent'].abs(),
                0.0,
            )
        else:
            df['chg_power_calc'] = 0.0

    # Fix hrlfc:
    # - Some datasets store raw 16-bit modulo counters (0..65535) and need unwrap.
    # - Others already store cumulative values > 65535 and should NOT be clamped.
    if 'hrlfc' in df.columns:
        print("    Fixing hrlfc (mode-aware unwrap)...")
        work = df[['vehicle_id', '_utc_num', 'hrlfc']].copy()
        work['hrlfc'] = pd.to_numeric(work['hrlfc'], errors='coerce')
        work = work.sort_values(['vehicle_id', '_utc_num'], kind='mergesort')

        raw = work['hrlfc']
        valid_vid = work['vehicle_id'].notna()
        high_flag = raw > HRLFC_VALID_MAX

        # Vehicle-level mode detection:
        # if many readings exceed the 16-bit envelope, treat stream as already unwrapped.
        high_frac = (
            high_flag.where(raw.notna())
            .groupby(work['vehicle_id'], sort=False, observed=True)
            .mean()
        )
        high_max = raw.groupby(work['vehicle_id'], sort=False, observed=True).max()
        already_unwrapped_vid = high_frac.index[
            (high_frac.fillna(0.0) >= 0.20) |
            (high_max.fillna(-np.inf) > (1.20 * HRLFC_VALID_MAX))
        ]

        unwrapped_mask = work['vehicle_id'].isin(already_unwrapped_vid)
        wrapped_mask = valid_vid & (~unwrapped_mask)

        # Clamp only wrapped-mode streams to remove out-of-range spikes before unwrap.
        h_for_unwrap = raw.where(~wrapped_mask | (raw <= HRLFC_VALID_MAX), np.nan)

        prev_vid = work['vehicle_id'].shift()
        prev_h = h_for_unwrap.shift()
        same_vid = work['vehicle_id'].eq(prev_vid)
        wrap_flag = (
            wrapped_mask &
            same_vid &
            h_for_unwrap.notna() &
            prev_h.notna() &
            ((h_for_unwrap + 500.0) < prev_h)
        )

        wraps = pd.Series(0.0, index=work.index, dtype=float)
        if wrapped_mask.any():
            wraps.loc[wrapped_mask] = (
                wrap_flag.loc[wrapped_mask]
                .groupby(work.loc[wrapped_mask, 'vehicle_id'], sort=False, observed=True)
                .cumsum()
                .astype(float)
            )

        h_fixed = h_for_unwrap.astype(float) + wraps * HRLFC_WRAP_MOD
        # For already-unwrapped streams, keep original values unchanged.
        h_fixed.loc[unwrapped_mask] = raw.loc[unwrapped_mask]

        df.loc[work.index, 'hrlfc'] = h_fixed.values

    # Derived signals
    if 'maxCellVoltage' in df.columns and 'minCellVoltage' in df.columns:
        df['volt_spread'] = df['maxCellVoltage'] - df['minCellVoltage']
    if 'maxCellTemp' in df.columns and 'minCellTemp' in df.columns:
        df['temp_spread'] = df['maxCellTemp'] - df['minCellTemp']

    print(f"    Loaded {len(df):,} rows | {df['vehicle_id'].nunique()} vehicles")
    vc = df['vehicle_id'].value_counts(dropna=False).head(5)
    print("    Top vehicle row counts:")
    for vid, cnt in vc.items():
        print(f"      {vid}: {cnt:,}")
    return df


# ------------------------------------------------------------------------------
# STEP 1 - SOC GAIN HELPER
# ------------------------------------------------------------------------------
def compute_soc_gain(soc_series, max_tail=MAX_TAIL_STRIP):
    """Tail-drop corrected SOC gain  same logic as notebook."""
    s = pd.Series(soc_series).dropna().astype(float).values
    if len(s) < 2:
        return np.nan
    removed = 0
    while len(s) > 1 and s[-1] < s[-2] and removed < max_tail:
        s = s[:-1]
        removed += 1
    return float(s[-1] - s[0]) if len(s) >= 2 else np.nan


# ------------------------------------------------------------------------------
# STEP 2 - BUILD SESSION FEATURE TABLE
# ------------------------------------------------------------------------------
def build_session_table(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Building charging session table...")

    work = df.copy()
    if 'bucket' in work.columns:
        work['bucket'] = _normalize_bucket_series(work['bucket'])
    else:
        work['bucket'] = 'OTHERS'

    chg = work[work['bucket'] == 'CHARGING'].copy()

    # Fallback for datasets with unusable/missing bucket labels:
    # infer CHARGING rows from current and split into contiguous sessions.
    if len(chg) == 0 and 'chargingCurrent' in work.columns:
        curr_abs = _finite_series(work['chargingCurrent']).abs().fillna(0.0)
        chg_mask = curr_abs > float(SEG_EPS_CURR_ENTRY)

        if 'vehicleSpeed' in work.columns:
            spd = _finite_series(work['vehicleSpeed'])
            chg_mask &= spd.fillna(0.0) <= max(10.0, 5.0 * float(SEG_EPS_SPEED))

        if int(chg_mask.sum()) > 0:
            print("    [WARN] No CHARGING bucket rows found; inferring charging sessions from chargingCurrent.")
            work = work.sort_values(['vehicle_id', '_utc_num'], kind='mergesort').reset_index(drop=True)
            curr_abs = _finite_series(work['chargingCurrent']).abs().fillna(0.0)
            chg_mask = curr_abs > float(SEG_EPS_CURR_ENTRY)
            if 'vehicleSpeed' in work.columns:
                spd = _finite_series(work['vehicleSpeed'])
                chg_mask &= spd.fillna(0.0) <= max(10.0, 5.0 * float(SEG_EPS_SPEED))

            utc = _finite_series(work['_utc_num']) if '_utc_num' in work.columns else pd.Series(
                np.arange(len(work), dtype=float), index=work.index
            )
            prev_chg = chg_mask.groupby(work['vehicle_id'], observed=True).shift(fill_value=False)
            prev_utc = utc.groupby(work['vehicle_id'], observed=True).shift()
            gap_break = ((utc - prev_utc) / 60.0 > float(SEG_MAX_GAP_MIN)).fillna(False)
            start_flag = chg_mask & ((~prev_chg) | gap_break)
            sess_local = start_flag.groupby(work['vehicle_id'], observed=True).cumsum() - 1

            work['bucket'] = np.where(chg_mask, 'CHARGING', 'OTHERS')
            work['session_id'] = sess_local.where(chg_mask, np.nan)
            chg = work[work['bucket'] == 'CHARGING'].copy()

    if len(chg) == 0:
        if 'bucket' in work.columns:
            bvc = work['bucket'].value_counts(dropna=False).to_dict()
            raise RuntimeError(
                "[CRITICAL] No CHARGING rows found in input data after bucket normalization/fallback. "
                f"bucket counts: {bvc}"
            )
        raise RuntimeError("[CRITICAL] Missing 'bucket' column and no charging rows could be inferred.")

    if 'charge_calc' not in chg.columns and {'chargingCurrent', 'dt_sec'}.issubset(chg.columns):
        chg['charge_calc'] = (chg['chargingCurrent'].abs() * chg['dt_sec']) / 3600.0

    missing_required = [c for c in ('fuelLevel', 'charge_calc') if c not in chg.columns]
    if missing_required:
        present = ", ".join(sorted(map(str, chg.columns.tolist())))
        raise RuntimeError(
            "[CRITICAL] Missing required charging-session inputs: "
            f"{missing_required}. "
            "Expected canonical columns or common aliases (e.g., SOC/fuelLevel and chargingCurrent+dt_sec). "
            f"Available CHARGING columns: {present}"
        )

    # Only aggregate columns that actually exist
    agg_dict = {
        'soc_series'       : ('fuelLevel',         list),
        'ah_total'         : ('charge_calc',        'sum'),
        'start_utc'        : ('_utc_num',           'first'),
        'end_utc'          : ('_utc_num',           'last'),
        'packets'          : ('bucket',             'count'),
    }
    optional = {
        'wh_total'         : ('chg_power_calc',     'sum'),
        'hrlfc_start'      : ('hrlfc',              'first'),
        'hrlfc_end'        : ('hrlfc',              'last'),
        'hrlfc_mid'        : ('hrlfc',              'mean'),
        'volt_spread_mean' : ('volt_spread',         'mean'),
        'volt_spread_max'  : ('volt_spread',         'max'),
        'pack_v_mean'      : ('battPackVoltage',     'mean'),
        'pack_v_min'       : ('battPackVoltage',     'min'),
        'pack_v_max'       : ('battPackVoltage',     'max'),
        'cell_v_max_mean'  : ('maxCellVoltage',      'mean'),
        'cell_v_max_peak'  : ('maxCellVoltage',      'max'),
        'cell_v_min_mean'  : ('minCellVoltage',      'mean'),
        'cell_v_min_floor' : ('minCellVoltage',      'min'),
        'current_mean'     : ('chargingCurrent',     'mean'),
        'current_max'      : ('chargingCurrent',     'max'),
        'temp_max'         : ('maxCellTemp',         'max'),
        'temp_start'       : ('maxCellTemp',         'first'),
        'temp_end'         : ('maxCellTemp',         'last'),
        'temp_spread_mean' : ('temp_spread',         'mean'),
        'bms_ah'           : ('bms_ah',              'mean'),
        'bms_soh'          : ('bms_soh',             'mean'),
        'bms_init_cap'     : ('bms_init_cap',        'mean'),
    }
    for key, (col, func) in optional.items():
        if col in chg.columns:
            agg_dict[key] = (col, func)

    sessions = (
        chg.groupby(['vehicle_id', 'session_id'])
        .agg(**agg_dict)
        .reset_index()
    )

    # SOC gain
    sessions['delta_soc_pct'] = sessions['soc_series'].apply(compute_soc_gain)
    sessions['soc_start'] = sessions['soc_series'].apply(
        lambda s: float(pd.Series(s).dropna().iloc[0]) if len(pd.Series(s).dropna()) > 0 else np.nan
    )
    sessions['soc_end'] = sessions['soc_series'].apply(
        lambda s: float(pd.Series(s).dropna().iloc[-1]) if len(pd.Series(s).dropna()) > 0 else np.nan
    )

    sessions['duration_min'] = (sessions['end_utc'] - sessions['start_utc']) / 60

    # Implied capacity: Q = Ah / (dSOC/100)
    sessions['implied_Q_Ah'] = np.where(
        sessions['delta_soc_pct'] > 0,
        sessions['ah_total'] / (sessions['delta_soc_pct'] / 100.0),
        np.nan
    )

    # Health indicators (only if source columns available)
    if 'wh_total' in sessions.columns and 'ah_total' in sessions.columns:
        sessions['charge_efficiency'] = np.where(
            sessions['ah_total'] > 0,
            sessions['wh_total'] / sessions['ah_total'], np.nan
        )
    if 'current_mean' in sessions.columns:
        sessions['avg_c_rate'] = sessions['current_mean'].abs() / Q_RATED_AH
    if 'temp_end' in sessions.columns and 'temp_start' in sessions.columns:
        sessions['dT_per_crate'] = np.where(
            sessions.get('avg_c_rate', pd.Series(0, index=sessions.index)) > 0,
            (sessions['temp_end'] - sessions['temp_start']) /
            sessions.get('avg_c_rate', pd.Series(1, index=sessions.index)),
            np.nan
        )
    if 'pack_v_mean' in sessions.columns and 'pack_v_max' in sessions.columns:
        sessions['pack_v_norm'] = sessions['pack_v_mean'] / sessions['pack_v_max'].replace(0, np.nan)
    if 'duration_min' in sessions.columns:
        sessions['ah_per_min'] = np.where(
            sessions['duration_min'] > 0,
            sessions['ah_total'] / sessions['duration_min'], np.nan
        )
    if 'volt_spread_max' in sessions.columns:
        sessions['volt_spread_eoc'] = sessions['volt_spread_max']

    # Normalize non-finite counters/timestamps early.
    for c in ['start_utc', 'end_utc', 'hrlfc_start', 'hrlfc_end', 'hrlfc_mid']:
        if c in sessions.columns:
            sessions[c] = _finite_series(sessions[c])

    # Keep hrlfc_mid and start_utc separate. Do not fill hrlfc_mid with UTC values,
    # otherwise the same axis mixes incompatible units and creates artificial clusters.

    print(f"    {len(sessions):,} charging sessions built")
    return sessions


# ------------------------------------------------------------------------------
# STEP 3 - FILTER + PSEUDO-SOH LABELS (dynamic per-vehicle bounds)
# ------------------------------------------------------------------------------
def compute_soh_labels(
    sessions: pd.DataFrame,
    init_capacity_overrides: dict = None,
    prior_pack_context: dict = None,
) -> pd.DataFrame:
    print("[3/6] Computing SOH labels...")

    # Basic sanity filter  loose, let per-vehicle IQR handle the rest
    base_mask = (
        (sessions['delta_soc_pct'] >= MIN_DELTA_SOC) &
        (sessions['ah_total']      >= MIN_AH) &
        (sessions['implied_Q_Ah']  > 0) &
        (sessions['implied_Q_Ah'].notna()) &
        (sessions['duration_min']  > 2)
    )

    # Diagnostic
    print(f"    Diagnostic - Charging sessions meeting individual criteria:")
    print(f"      - delta_SOC >= {MIN_DELTA_SOC}%: {(sessions['delta_soc_pct'] >= MIN_DELTA_SOC).sum()}")
    print(f"      - Ah_total  >= {MIN_AH}: {(sessions['ah_total'] >= MIN_AH).sum()}")
    print(f"      - Implied Q > 0: {(sessions['implied_Q_Ah'] > 0).sum()}")
    print(f"      - Duration  > 2 min: {(sessions['duration_min'] > 2).sum()}")
    print(f"    Charging sessions passing base filter: {base_mask.sum():,} / {len(sessions):,}")

    df = sessions[base_mask].copy()
    if len(df) == 0:
        raise RuntimeError(
            "[CRITICAL] No sessions passed base filter.\n"
            "Check that your CSV has 'charge_calc', 'fuelLevel', and 'bucket'=='CHARGING' rows."
        )

    df['weight'] = df['delta_soc_pct'] ** 2

    out = []
    for vid, g in df.groupby('vehicle_id'):
        axis_col = _pick_axis_col(g)
        if axis_col == '__session_idx':
            g = g.copy()
            g['__session_idx'] = np.arange(len(g), dtype=float)
        g = g.sort_values(axis_col).reset_index(drop=True)
        x_axis = _finite_series(g[axis_col]).values.astype(float)

        #  Per-vehicle dynamic Q bounds (5th95th percentile) 
        q_lo = g['implied_Q_Ah'].quantile(0.05)
        q_hi = g['implied_Q_Ah'].quantile(0.95)
        g = g[
            (g['implied_Q_Ah'] >= q_lo) &
            (g['implied_Q_Ah'] <= q_hi)
        ].reset_index(drop=True)

        if len(g) < 5:
            print(f"    [INFO] Vehicle {vid}: only {len(g)} sessions after IQR filter, skipping")
            continue

        # BMS Coulomb-counting correction: scale implied_Q_Ah to match BMS reported Ah
        bms_k = _compute_bms_correction_factor(g) if BMS_CALIBRATION_ENABLED else 1.0
        if abs(bms_k - 1.0) > 0.001:
            g = g.copy()
            g['implied_Q_Ah'] = g['implied_Q_Ah'] * bms_k
        g['bms_correction_factor'] = bms_k

        q_est = _estimate_initial_capacity_ah(g, first_cycles=INIT_CAPACITY_CYCLES)
        q_ref_robust = q_est.get('q_ref_robust_ah', np.nan)
        q_init_first_n = q_est.get('q_init_first_ah', np.nan)
        q_settle = q_est.get('q_settle_ah', np.nan)
        q_ref_for_soh = q_est.get('q_base_ah', np.nan)
        q_source = q_est.get('source', 'unknown')
        q_prev_stable = np.nan
        if isinstance(init_capacity_overrides, dict) and (vid in init_capacity_overrides):
            q_prev_stable = pd.to_numeric(pd.Series([init_capacity_overrides.get(vid)]), errors='coerce').iloc[0]

        if isinstance(init_capacity_overrides, dict) and (vid in init_capacity_overrides):
            ov = pd.to_numeric(pd.Series([init_capacity_overrides.get(vid)]), errors='coerce').iloc[0]
            if np.isfinite(ov) and (ov > 0):
                q_ref_for_soh = float(ov)
                q_source = 'state_override'
                # Guard: if early implied capacity is much higher than cached baseline,
                # likely pack replacement / telemetry reset -> re-anchor from current data.
                q_early_now = _finite_series(g.head(max(int(INIT_CAPACITY_CYCLES), 5))['implied_Q_Ah']).median()
                if np.isfinite(q_early_now) and (q_early_now > 1.25 * q_ref_for_soh):
                    q_ref_for_soh = q_est.get('q_base_ah', q_early_now)
                    q_source = 'override_rejected_reanchor'

        pack_cfg = _infer_pack_config_options(g, q_anchor=q_ref_for_soh, q_prev=q_prev_stable)

        prior_cfg = 'unknown'
        prior_epoch = 0
        if isinstance(prior_pack_context, dict) and (vid in prior_pack_context):
            pctx = prior_pack_context.get(vid, {}) if isinstance(prior_pack_context.get(vid, {}), dict) else {}
            prior_cfg = str(pctx.get('pack_config_guess', 'unknown'))
            prior_epoch = int(pd.to_numeric(pd.Series([pctx.get('config_epoch_id', 0)]), errors='coerce').fillna(0).iloc[0])

        cfg_guess = str(pack_cfg.get('config_guess', 'unknown'))
        cfg_conf = pd.to_numeric(pd.Series([pack_cfg.get('score_confidence', np.nan)]), errors='coerce').iloc[0]
        cfg_source = str(pack_cfg.get('options_source', ''))
        force_split_signal = cfg_source.startswith('force_208_')
        config_changed_strong = False
        if prior_cfg != 'unknown' and cfg_guess != 'unknown' and cfg_guess != prior_cfg:
            if force_split_signal or (np.isfinite(cfg_conf) and (cfg_conf >= float(PACK_CONFIG_SWITCH_MIN_CONF))):
                config_changed_strong = True
            else:
                # Low-confidence disagreement: lock to previous config.
                pack_cfg['config_guess'] = prior_cfg
                ps, pp = _parse_pack_config_guess(prior_cfg)
                pack_cfg['series_cells'] = ps
                pack_cfg['parallel_count'] = pp
                pack_cfg['config_family'] = f"{int(ps)}s" if np.isfinite(ps) else str(pack_cfg.get('config_family', 'unknown'))
                pack_cfg['capacity_options'] = _capacity_options_from_pack_config(prior_cfg)
                pack_cfg['options_source'] = 'locked_previous_config'

        config_epoch_id = prior_epoch + 1 if config_changed_strong else prior_epoch
        cap_opts_local = tuple(pack_cfg.get('capacity_options', EXPECTED_CAPACITY_OPTIONS_AH))

        q_ref_cal, q_nom, q_scale, q_err_pct = _calibrate_capacity_ah(
            q_ref_for_soh, options=cap_opts_local
        )
        use_calibrated = (
            np.isfinite(q_ref_cal) and
            np.isfinite(q_err_pct) and
            (q_err_pct <= float(CAPACITY_CAL_MAX_ERR_PCT))
        )
        q_base_for_soh = q_ref_cal if use_calibrated else q_ref_for_soh
        q_base_source = f"{q_source}|calibrated" if use_calibrated else f"{q_source}|raw_high_cal_err"
        if (not np.isfinite(q_base_for_soh)) or (q_base_for_soh <= 0):
            q_base_for_soh = q_ref_robust
            q_base_source = f"{q_source}|fallback_robust"
        if (not np.isfinite(q_base_for_soh)) or (q_base_for_soh <= 0):
            q_med_fallback = _finite_series(g['implied_Q_Ah']).median()
            q_base_for_soh = q_med_fallback if np.isfinite(q_med_fallback) and (q_med_fallback > 0) else np.nan
            q_base_source = f"{q_source}|fallback_median"
        replacement_hint = str(q_source).startswith('override_rejected_reanchor')
        if np.isfinite(q_prev_stable) and (q_prev_stable > 0) and np.isfinite(q_base_for_soh) and (q_base_for_soh > 0) and (not replacement_hint):
            lo = float(q_prev_stable) * (1.0 - float(BASELINE_MAX_DRIFT_PCT) / 100.0)
            hi = float(q_prev_stable) * (1.0 + float(BASELINE_MAX_DRIFT_PCT) / 100.0)
            q_bounded = min(max(float(q_base_for_soh), lo), hi)
            if abs(q_bounded - float(q_base_for_soh)) > 1e-9:
                q_base_for_soh = q_bounded
                q_base_source = f"{q_base_source}|drift_clamped"

        # BMS initial capacity override: use BMS rated cap as q_base when available
        if BMS_CALIBRATION_ENABLED and BMS_INIT_CAP_OVERRIDE and 'bms_init_cap' in g.columns:
            q_bms_init = _finite_series(g['bms_init_cap']).median()
            if np.isfinite(q_bms_init) and q_bms_init > 0:
                q_base_for_soh = float(q_bms_init)
                q_base_source += '|bms_init_cap'

        q_nom_strict = np.nan
        q_base_strict = np.nan
        if STRICT_PACK_BASELINE_ENABLED:
            q_nom_strict, q_base_strict = _strict_baseline_from_pack_config(pack_cfg.get('config_guess', 'unknown'))
            if np.isfinite(q_base_strict) and (q_base_strict > 0):
                q_base_for_soh = float(q_base_strict)
                q_base_source = f"{q_base_source}|strict_cfg_baseline"
                if np.isfinite(q_nom_strict) and (q_nom_strict > 0):
                    q_nom = float(q_nom_strict)
            elif np.isfinite(q_prev_stable) and (q_prev_stable > 0):
                q_base_for_soh = float(q_prev_stable)
                q_base_source = f"{q_base_source}|strict_fallback_prev"

        g['q_rated']   = q_ref_robust
        g['q_init_first_cycles_ah'] = q_init_first_n
        g['q_settle_window_ah'] = q_settle
        g['q_rated_raw_ah'] = q_ref_for_soh
        g['q_rated_cal_ah'] = q_ref_cal
        g['q_nominal_strict_ah'] = q_nom_strict
        g['q_base_strict_usable_ah'] = q_base_strict
        g['q_rated_used_for_soh_ah'] = q_base_for_soh
        g['q_base_source'] = q_base_source
        g['q_nominal_ah'] = q_nom
        g['q_scale_factor'] = q_scale
        g['q_match_err_pct'] = q_err_pct
        g['pack_series_cells'] = pack_cfg.get('series_cells', np.nan)
        g['pack_parallel_guess'] = pack_cfg.get('parallel_count', np.nan)
        g['pack_voltage_ref_v'] = pack_cfg.get('voltage_ref_v', np.nan)
        g['pack_series_est_cellratio'] = pack_cfg.get('series_est_cellratio', np.nan)
        g['pack_series_est_nominal'] = pack_cfg.get('series_est_nominal', np.nan)
        g['pack_score_confidence'] = pack_cfg.get('score_confidence', np.nan)
        g['pack_q_data_median_ah'] = pack_cfg.get('q_data_median_ah', np.nan)
        g['pack_q_data_count'] = pack_cfg.get('q_data_count', 0)
        g['pack_options_source'] = pack_cfg.get('options_source', 'default')
        g['pack_series_source'] = pack_cfg.get('series_source', 'unknown')
        g['pack_config_family'] = pack_cfg.get('config_family', 'unknown')
        g['pack_config_guess'] = pack_cfg.get('config_guess', 'unknown')
        g['pack_config_changed_strong'] = bool(config_changed_strong)
        g['config_epoch_id'] = int(config_epoch_id)
        g['pack_capacity_options_ah'] = ",".join(f"{float(v):.1f}" for v in cap_opts_local)
        g['soh_label'] = (g['implied_Q_Ah'] / q_base_for_soh * 100).clip(0, 100.0)

        #  Smooth + enforce monotone 
        g['soh_smooth'] = (
            g['soh_label']
            .rolling(window=7, min_periods=2, center=True)
            .median()
            .bfill()
            .ffill()
        )
        min_drop_label = _adaptive_min_drop(g['soh_label'].values, SOFT_MIN_DROP_LABEL)
        g['soh_smooth'] = _soft_monotone_curve(
            g['soh_smooth'].values,
            x=x_axis,
            min_total_drop=min_drop_label,
            smooth_window=9,
        )

        out.append(g)
        print(
            f"    {vid}: {len(g)} sessions | "
            f"Q_init_first_{int(INIT_CAPACITY_CYCLES)}={q_init_first_n:.1f} Ah | "
            f"Q_settle={q_settle:.1f} Ah | "
            f"Pack={str(g['pack_config_guess'].iloc[0])} "
            f"(Vref={g['pack_voltage_ref_v'].iloc[0]:.1f}V, conf={g['pack_score_confidence'].iloc[0]:.3f}, "
            f"Qmed={g['pack_q_data_median_ah'].iloc[0]:.1f}Ah/{int(pd.to_numeric(pd.Series([g['pack_q_data_count'].iloc[0]]), errors='coerce').fillna(0).iloc[0])}, "
            f"opts={g['pack_capacity_options_ah'].iloc[0]}, src={g['pack_options_source'].iloc[0]}) | "
            f"Q_ref_robust={q_ref_robust:.1f} Ah -> Q_ref_cal={q_ref_cal:.1f} Ah "
            f"(nom={q_nom:.0f}, scale={q_scale:g}, err={q_err_pct:.1f}%) | "
            f"Q_base_for_SOH={q_base_for_soh:.1f} Ah ({q_base_source}) | "
            f"SOH {g['soh_label'].min():.1f}%{g['soh_label'].max():.1f}%"
        )

    if not out:
        raise RuntimeError(
            "[CRITICAL] Pipeline stopped: No valid sessions found after filtering.\n"
            "Check if your data has enough charging sessions with >3% SOC swing."
        )

    df = pd.concat(out, ignore_index=True)
    print(f"    Total labeled sessions: {len(df):,}")
    return df


# ------------------------------------------------------------------------------
# STEP 4 - XGBOOST: FEATURES -> SOH
# ------------------------------------------------------------------------------
FEATURE_COLS = [
    'delta_soc_pct', 'ah_total', 'volt_spread_mean', 'volt_spread_eoc',
    'temp_max', 'dT_per_crate', 'avg_c_rate', 'pack_v_norm',
    'ah_per_min', 'charge_efficiency', 'hrlfc_mid',
    'duration_min', 'temp_spread_mean',
]

def train_xgboost_soh(labeled: pd.DataFrame) -> dict:
    print("[4/6] Training XGBoost SOH model...")
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    results = {}
    n_fit = 0
    n_fallback = 0

    from tqdm import tqdm
    vehicles = list(labeled.groupby('vehicle_id'))
    for vid, g in tqdm(vehicles, desc='Training XGBoost', unit='vehicle'):
        sort_col = _pick_axis_col(g)
        if sort_col == '__session_idx':
            g = g.copy()
            g['__session_idx'] = np.arange(len(g), dtype=float)
        g = g.sort_values(sort_col).reset_index(drop=True)
        x_axis = _finite_series(g[sort_col]).values.astype(float)

        feats = [c for c in FEATURE_COLS if c in g.columns]
        X_raw = g[feats].replace([np.inf, -np.inf], np.nan)
        X = X_raw.fillna(X_raw.median(numeric_only=True)).fillna(0.0)
        y = g['soh_smooth'].values

        if len(X) < 8:
            # Keep per-vehicle output instead of dropping the vehicle.
            min_drop_fb = _adaptive_min_drop(g['soh_smooth'].values, SOFT_MIN_DROP_LABEL)
            g['soh_xgb'] = _soft_monotone_curve(
                g['soh_smooth'].values,
                x=x_axis,
                min_total_drop=min_drop_fb,
                smooth_window=9,
            )
            fi = pd.Series(0.0, index=feats if len(feats) > 0 else ['no_features'])
            results[vid] = {
                'model': None, 'scaler': None, 'features': feats,
                'sessions': g, 'mae': np.nan, 'r2': np.nan, 'feature_importance': fi,
            }
            n_fallback += 1
            print(f"    {vid}: insufficient sessions ({len(X)}), using fallback SOH curve")
            continue

        split  = max(1, int(len(X) * 0.8))
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y[:split],       y[split:]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        # Monotone constraint: SOH must not increase with hrlfc
        hrlfc_idx = feats.index('hrlfc_mid') if ('hrlfc_mid' in feats and sort_col == 'hrlfc_mid') else -1
        mono      = tuple(-1 if i == hrlfc_idx else 0 for i in range(len(feats)))

        model = XGBRegressor(
            n_estimators         = 400,
            max_depth            = 3,
            learning_rate        = 0.03,
            subsample            = 0.7,
            colsample_bytree     = 0.7,
            reg_alpha            = 1.0,
            reg_lambda           = 5.0,
            min_child_weight     = 5,
            monotone_constraints = mono,
            random_state         = 42,
            verbosity            = 0,
        )
        model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)

        # Predict + post-process monotone
        raw_pred = model.predict(scaler.transform(X.values))
        min_drop_xgb = _adaptive_min_drop(y, SOFT_MIN_DROP_XGB)
        g['soh_xgb'] = _soft_monotone_curve(
            raw_pred,
            x=x_axis,
            min_total_drop=min_drop_xgb,
            smooth_window=11,
        )

        mae = mean_absolute_error(y_te, model.predict(X_te_s))
        r2  = r2_score(y_te, model.predict(X_te_s)) if len(y_te) > 1 else np.nan
        fi  = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

        results[vid] = {
            'model': model, 'scaler': scaler, 'features': feats,
            'sessions': g, 'mae': mae, 'r2': r2, 'feature_importance': fi,
        }
        n_fit += 1
        print(f"    {vid}: sessions={len(g)}, MAE={mae:.2f}%, R={r2:.3f}")
        print(f"    Top features: {', '.join(fi.head(3).index.tolist())}")

    if not results:
        raise RuntimeError(
            "[CRITICAL] XGBoost: no vehicles had enough sessions to train.\n"
            "Need at least 8 labeled sessions per vehicle."
        )
    print(f"    XGBoost vehicles: trained={n_fit}, fallback={n_fallback}, total={len(results)}")
    return results


# ------------------------------------------------------------------------------
# STEP 5 - LSTM: SOH SEQUENCE -> TRAJECTORY
# ------------------------------------------------------------------------------
def build_lstm_sequences(soh_series: np.ndarray, lookback: int = 10):
    X, y = [], []
    for i in range(lookback, len(soh_series)):
        X.append(soh_series[i - lookback:i])
        y.append(soh_series[i])
    return np.array(X)[..., np.newaxis], np.array(y)


def train_lstm_trajectory(xgb_results: dict, lookback: int = 10) -> dict:
    print("[5/6] Training LSTM trajectory model...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler

    lstm_results = {}

    from tqdm import tqdm
    for vid, res in tqdm(xgb_results.items(), desc='Training LSTM', unit='vehicle'):
        sort_col = _pick_axis_col(res['sessions'])
        g = res['sessions'].copy()
        if sort_col == '__session_idx':
            g['__session_idx'] = np.arange(len(g), dtype=float)
        g = g.sort_values(sort_col)
        soh_seq = g['soh_xgb'].values.astype(float)
        hrlfc_seq = _finite_series(g[sort_col]).values

        finite = np.isfinite(hrlfc_seq) & np.isfinite(soh_seq)
        hrlfc_seq = hrlfc_seq[finite]
        soh_seq = soh_seq[finite]

        if len(soh_seq) < lookback + 5:
            print(f"    {vid}: too few sessions ({len(soh_seq)}) for LSTM, skipping")
            continue

        scaler_lstm = MinMaxScaler()
        soh_scaled  = scaler_lstm.fit_transform(soh_seq.reshape(-1, 1)).flatten()

        X, y  = build_lstm_sequences(soh_scaled, lookback)
        split = max(1, int(len(X) * 0.8))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        tf.random.set_seed(42)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1,  activation='linear'),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

        model.fit(
            X_tr, y_tr,
            validation_data = (X_te, y_te),
            epochs          = 200,
            batch_size      = min(16, len(X_tr)),
            callbacks       = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10),
            ],
            verbose=0,
        )

        soh_pred = scaler_lstm.inverse_transform(
            model.predict(X, verbose=0).reshape(-1, 1)
        ).flatten()
        x_pred = hrlfc_seq[lookback:]
        min_drop_lstm = _adaptive_min_drop(soh_seq, SOFT_MIN_DROP_LSTM)
        soh_pred = _soft_monotone_curve(
            soh_pred,
            x=x_pred,
            min_total_drop=min_drop_lstm,
            smooth_window=7,
        )

        lstm_results[vid] = {
            'model': model, 'scaler': scaler_lstm, 'lookback': lookback,
            'soh_seq': soh_seq, 'hrlfc_seq': hrlfc_seq, 'soh_pred': soh_pred,
        }
        print(f"    {vid}: LSTM trained on {len(soh_seq)} sessions")

    return lstm_results


# ------------------------------------------------------------------------------
# STEP 6 - RUL: EXTRAPOLATE SOH TO 80% EOL
# ------------------------------------------------------------------------------
def extrapolate_rul(hrlfc_seq, soh_seq, hrlfc_to_days,
                    eol=SOH_EOL, n_mc=200) -> dict:
    hrlfc_seq = np.array(hrlfc_seq, dtype=float)
    soh_seq   = np.array(soh_seq,   dtype=float)

    finite = np.isfinite(hrlfc_seq) & np.isfinite(soh_seq)
    hrlfc_seq = hrlfc_seq[finite]
    soh_seq = soh_seq[finite]
    if len(hrlfc_seq) < 4:
        return {
            'knee_hrlfc'    : np.nan,
            'phase2_slope'  : 0.0,
            'soh_now'       : soh_seq[-1] if len(soh_seq) else np.nan,
            'hrlfc_now'     : hrlfc_seq[-1] if len(hrlfc_seq) else np.nan,
            'rul_hrlfc_p10' : np.nan, 'rul_hrlfc_p50' : np.nan, 'rul_hrlfc_p90' : np.nan,
            'rul_days_p10'  : np.nan, 'rul_days_p50'  : np.nan, 'rul_days_p90'  : np.nan,
        }

    order = np.argsort(hrlfc_seq)
    hrlfc_seq = hrlfc_seq[order]
    soh_seq = soh_seq[order]

    dedup = (
        pd.DataFrame({'x': hrlfc_seq, 'y': soh_seq})
        .groupby('x', as_index=False)['y'].median()
    )
    hrlfc_seq = dedup['x'].values
    soh_seq = dedup['y'].values

    if len(hrlfc_seq) < 3:
        return {
            'knee_hrlfc'    : hrlfc_seq[-1],
            'phase2_slope'  : 0.0,
            'soh_now'       : soh_seq[-1],
            'hrlfc_now'     : hrlfc_seq[-1],
            'rul_hrlfc_p10' : np.nan, 'rul_hrlfc_p50' : np.nan, 'rul_hrlfc_p90' : np.nan,
            'rul_days_p10'  : np.nan, 'rul_days_p50'  : np.nan, 'rul_days_p90'  : np.nan,
            'slope_basis'   : 'insufficient_points',
        }

    if (not np.isfinite(hrlfc_to_days)) or (hrlfc_to_days <= 0):
        hrlfc_to_days = 0.07

    # Knee detection
    if len(soh_seq) >= 5 and np.ptp(hrlfc_seq) > 0:
        smooth  = pd.Series(soh_seq).rolling(5, center=True, min_periods=2).mean().values
        try:
            dy2 = np.gradient(np.gradient(smooth, hrlfc_seq), hrlfc_seq)
            finite_dy2 = np.where(np.isfinite(dy2), dy2, np.nan)
            knee_idx = int(np.nanargmin(finite_dy2)) if np.isfinite(finite_dy2).any() else (len(soh_seq) // 2)
        except Exception:
            knee_idx = len(soh_seq) // 2
    else:
        knee_idx = len(soh_seq) // 2

    knee_hrlfc = hrlfc_seq[knee_idx]
    total_drop = float(soh_seq[0] - soh_seq[-1]) if len(soh_seq) >= 2 else 0.0

    def _fit_line(x, y, label):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if len(x) < 3 or np.all(x == x[0]):
            return None
        s, b, *_ = stats.linregress(x, y)
        if (not np.isfinite(s)) or (not np.isfinite(b)):
            return None
        return {'label': label, 'x': x, 'y': y, 'slope': float(s), 'intercept': float(b)}

    n = len(hrlfc_seq)
    windows = []
    # knee-to-end
    mask_knee = hrlfc_seq >= knee_hrlfc
    if mask_knee.sum() >= 3:
        windows.append((hrlfc_seq[mask_knee], soh_seq[mask_knee], 'knee_tail'))
    # fixed tail fractions
    for frac in [0.30, 0.40, 0.50]:
        k = max(8, int(np.ceil(frac * n)))
        k = min(k, n)
        windows.append((hrlfc_seq[-k:], soh_seq[-k:], f'tail_{int(frac*100)}'))
    # global
    windows.append((hrlfc_seq, soh_seq, 'global'))

    fits = []
    seen = set()
    for xw, yw, label in windows:
        key = (len(xw), float(xw[0]) if len(xw) else np.nan, float(xw[-1]) if len(xw) else np.nan, label)
        if key in seen:
            continue
        seen.add(key)
        f = _fit_line(xw, yw, label)
        if f is not None:
            fits.append(f)

    if not fits:
        return {
            'knee_hrlfc'    : knee_hrlfc,
            'phase2_slope'  : 0.0,
            'soh_now'       : soh_seq[-1],
            'hrlfc_now'     : hrlfc_seq[-1],
            'rul_hrlfc_p10' : np.nan, 'rul_hrlfc_p50' : np.nan, 'rul_hrlfc_p90' : np.nan,
            'rul_days_p10'  : np.nan, 'rul_days_p50'  : np.nan, 'rul_days_p90'  : np.nan,
            'slope_basis'   : 'no_valid_fit',
        }

    tail_fits = [f for f in fits if f['label'] != 'global']
    tail_neg = [f for f in tail_fits if f['slope'] < RUL_MIN_NEG_SLOPE]
    global_fit = next((f for f in fits if f['label'] == 'global'), None)

    chosen = None
    if tail_neg:
        # Robust center of negative tail slopes.
        slopes = np.array([f['slope'] for f in tail_neg], dtype=float)
        target = float(np.median(slopes))
        chosen = min(tail_neg, key=lambda f: abs(f['slope'] - target))
    elif (global_fit is not None) and (global_fit['slope'] < RUL_MIN_NEG_SLOPE) and (total_drop >= RUL_GLOBAL_DROP_TRIGGER_PCT):
        # Tail looks flat but overall decline is meaningful.
        chosen = global_fit
    else:
        # Pick least steep fit only for metadata; RUL likely infinite.
        chosen = min(fits, key=lambda f: abs(f['slope']))

    base_slope = float(chosen['slope'])
    base_intercept = float(chosen['intercept'])
    x2, y2 = chosen['x'], chosen['y']
    slope_basis = chosen['label']

    noise_std = float(np.nanstd(np.diff(y2))) if len(y2) > 2 else 1.0
    noise_std = max(noise_std, 0.3)
    rul_samples = []
    for _ in range(n_mc):
        y_noisy = y2 + np.random.normal(0, noise_std, size=len(y2))
        s, b, *_ = stats.linregress(x2, y_noisy)
        if (not np.isfinite(s)) or (not np.isfinite(b)) or s >= RUL_MIN_NEG_SLOPE:
            continue
        rul_hrlfc = max(0, (eol - b) / s - hrlfc_seq[-1])
        if np.isfinite(rul_hrlfc):
            rul_samples.append(rul_hrlfc)

    if not rul_samples:
        if base_slope < RUL_MIN_NEG_SLOPE:
            rul_det = max(0, (eol - base_intercept) / base_slope - hrlfc_seq[-1])
        else:
            rul_det = 0.0 if soh_seq[-1] <= eol else np.inf

        return {
            'knee_hrlfc'    : knee_hrlfc,
            'phase2_slope'  : base_slope,
            'soh_now'       : soh_seq[-1],
            'hrlfc_now'     : hrlfc_seq[-1],
            'rul_hrlfc_p10' : rul_det, 'rul_hrlfc_p50' : rul_det, 'rul_hrlfc_p90' : rul_det,
            'rul_days_p10'  : rul_det * hrlfc_to_days,
            'rul_days_p50'  : rul_det * hrlfc_to_days,
            'rul_days_p90'  : rul_det * hrlfc_to_days,
            'slope_basis'   : slope_basis,
        }

    rul_arr = np.array(rul_samples)
    return {
        'knee_hrlfc'    : knee_hrlfc,
        'phase2_slope'  : base_slope,
        'soh_now'       : soh_seq[-1],
        'hrlfc_now'     : hrlfc_seq[-1],
        'rul_hrlfc_p10' : np.percentile(rul_arr, 10),
        'rul_hrlfc_p50' : np.percentile(rul_arr, 50),
        'rul_hrlfc_p90' : np.percentile(rul_arr, 90),
        'rul_days_p10'  : np.percentile(rul_arr, 10) * hrlfc_to_days,
        'rul_days_p50'  : np.percentile(rul_arr, 50) * hrlfc_to_days,
        'rul_days_p90'  : np.percentile(rul_arr, 90) * hrlfc_to_days,
        'slope_basis'   : slope_basis,
    }

def compute_all_rul(xgb_results, lstm_results, df_raw, prev_rul_all: dict = None) -> dict:
    print("[6/6] Computing RUL...")
    rul_all = {}

    for vid, res in xgb_results.items():
        g = res['sessions'].copy()
        prev_rr = (prev_rul_all or {}).get(vid, {}) if isinstance((prev_rul_all or {}).get(vid, {}), dict) else {}
        sort_col = _pick_axis_col(g)

        if sort_col == '__session_idx':
            g['__session_idx'] = np.arange(len(g), dtype=float)
            g = g.sort_values('__session_idx')
            axis_name = 'session_idx'
            axis_vals = g['__session_idx'].values.astype(float)
            axis_zero = 0.0
        elif sort_col == 'start_utc':
            g = g.sort_values('start_utc')
            t = _finite_series(g['start_utc'])
            axis_zero = t.min() if t.notna().any() else 0.0
            axis_name = 'elapsed_days'
            axis_vals = ((t - axis_zero) / 86400.0).values.astype(float)
        else:
            g = g.sort_values('hrlfc_mid')
            axis_name = 'hrlfc_mid'
            axis_vals = _finite_series(g['hrlfc_mid']).values.astype(float)
            axis_zero = np.nanmin(axis_vals) if np.isfinite(axis_vals).any() else 0.0

        raw_v = df_raw[df_raw['vehicle_id'] == vid]

        utc_vals = _finite_series(raw_v['_utc_num']) if '_utc_num' in raw_v.columns else pd.Series(dtype=float)
        utc_span = (utc_vals.max() - utc_vals.min()) if utc_vals.notna().sum() >= 2 else np.nan
        days_span = (utc_span / 86400) if np.isfinite(utc_span) else np.nan
        last_utc = utc_vals.max() if utc_vals.notna().sum() >= 1 else np.nan
        last_dt_ist = _utc_to_ist_datetime(last_utc) if np.isfinite(last_utc) else pd.NaT

        dist_vals = _finite_series(raw_v['totalDistance']) if 'totalDistance' in raw_v.columns else pd.Series(dtype=float)
        km_run_till_date = (dist_vals.max() - dist_vals.min()) if dist_vals.notna().sum() >= 2 else np.nan
        km_per_day_hist = (km_run_till_date / days_span) if (np.isfinite(km_run_till_date) and np.isfinite(days_span) and days_span > 0) else np.nan
        axis_span = (np.nanmax(axis_vals) - np.nanmin(axis_vals)) if np.isfinite(axis_vals).sum() >= 2 else np.nan

        if axis_name == 'elapsed_days':
            htd = 1.0
        elif axis_name == 'session_idx':
            htd = (days_span / max(len(g) - 1, 1)) if np.isfinite(days_span) and len(g) > 1 else 0.07
        else:
            htd = (days_span / axis_span) if (np.isfinite(days_span) and np.isfinite(axis_span) and axis_span > 0) else 0.07

        # Estimate throughput per axis unit for energy-to-EOL reporting.
        ah_per_axis = np.nan
        if 'ah_total' in g.columns:
            ah_s = _finite_series(g['ah_total']).values.astype(float)
            mask_aa = np.isfinite(ah_s) & np.isfinite(axis_vals)
            if axis_name == 'session_idx':
                ah_per_axis = np.nanmedian(ah_s[mask_aa]) if np.any(mask_aa) else np.nan
            else:
                ax_min = np.nanmin(axis_vals[mask_aa]) if np.any(mask_aa) else np.nan
                ax_max = np.nanmax(axis_vals[mask_aa]) if np.any(mask_aa) else np.nan
                ax_span = (ax_max - ax_min) if np.isfinite(ax_min) and np.isfinite(ax_max) else np.nan
                if np.isfinite(ax_span) and ax_span > 0:
                    ah_per_axis = np.nansum(ah_s[mask_aa]) / ax_span
                elif np.any(mask_aa):
                    ah_per_axis = np.nanmedian(ah_s[mask_aa])

        if vid in lstm_results:
            lr = lstm_results[vid]
            lb = lr['lookback']
            soh_seq = lr['soh_pred']
            if axis_name == 'elapsed_days':
                hrlfc_seq = (np.asarray(lr['hrlfc_seq'][lb:], dtype=float) - axis_zero) / 86400.0
            elif axis_name == 'session_idx':
                hrlfc_seq = np.arange(len(soh_seq), dtype=float)
            else:
                hrlfc_seq = np.asarray(lr['hrlfc_seq'][lb:], dtype=float)
        else:
            hrlfc_seq = np.asarray(axis_vals, dtype=float)
            soh_seq = _finite_series(g['soh_xgb']).values

        q_base_ah = np.nan
        if 'q_rated_used_for_soh_ah' in g.columns:
            q_base_ah = _finite_series(g['q_rated_used_for_soh_ah']).median()
        if not np.isfinite(q_base_ah) and 'q_init_first_cycles_ah' in g.columns:
            q_base_ah = _finite_series(g['q_init_first_cycles_ah']).median()
        if not np.isfinite(q_base_ah) and 'q_rated' in g.columns:
            q_base_ah = _finite_series(g['q_rated']).median()
        q_nom = _finite_series(g['q_nominal_ah']).median() if 'q_nominal_ah' in g.columns else np.nan
        v_nom = np.nan
        for vc in ['pack_v_mean', 'pack_v_max', 'pack_v_min']:
            if vc in g.columns:
                v_nom = _finite_series(g[vc]).median()
                if np.isfinite(v_nom):
                    break
        kwh_per_axis = ah_per_axis * v_nom / 1000.0 if np.isfinite(ah_per_axis) and np.isfinite(v_nom) else np.nan
        kwh_per_day_hist = (kwh_per_axis / htd) if np.isfinite(kwh_per_axis) and np.isfinite(htd) and htd > 0 else np.nan
        ah80_model = 0.8 * q_base_ah if np.isfinite(q_base_ah) else np.nan
        ah80_nom = 0.8 * q_nom if np.isfinite(q_nom) else np.nan
        e80_model_kwh = ah80_model * v_nom / 1000.0 if np.isfinite(ah80_model) and np.isfinite(v_nom) else np.nan
        e80_nom_kwh = ah80_nom * v_nom / 1000.0 if np.isfinite(ah80_nom) and np.isfinite(v_nom) else np.nan
        charging_events_count = int(len(g))
        partial_charging_events_count = np.nan
        if 'delta_soc_pct' in g.columns:
            ds = _finite_series(g['delta_soc_pct']).values.astype(float)
            # Treat events with SOC gain < 80% as partial charges.
            partial_charging_events_count = int(np.nansum(ds < 80.0)) if len(ds) else 0
        total_ah_throughput = np.nan
        if 'ah_total' in g.columns:
            total_ah_throughput = float(np.nansum(_finite_series(g['ah_total']).values.astype(float)))

        rul = extrapolate_rul(hrlfc_seq, soh_seq, htd)
        soh_now = rul.get('soh_now', np.nan)
        init_cap_ah = q_base_ah if np.isfinite(q_base_ah) else np.nan
        current_cap_ah = (init_cap_ah * soh_now / 100.0) if np.isfinite(init_cap_ah) and np.isfinite(soh_now) else np.nan
        init_cap_kwh = (init_cap_ah * v_nom / 1000.0) if np.isfinite(init_cap_ah) and np.isfinite(v_nom) else np.nan
        current_cap_kwh = (current_cap_ah * v_nom / 1000.0) if np.isfinite(current_cap_ah) and np.isfinite(v_nom) else np.nan
        eol_cap_ah = (0.8 * init_cap_ah) if np.isfinite(init_cap_ah) else np.nan
        equivalent_full_cycles = (total_ah_throughput / init_cap_ah) if np.isfinite(total_ah_throughput) and np.isfinite(init_cap_ah) and init_cap_ah > 0 else np.nan
        pack_cfg_guess = str(g['pack_config_guess'].iloc[0]) if ('pack_config_guess' in g.columns and len(g) > 0) else str(prev_rr.get('pack_config_guess', 'unknown'))
        pack_cfg_conf = _finite_series(g['pack_score_confidence']).median() if 'pack_score_confidence' in g.columns else pd.to_numeric(pd.Series([prev_rr.get('pack_score_confidence', np.nan)]), errors='coerce').iloc[0]
        prev_cfg_guess = str(prev_rr.get('pack_config_guess', 'unknown'))
        prev_epoch = int(pd.to_numeric(pd.Series([prev_rr.get('config_epoch_id', 0)]), errors='coerce').fillna(0).iloc[0])
        cfg_changed_strong = (
            (prev_cfg_guess != 'unknown') and
            (pack_cfg_guess != 'unknown') and
            (pack_cfg_guess != prev_cfg_guess) and
            np.isfinite(pack_cfg_conf) and
            (pack_cfg_conf >= float(PACK_CONFIG_SWITCH_MIN_CONF))
        )
        config_epoch_id = prev_epoch + 1 if cfg_changed_strong else prev_epoch
        equivalent_full_cycles_epoch = 0.0 if cfg_changed_strong else equivalent_full_cycles

        r10 = rul.get('rul_hrlfc_p10', np.nan)
        r50 = rul.get('rul_hrlfc_p50', np.nan)
        r90 = rul.get('rul_hrlfc_p90', np.nan)
        d10 = rul.get('rul_days_p10', np.nan)
        d50 = rul.get('rul_days_p50', np.nan)
        d90 = rul.get('rul_days_p90', np.nan)
        e_to_eol_p10 = (r10 * kwh_per_axis) if np.isfinite(r10) and np.isfinite(kwh_per_axis) else (np.inf if np.isinf(r10) and np.isfinite(kwh_per_axis) else np.nan)
        e_to_eol_p50 = (r50 * kwh_per_axis) if np.isfinite(r50) and np.isfinite(kwh_per_axis) else (np.inf if np.isinf(r50) and np.isfinite(kwh_per_axis) else np.nan)
        e_to_eol_p90 = (r90 * kwh_per_axis) if np.isfinite(r90) and np.isfinite(kwh_per_axis) else (np.inf if np.isinf(r90) and np.isfinite(kwh_per_axis) else np.nan)
        km_to_eol_p10 = (d10 * km_per_day_hist) if np.isfinite(d10) and np.isfinite(km_per_day_hist) else (np.inf if np.isinf(d10) and np.isfinite(km_per_day_hist) else np.nan)
        km_to_eol_p50 = (d50 * km_per_day_hist) if np.isfinite(d50) and np.isfinite(km_per_day_hist) else (np.inf if np.isinf(d50) and np.isfinite(km_per_day_hist) else np.nan)
        km_to_eol_p90 = (d90 * km_per_day_hist) if np.isfinite(d90) and np.isfinite(km_per_day_hist) else (np.inf if np.isinf(d90) and np.isfinite(km_per_day_hist) else np.nan)
        eol_date_p10 = _safe_project_date(last_dt_ist, d10, horizon_days=REPORT_RUL_CAP_DAYS)
        eol_date_p50 = _safe_project_date(last_dt_ist, d50, horizon_days=REPORT_RUL_CAP_DAYS)
        eol_date_p90 = _safe_project_date(last_dt_ist, d90, horizon_days=REPORT_RUL_CAP_DAYS)
        rul.update({
            'vehicle_id': vid,
            'hrlfc_to_days': htd,
            'n_sessions': len(g),
            'data_span_days': days_span if np.isfinite(days_span) else np.nan,
            'axis_used': axis_name,
            'q_base_for_soh_ah': q_base_ah,
            'q_ref_cal_ah': q_base_ah,
            'q_init_first_cycles_ah': _finite_series(g['q_init_first_cycles_ah']).median() if 'q_init_first_cycles_ah' in g.columns else np.nan,
            'q_settle_window_ah': _finite_series(g['q_settle_window_ah']).median() if 'q_settle_window_ah' in g.columns else np.nan,
            'q_base_source': str(g['q_base_source'].iloc[0]) if ('q_base_source' in g.columns and len(g) > 0) else '',
            'q_nominal_ah': q_nom,
            'pack_config_guess': pack_cfg_guess,
            'pack_score_confidence': pack_cfg_conf,
            'pack_options_source': str(g['pack_options_source'].iloc[0]) if ('pack_options_source' in g.columns and len(g) > 0) else str(prev_rr.get('pack_options_source', '')),
            'pack_series_cells': _finite_series(g['pack_series_cells']).median() if 'pack_series_cells' in g.columns else pd.to_numeric(pd.Series([prev_rr.get('pack_series_cells', np.nan)]), errors='coerce').iloc[0],
            'pack_parallel_guess': _finite_series(g['pack_parallel_guess']).median() if 'pack_parallel_guess' in g.columns else pd.to_numeric(pd.Series([prev_rr.get('pack_parallel_guess', np.nan)]), errors='coerce').iloc[0],
            'pack_voltage_ref_v': _finite_series(g['pack_voltage_ref_v']).median() if 'pack_voltage_ref_v' in g.columns else pd.to_numeric(pd.Series([prev_rr.get('pack_voltage_ref_v', np.nan)]), errors='coerce').iloc[0],
            'pack_q_data_median_ah': _finite_series(g['pack_q_data_median_ah']).median() if 'pack_q_data_median_ah' in g.columns else pd.to_numeric(pd.Series([prev_rr.get('pack_q_data_median_ah', np.nan)]), errors='coerce').iloc[0],
            'pack_q_data_count': _finite_series(g['pack_q_data_count']).median() if 'pack_q_data_count' in g.columns else pd.to_numeric(pd.Series([prev_rr.get('pack_q_data_count', np.nan)]), errors='coerce').iloc[0],
            'pack_config_changed_strong': bool(cfg_changed_strong),
            'config_epoch_id': int(config_epoch_id),
            'init_capacity_ah': init_cap_ah,
            'init_capacity_kwh': init_cap_kwh,
            'current_capacity_ah': current_cap_ah,
            'current_capacity_kwh': current_cap_kwh,
            'eol_capacity_ah': eol_cap_ah,
            'charging_events_count': charging_events_count,
            'partial_charging_events_count': partial_charging_events_count,
            'total_ah_throughput': total_ah_throughput,
            'equivalent_full_cycles': equivalent_full_cycles,
            'equivalent_full_cycles_epoch': equivalent_full_cycles_epoch,
            'km_run_till_date': km_run_till_date,
            'km_per_day_hist': km_per_day_hist,
            'km_to_eol_p10': km_to_eol_p10,
            'km_to_eol_p50': km_to_eol_p50,
            'km_to_eol_p90': km_to_eol_p90,
            'last_seen_datetime_ist': last_dt_ist,
            'eol_date_p10_ist': eol_date_p10,
            'eol_date_p50_ist': eol_date_p50,
            'eol_date_p90_ist': eol_date_p90,
            'v_nom_v': v_nom,
            'ah_per_axis': ah_per_axis,
            'kwh_per_axis': kwh_per_axis,
            'kwh_per_day_hist': kwh_per_day_hist,
            'energy_to_eol_kwh_p10': e_to_eol_p10,
            'energy_to_eol_kwh_p50': e_to_eol_p50,
            'energy_to_eol_kwh_p90': e_to_eol_p90,
            'deliverable_ah_at_80_model': ah80_model,
            'deliverable_kwh_at_80_model': e80_model_kwh,
            'deliverable_ah_at_80_nominal': ah80_nom,
            'deliverable_kwh_at_80_nominal': e80_nom_kwh,
            'bms_correction_factor': float(_finite_series(g['bms_correction_factor']).median()) if 'bms_correction_factor' in g.columns else 1.0,
        })
        rul_all[vid] = rul

        print(f"\nIMEI: {_imei_text(vid)}")
        print(f"Battery Health  : {rul.get('soh_now', np.nan):.2f}%")
        print(f"Initial capacity: {_fmt_ah(rul.get('init_capacity_ah', np.nan))} ({_fmt_kwh(rul.get('init_capacity_kwh', np.nan))})")
        print(f"Capacity today  : {_fmt_ah(rul.get('current_capacity_ah', np.nan))} ({_fmt_kwh(rul.get('current_capacity_kwh', np.nan))})")
        print("")
        print(
            f"Expected life remaining : "
            f"{_life_remaining_text(rul.get('rul_days_p50', np.nan), rul.get('rul_days_p10', np.nan), rul.get('rul_days_p90', np.nan))}"
        )
        print(
            "Est. end-of-life date   : "
            f"worst={_fmt_eol_date(rul.get('eol_date_p10_ist', pd.NaT), rul.get('rul_days_p10', np.nan))} | "
            f"likely={_fmt_eol_date(rul.get('eol_date_p50_ist', pd.NaT), rul.get('rul_days_p50', np.nan))} | "
            f"best={_fmt_eol_date(rul.get('eol_date_p90_ist', pd.NaT), rul.get('rul_days_p90', np.nan))}"
        )
        print(f"Distance covered        : {_fmt_km(rul.get('km_run_till_date', np.nan))}")
        print(
            "Distance remaining      : "
            f"worst={_fmt_km(rul.get('km_to_eol_p10', np.nan), approx=True)} | "
            f"likely={_fmt_km(rul.get('km_to_eol_p50', np.nan), approx=True)} | "
            f"best={_fmt_km(rul.get('km_to_eol_p90', np.nan), approx=True)}"
        )

    return rul_all

# ------------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------------
def _smooth_line_until_soh_eol_for_plot(x, y, eol=SOH_EOL, window=9):
    """
    Display-only smoothing: smooth the visible curve up to the first EOL crossing.
    Does not alter any model outputs or saved numeric results.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    out = y_arr.copy()

    if x_arr.shape != y_arr.shape:
        return out

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(finite.sum()) < 5:
        return out

    idx = np.where(finite)[0]
    x_f = x_arr[idx]
    y_f = y_arr[idx]

    order = np.argsort(x_f, kind='stable')
    idx = idx[order]
    y_f = y_f[order]

    hit = np.where(y_f <= float(eol))[0]
    end_pos = int(hit[0]) if len(hit) else (len(y_f) - 1)
    if end_pos < 4:
        return out

    seg = y_f[:end_pos + 1]
    w = min(int(window), len(seg))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return out

    seg_sm = (
        pd.Series(seg)
        .rolling(window=w, min_periods=1, center=True)
        .mean()
        .values
        .astype(float)
    )
    out[idx[:end_pos + 1]] = seg_sm
    return out


def _ease_curve_to_soh_eol_for_plot(y, eol=SOH_EOL):
    """
    Display-only easing for extrapolation lines before the first EOL crossing.
    """
    y_arr = np.asarray(y, dtype=float)
    out = y_arr.copy()
    finite = np.isfinite(y_arr)
    if int(finite.sum()) < 4:
        return out

    idx = np.where(finite)[0]
    y_f = y_arr[idx]
    hit = np.where(y_f <= float(eol))[0]
    if len(hit) == 0:
        return out

    end_pos = int(hit[0])
    if end_pos < 2:
        return out

    y0 = float(y_f[0])
    if y0 <= float(eol):
        return out

    t = np.linspace(0.0, 1.0, end_pos + 1)
    smoothstep = t * t * (3.0 - 2.0 * t)
    eased = y0 + (float(eol) - y0) * smoothstep
    out[idx[:end_pos + 1]] = eased
    return out


def _prepare_smooth_plot_curve(x, y, eol=SOH_EOL, window=9, points=240):
    """
    Build a dense, visually smooth monotone-like curve for plotting only.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        return x_arr, y_arr

    y_sm = _smooth_line_until_soh_eol_for_plot(x_arr, y_arr, eol=eol, window=window)
    finite = np.isfinite(x_arr) & np.isfinite(y_sm)
    if int(finite.sum()) < 4:
        return x_arr, y_sm

    x_f = x_arr[finite]
    y_f = y_sm[finite]
    order = np.argsort(x_f, kind='stable')
    x_f = x_f[order]
    y_f = y_f[order]

    ux, inv = np.unique(x_f, return_inverse=True)
    if len(ux) < 4:
        return x_f, y_f
    y_u = np.zeros(len(ux), dtype=float)
    for i in range(len(ux)):
        y_u[i] = float(np.nanmean(y_f[inv == i]))

    # Extra display-only shaping up to first EOL hit:
    # turn stair-like plateaus into a smooth declining visual trend.
    hit = np.where(y_u <= float(eol))[0]
    end_pos = int(hit[0]) if len(hit) else (len(y_u) - 1)
    if end_pos >= 5:
        x_seg = ux[:end_pos + 1]
        y_seg = y_u[:end_pos + 1]

        # Smoothing spline (not used in model computation, plot only).
        try:
            k = int(min(3, len(x_seg) - 1))
            s = max(1e-6, 0.08 * len(x_seg))
            spl = UnivariateSpline(x_seg, y_seg, k=k, s=s)
            y_seg_sm = np.asarray(spl(x_seg), dtype=float)
        except Exception:
            y_seg_sm = (
                pd.Series(y_seg)
                .rolling(window=min(11, len(y_seg) if len(y_seg) % 2 == 1 else max(3, len(y_seg) - 1)),
                         min_periods=1, center=True)
                .mean()
                .values
                .astype(float)
            )

        # Prevent long flats by applying a tiny visual decline floor.
        y0 = float(y_seg_sm[0])
        min_total_drop = max(0.6, 0.12 * np.log1p(len(y_seg_sm)))
        drift = y0 - np.linspace(0.0, min_total_drop, len(y_seg_sm))
        y_seg_sm = np.minimum(y_seg_sm, drift)

        y_u[:end_pos + 1] = y_seg_sm

    n_dense = max(int(points), len(ux) * 10)
    x_dense = np.linspace(float(ux[0]), float(ux[-1]), n_dense)
    try:
        interp = PchipInterpolator(ux, y_u, extrapolate=False)
        y_dense = np.asarray(interp(x_dense), dtype=float)
    except Exception:
        y_dense = np.interp(x_dense, ux, y_u)

    # Final anti-stair pass on dense samples (plot only).
    if len(y_dense) >= 15:
        k = min(31, (len(y_dense) // 2) * 2 - 1)
        if k >= 5:
            win = np.hanning(k)
            if np.allclose(win.sum(), 0.0):
                win = np.ones(k, dtype=float)
            win = win / win.sum()
            pad = k // 2
            y_pad = np.pad(y_dense, (pad, pad), mode='edge')
            y_dense = np.convolve(y_pad, win, mode='valid')
    return x_dense, y_dense


def plot_results(xgb_results, lstm_results, rul_all, save_path):
    """
    Save one figure per vehicle (3 panels each) and a small summary message.
    """
    out_root = Path(save_path)
    out_dir = out_root.with_suffix('')
    out_dir = out_dir.parent / f"{out_root.stem}_by_vehicle"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for vid, res in xgb_results.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        g = res['sessions'].copy()
        sort_col = _pick_axis_col(g)

        if sort_col == '__session_idx':
            g['__session_idx'] = np.arange(len(g), dtype=float)
            g = g.sort_values('__session_idx')
            x = g['__session_idx'].values.astype(float)
            x_label = 'session_idx'
            x_zero = 0.0
        elif sort_col == 'start_utc':
            g = g.sort_values('start_utc')
            t = _finite_series(g['start_utc'])
            x_zero = t.min() if t.notna().any() else 0.0
            x = ((t - x_zero) / 86400.0).values.astype(float)
            x_label = 'elapsed_days'
        else:
            g = g.sort_values('hrlfc_mid')
            x = _finite_series(g['hrlfc_mid']).values.astype(float)
            x_label = 'hrlfc_mid'
            x_zero = np.nanmin(x) if np.isfinite(x).any() else 0.0

        y_label = _finite_series(g['soh_label']).values
        y_smooth = _finite_series(g['soh_smooth']).values
        y_xgb = _finite_series(g['soh_xgb']).values
        y_smooth_plot = _smooth_line_until_soh_eol_for_plot(x, y_smooth, eol=SOH_EOL, window=9)
        y_xgb_plot = _smooth_line_until_soh_eol_for_plot(x, y_xgb, eol=SOH_EOL, window=9)
        w = _finite_series(g['weight']).values if 'weight' in g.columns else np.ones(len(g), dtype=float)
        wmax = np.nanmax(w) if np.isfinite(w).any() and np.nanmax(w) > 0 else 1.0

        ax = axes[0]
        mask_label = np.isfinite(x) & np.isfinite(y_label)
        if np.any(mask_label):
            ax.scatter(
                x[mask_label], y_label[mask_label],
                s=(w[mask_label] / wmax * 60 + 10),
                alpha=0.5, label='Pseudo-SOH', color='steelblue'
            )
        mask_smooth = np.isfinite(x) & np.isfinite(y_smooth_plot)
        if np.any(mask_smooth):
            xs, ys = _prepare_smooth_plot_curve(x[mask_smooth], y_smooth_plot[mask_smooth], eol=SOH_EOL, window=9)
            m = np.isfinite(xs) & np.isfinite(ys)
            if np.any(m):
                ax.plot(xs[m], ys[m], 'b--', alpha=0.6, label='Smoothed')
        mask_xgb = np.isfinite(x) & np.isfinite(y_xgb_plot)
        if np.any(mask_xgb):
            xs, ys = _prepare_smooth_plot_curve(x[mask_xgb], y_xgb_plot[mask_xgb], eol=SOH_EOL, window=9)
            m = np.isfinite(xs) & np.isfinite(ys)
            if np.any(m):
                ax.plot(xs[m], ys[m], 'r-', lw=2, label='XGBoost')

        ax.axhline(SOH_EOL, color='red', linestyle=':', alpha=0.5, label='EOL 80%')
        ax.set_title(f"{vid}\nXGBoost SOH", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel('SOH (%)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(70, 105)

        ax = axes[1]
        if vid in lstm_results:
            lr = lstm_results[vid]
            lb = lr['lookback']

            x_all_raw = np.asarray(lr['hrlfc_seq'], dtype=float)
            x_pred_raw = np.asarray(lr['hrlfc_seq'][lb:], dtype=float)
            if x_label == 'elapsed_days':
                x_all = (x_all_raw - x_zero) / 86400.0
                x_pred = (x_pred_raw - x_zero) / 86400.0
            elif x_label == 'session_idx':
                x_all = np.arange(len(lr['soh_seq']), dtype=float)
                x_pred = np.arange(len(lr['soh_pred']), dtype=float)
            else:
                x_all = x_all_raw
                x_pred = x_pred_raw

            y_all = np.asarray(lr['soh_seq'], dtype=float)
            y_pred = np.asarray(lr['soh_pred'], dtype=float)
            y_all_plot = _smooth_line_until_soh_eol_for_plot(x_all, y_all, eol=SOH_EOL, window=9)
            y_pred_plot = _smooth_line_until_soh_eol_for_plot(x_pred, y_pred, eol=SOH_EOL, window=9)

            mask_all = np.isfinite(x_all) & np.isfinite(y_all_plot)
            if np.any(mask_all):
                xs, ys = _prepare_smooth_plot_curve(x_all[mask_all], y_all_plot[mask_all], eol=SOH_EOL, window=9)
                m = np.isfinite(xs) & np.isfinite(ys)
                if np.any(m):
                    ax.plot(xs[m], ys[m], 'b--', alpha=0.5, label='XGBoost input')

            mask_pred = np.isfinite(x_pred) & np.isfinite(y_pred_plot)
            if np.any(mask_pred):
                xs, ys = _prepare_smooth_plot_curve(x_pred[mask_pred], y_pred_plot[mask_pred], eol=SOH_EOL, window=9)
                m = np.isfinite(xs) & np.isfinite(ys)
                if np.any(m):
                    ax.plot(xs[m], ys[m], 'g-', lw=2, label='LSTM trajectory')

            rul = rul_all.get(vid, {})
            if (rul.get('phase2_slope', 0) < 0 and
                np.isfinite(rul.get('hrlfc_now', np.nan)) and
                np.isfinite(rul.get('rul_hrlfc_p50', np.nan))):
                h_now = rul['hrlfc_now']
                s = rul['phase2_slope']
                b = rul['soh_now'] - s * h_now
                x_ext = np.linspace(h_now, h_now + rul.get('rul_hrlfc_p50', 0) * 1.2, 100)
                y_ext = s * x_ext + b
                y_ext_plot = _ease_curve_to_soh_eol_for_plot(y_ext, eol=SOH_EOL)
                ax.plot(x_ext, y_ext_plot, 'r--', lw=1.5, alpha=0.7, label='Extrapolation')
                ax.axvline(h_now + rul.get('rul_hrlfc_p50', 0), color='red', linestyle=':', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for LSTM',
                    ha='center', va='center', transform=ax.transAxes)

        ax.axhline(SOH_EOL, color='red', linestyle=':', alpha=0.5)
        ax.set_title(f"{vid}\nLSTM + RUL", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel('SOH (%)')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(70, 105)

        ax = axes[2]
        res['feature_importance'].head(8).plot(kind='barh', ax=ax, color='teal', alpha=0.7)
        ax.set_title(f"{vid}\nFeature importance", fontsize=10)
        ax.set_xlabel('Importance')
        ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        safe_vid = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(vid))
        out_file = out_dir / f"{safe_vid}.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_file)

    print(f"\n  Saved {len(saved)} per-vehicle plots -> {out_dir}")


def _risk_label(soh_now, rul_p50_days):
    if not np.isfinite(soh_now):
        return "Unknown", "gray"
    if (soh_now < 82) or (np.isfinite(rul_p50_days) and rul_p50_days < 180):
        return "High risk", "crimson"
    if (soh_now < 88) or (np.isfinite(rul_p50_days) and rul_p50_days < 365):
        return "Watch", "darkorange"
    return "Healthy", "seagreen"


def plot_customer_views(xgb_results, lstm_results, rul_all, replacement_events, save_path):
    """
    Customer-friendly visuals:
      1) fleet dashboard summary
      2) one-page health card per vehicle
    """
    out_root = Path(save_path)
    dash_path = out_root.parent / f"{out_root.stem}_customer_dashboard.png"
    cards_dir = out_root.parent / f"{out_root.stem}_customer_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vid, r in rul_all.items():
        rows.append({
            'vehicle_id': vid,
            'soh_now': r.get('soh_now', np.nan),
            'rul_p50_days': r.get('rul_days_p50', np.nan),
            'rul_p10_days': r.get('rul_days_p10', np.nan),
            'rul_p90_days': r.get('rul_days_p90', np.nan),
            'last_seen_datetime_ist': r.get('last_seen_datetime_ist', pd.NaT),
            'eol_date_p50_ist': r.get('eol_date_p50_ist', pd.NaT),
            'init_kwh': r.get('init_capacity_kwh', np.nan),
            'e_p10': r.get('energy_to_eol_kwh_p10', np.nan),
            'e_p50': r.get('energy_to_eol_kwh_p50', np.nan),
            'e_p90': r.get('energy_to_eol_kwh_p90', np.nan),
            'current_kwh': r.get('current_capacity_kwh', np.nan),
            'deliver80_kwh': r.get('deliverable_kwh_at_80_model', np.nan),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values('vehicle_id').reset_index(drop=True)

    # -----------------------------
    # Fleet Dashboard (manager view)
    # -----------------------------
    from matplotlib import dates as mdates

    def _mgr_status(soh):
        if not np.isfinite(soh):
            return "Unknown", "lightgray", "gray"
        if soh > 85:
            return "Green", "#D9F2D9", "seagreen"
        if soh >= 75:
            return "Amber", "#FFF1CC", "darkorange"
        return "Red", "#F9D5D3", "crimson"

    df['soh_now'] = pd.to_numeric(df['soh_now'], errors='coerce')
    df['rul_p50_days'] = pd.to_numeric(df['rul_p50_days'], errors='coerce')
    df['rul_p10_days'] = pd.to_numeric(df['rul_p10_days'], errors='coerce')
    df['rul_p90_days'] = pd.to_numeric(df['rul_p90_days'], errors='coerce')
    df['months_to_eol_p50'] = df['rul_p50_days'] / 30.44
    df['months_to_eol_p10'] = df['rul_p10_days'] / 30.44
    df['months_to_eol_p90'] = df['rul_p90_days'] / 30.44
    df['status'] = df['soh_now'].apply(lambda v: _mgr_status(v)[0])
    df['status_bg'] = df['soh_now'].apply(lambda v: _mgr_status(v)[1])
    df['status_fg'] = df['soh_now'].apply(lambda v: _mgr_status(v)[2])

    last_seen = pd.to_datetime(df['last_seen_datetime_ist'], errors='coerce')
    base_ts = last_seen.max() if last_seen.notna().any() else pd.Timestamp.now()
    if pd.isna(base_ts):
        base_ts = pd.Timestamp.now()

    eol_ts = pd.to_datetime(df['eol_date_p50_ist'], errors='coerce')
    eol_fallback = [(_safe_project_date(base_ts, d) if np.isfinite(d) else pd.NaT) for d in df['rul_p50_days'].tolist()]
    eol_final = pd.Series(eol_ts).where(pd.Series(eol_ts).notna(), pd.Series(eol_fallback))
    df['eol_date_p50'] = pd.to_datetime(eol_final, errors='coerce')

    due_12m = df['months_to_eol_p50'].apply(lambda v: np.isfinite(v) and (v <= 12))
    repl_12_n = int(due_12m.fillna(False).sum())
    pack_cost_low_lakh = 8.0
    pack_cost_high_lakh = 12.0

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.65, 2.0, 2.0], width_ratios=[1.25, 1.75], hspace=0.30, wspace=0.25)

    # KPI banner
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('off')
    fleet_n = len(df)
    avg_soh = np.nanmean(df['soh_now'].values.astype(float)) if fleet_n else np.nan
    green_n = int((df['status'] == 'Green').sum())
    amber_n = int((df['status'] == 'Amber').sum())
    red_n = int((df['status'] == 'Red').sum())
    budget_low = repl_12_n * pack_cost_low_lakh
    budget_high = repl_12_n * pack_cost_high_lakh
    ax_kpi.text(0.01, 0.80, "Fleet At A Glance (Manager View)", fontsize=18, fontweight='bold')
    ax_kpi.text(
        0.01, 0.50,
        f"Vehicles: {fleet_n}   |   Avg SOH: {avg_soh:.1f}%   |   Traffic: Green={green_n}, Amber={amber_n}, Red={red_n}",
        fontsize=12.5
    )
    ax_kpi.text(
        0.01, 0.22,
        f"Replacements due in next 12 months: {repl_12_n}   |   Estimated budget: ₹{budget_low:.0f}L–₹{budget_high:.0f}L",
        fontsize=12.5, color='crimson' if repl_12_n > 0 else 'dimgray'
    )

    # SOH traffic-light table (all vehicles)
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis('off')
    tbl_df = df[['vehicle_id', 'soh_now', 'status', 'months_to_eol_p50']].copy()
    tbl_df = tbl_df.sort_values(['soh_now', 'vehicle_id'], na_position='last').reset_index(drop=True)
    tbl_df['SOH %'] = tbl_df['soh_now'].apply(lambda v: f"{v:.1f}%" if np.isfinite(v) else "NA")
    tbl_df['Months to EOL'] = tbl_df['months_to_eol_p50'].apply(lambda v: f"{v:.1f}" if np.isfinite(v) else "NA")
    render_tbl = tbl_df[['vehicle_id', 'SOH %', 'status', 'Months to EOL']]
    render_tbl.columns = ['Vehicle', 'SOH %', 'Status', 'Months to EOL']
    table = ax_tbl.table(
        cellText=render_tbl.values.tolist(),
        colLabels=render_tbl.columns.tolist(),
        loc='center',
        cellLoc='left',
        colLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    y_scale = 1.25 if len(render_tbl) <= 12 else max(0.72, 15.0 / max(len(render_tbl), 1))
    table.scale(1.05, y_scale)
    for j in range(len(render_tbl.columns)):
        table[(0, j)].set_facecolor('#E8EEF7')
        table[(0, j)].set_text_props(weight='bold')
    for i in range(len(render_tbl)):
        status = str(render_tbl.iloc[i]['Status'])
        bg = '#D9F2D9' if status == 'Green' else ('#FFF1CC' if status == 'Amber' else ('#F9D5D3' if status == 'Red' else '#F2F2F2'))
        fg = 'seagreen' if status == 'Green' else ('darkorange' if status == 'Amber' else ('crimson' if status == 'Red' else 'gray'))
        for j in range(len(render_tbl.columns)):
            table[(i + 1, j)].set_facecolor(bg if j == 2 else 'white')
        table[(i + 1, 2)].set_text_props(color=fg, weight='bold')
    ax_tbl.set_title("SOH Traffic-Light Table (All Vehicles)", fontsize=12, pad=8)

    # Months-to-EOL ranked chart (replaces energy uncertainty)
    ax_m = fig.add_subplot(gs[1, 1])
    show_m = df[np.isfinite(df['months_to_eol_p50'])].copy()
    if len(show_m) > 0:
        show_m = show_m.sort_values('months_to_eol_p50').head(20)
        y = np.arange(len(show_m))
        colors = show_m['status_fg'].tolist()
        m50 = show_m['months_to_eol_p50'].values.astype(float)
        m10 = show_m['months_to_eol_p10'].values.astype(float)
        m90 = show_m['months_to_eol_p90'].values.astype(float)
        xerr_low = np.where(np.isfinite(m10), np.maximum(0.0, m50 - m10), 0.0)
        xerr_high = np.where(np.isfinite(m90), np.maximum(0.0, m90 - m50), 0.0)
        ax_m.barh(y, m50, color=colors, alpha=0.85)
        ax_m.errorbar(m50, y, xerr=[xerr_low, xerr_high], fmt='none', ecolor='black', elinewidth=1.0, alpha=0.55)
        ax_m.set_yticks(y)
        ax_m.set_yticklabels(show_m['vehicle_id'])
        ax_m.invert_yaxis()
        ax_m.set_xlabel("Estimated Months to EOL (P50)")
        ax_m.set_title("Months-to-EOL (Ranked) with Uncertainty")
        ax_m.grid(alpha=0.25, axis='x')
        ax_m.axvline(12, color='crimson', linestyle=':', alpha=0.7, label='12-month planning line')
        ax_m.legend(fontsize=8)
    else:
        ax_m.axis('off')
        ax_m.text(0.03, 0.50, "Months-to-EOL unavailable (insufficient finite RUL).", fontsize=12)

    # Replacement timeline (12-month budget plan)
    ax_g = fig.add_subplot(gs[2, 0])
    horizon_end = base_ts + pd.Timedelta(days=365)
    gantt_df = df[df['eol_date_p50'].notna()].copy()
    gantt_df = gantt_df[gantt_df['eol_date_p50'] <= horizon_end]
    gantt_df = gantt_df.sort_values('eol_date_p50')
    if len(gantt_df) > 0:
        y = np.arange(len(gantt_df))
        start_num = mdates.date2num(base_ts.to_pydatetime())
        end_num = mdates.date2num(gantt_df['eol_date_p50'].dt.to_pydatetime())
        widths = np.maximum(1.0, end_num - start_num)
        ax_g.barh(y, widths, left=start_num, color=gantt_df['status_fg'].tolist(), alpha=0.55)
        ax_g.set_yticks(y)
        ax_g.set_yticklabels(gantt_df['vehicle_id'])
        ax_g.invert_yaxis()
        ax_g.xaxis_date()
        ax_g.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax_g.axvline(start_num, color='black', linestyle='--', alpha=0.8)
        ax_g.set_title("12-Month Replacement Plan (P50 EOL Date)")
        ax_g.set_xlabel("Target replacement timeline")
        ax_g.grid(alpha=0.25, axis='x')
        for i, (_, row) in enumerate(gantt_df.iterrows()):
            q = pd.Timestamp(row['eol_date_p50']).quarter
            ylab = f"Q{q} | ₹{pack_cost_low_lakh:.0f}–{pack_cost_high_lakh:.0f}L"
            ax_g.text(mdates.date2num(pd.Timestamp(row['eol_date_p50']).to_pydatetime()) + 4, i, ylab, va='center', fontsize=7.5, color='dimgray')
    else:
        ax_g.axis('off')
        ax_g.text(0.03, 0.55, "No vehicles projected to reach EOL within 12 months.", fontsize=12)
        ax_g.text(0.03, 0.35, "Replacement budget not urgent for next 4 quarters.", fontsize=11, color='dimgray')

    # Fleet SOH trend for worst vehicles
    ax_t = fig.add_subplot(gs[2, 1])
    worst = df.sort_values('soh_now', na_position='last').head(min(3, len(df)))['vehicle_id'].tolist()
    plotted = 0
    for vid in worst:
        if vid not in xgb_results:
            continue
        g = xgb_results[vid]['sessions'].copy()
        s = _finite_series(g['start_utc']) if 'start_utc' in g.columns else pd.Series(dtype=float)
        yv = _finite_series(g['soh_xgb']) if 'soh_xgb' in g.columns else _finite_series(g['soh_smooth'])
        if len(yv) == 0:
            continue
        if s.notna().sum() >= max(5, int(0.6 * max(len(g), 1))):
            g = g.assign(_x=((s - s.min()) / 86400.0 / 30.44).values.astype(float)).sort_values('_x')
            xv = g['_x'].values.astype(float)
        else:
            xv = np.arange(len(yv), dtype=float)
        yarr = _finite_series(yv).values.astype(float)
        yarr_plot = _smooth_line_until_soh_eol_for_plot(xv, yarr, eol=SOH_EOL, window=9)
        m = np.isfinite(xv) & np.isfinite(yarr_plot)
        if np.any(m):
            xs, ys = _prepare_smooth_plot_curve(xv[m], yarr_plot[m], eol=SOH_EOL, window=9)
            md = np.isfinite(xs) & np.isfinite(ys)
            if np.any(md):
                ax_t.plot(xs[md], ys[md], lw=2.2, label=f"{vid} ({np.nanmean(yarr[m]):.1f}%)")
            plotted += 1
    if plotted > 0:
        ax_t.axhline(85, color='darkorange', linestyle=':', alpha=0.6, label='Monitor threshold (85%)')
        ax_t.axhline(75, color='crimson', linestyle=':', alpha=0.6, label='Replace-soon threshold (75%)')
        ax_t.set_title("SOH Trend Over Time (Worst Vehicles)")
        ax_t.set_xlabel("Elapsed months")
        ax_t.set_ylabel("SOH (%)")
        ax_t.grid(alpha=0.25)
        ax_t.legend(fontsize=8)
    else:
        ax_t.axis('off')
        ax_t.text(0.03, 0.55, "Insufficient trend data for worst-vehicle SOH trajectories.", fontsize=12)

    plt.tight_layout()
    plt.savefig(dash_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------
    # Vehicle health cards
    # -----------------------------
    for vid, res in xgb_results.items():
        g = res['sessions'].copy()
        # Customer-facing axis: use elapsed days for readability and consistency.
        start_vals = _finite_series(g['start_utc']) if 'start_utc' in g.columns else pd.Series(dtype=float)
        min_needed = max(5, int(0.6 * max(len(g), 1)))
        if start_vals.notna().sum() >= min_needed and np.isfinite(start_vals.max() - start_vals.min()) and (start_vals.max() - start_vals.min()) > 0:
            g = g.sort_values('start_utc').reset_index(drop=True)
            x0 = _finite_series(g['start_utc']).min()
            x = ((_finite_series(g['start_utc']) - x0) / 86400.0).values.astype(float)
            x_label = 'Elapsed days since first charge session'
        else:
            g = g.reset_index(drop=True)
            g['__session_idx'] = np.arange(len(g), dtype=float)
            x = g['__session_idx'].values.astype(float)
            x_label = 'Charging session index'
        y_raw = _finite_series(g['soh_label']).values
        y_trend = _finite_series(g['soh_xgb']).values
        y_trend_plot = _smooth_line_until_soh_eol_for_plot(x, y_trend, eol=SOH_EOL, window=9)
        rr = rul_all.get(vid, {})
        risk, risk_color = _risk_label(rr.get('soh_now', np.nan), rr.get('rul_days_p50', np.nan))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [2.4, 1]})
        m1 = np.isfinite(x) & np.isfinite(y_raw)
        if np.any(m1):
            ax1.scatter(x[m1], y_raw[m1], s=10, alpha=0.35, color='steelblue', label='Session SOH points')
        m2 = np.isfinite(x) & np.isfinite(y_trend_plot)
        if np.any(m2):
            xs, ys = _prepare_smooth_plot_curve(x[m2], y_trend_plot[m2], eol=SOH_EOL, window=9)
            md = np.isfinite(xs) & np.isfinite(ys)
            if np.any(md):
                ax1.plot(xs[md], ys[md], color='crimson', lw=2, label='Health trend')
        ax1.axhline(SOH_EOL, color='red', linestyle=':', alpha=0.6, label='EOL 80%')
        ax1.set_title(f"{vid} - Battery Health Trend")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('SOH (%)')
        ax1.set_ylim(70, 105)
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        ax2.axis('off')
        ax2.text(0.03, 0.93, f"IMEI: {_imei_text(vid)}", fontsize=13, fontweight='bold')
        ax2.text(0.03, 0.82, f"Battery Health  : {rr.get('soh_now', np.nan):.2f}%" if np.isfinite(rr.get('soh_now', np.nan)) else "Battery Health  : NA", fontsize=11.5)
        ax2.text(0.03, 0.74, f"Initial capacity: {_fmt_ah(rr.get('init_capacity_ah', np.nan))} ({_fmt_kwh(rr.get('init_capacity_kwh', np.nan))})", fontsize=10.5)
        ax2.text(0.03, 0.66, f"Capacity today  : {_fmt_ah(rr.get('current_capacity_ah', np.nan))} ({_fmt_kwh(rr.get('current_capacity_kwh', np.nan))})", fontsize=10.5)
        ax2.text(
            0.03, 0.57,
            f"Expected life remaining : {_life_remaining_text(rr.get('rul_days_p50', np.nan), rr.get('rul_days_p10', np.nan), rr.get('rul_days_p90', np.nan))}",
            fontsize=10.2
        )
        ax2.text(0.03, 0.48, f"Est. end-of-life date   : {_fmt_eol_date(rr.get('eol_date_p50_ist', pd.NaT), rr.get('rul_days_p50', np.nan))}", fontsize=10.2)
        ax2.text(0.03, 0.39, f"Distance covered        : {_fmt_km(rr.get('km_run_till_date', np.nan))}", fontsize=10.2)
        ax2.text(
            0.03, 0.30,
            f"Distance remaining      : likely={_fmt_km(rr.get('km_to_eol_p50', np.nan), approx=True)}",
            fontsize=10.2
        )
        ax2.text(0.03, 0.20, f"Cycles (equiv full)     : {rr.get('equivalent_full_cycles', np.nan):.0f}" if np.isfinite(rr.get('equivalent_full_cycles', np.nan)) else "Cycles (equiv full)     : NA", fontsize=9.8)
        ax2.text(0.03, 0.12, f"Partial charging events : {int(rr.get('partial_charging_events_count', 0))}" if np.isfinite(rr.get('partial_charging_events_count', np.nan)) else "Partial charging events : NA", fontsize=9.8)
        ax2.text(0.03, 0.06, f"Risk: {risk}", color=risk_color, fontsize=10.5, fontweight='bold')

        plt.tight_layout()
        safe_vid = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(vid))
        out_file = cards_dir / f"{safe_vid}_card.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\n  Customer dashboard -> {dash_path}")
    print(f"  Customer vehicle cards -> {cards_dir}")


def export_results_csv(xgb_results, lstm_results, rul_all, replacement_events, save_path):
    """
    Export final outputs to CSV files.
    """
    out_root = Path(save_path)
    out_dir = out_root.parent / f"{out_root.stem}_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fleet summary
    summary_rows = []
    for vid, r in rul_all.items():
        summary_rows.append({
            'vehicle_id': vid,
            'soh_now_pct': r.get('soh_now', np.nan),
            'init_capacity_ah': r.get('init_capacity_ah', np.nan),
            'init_capacity_kwh': r.get('init_capacity_kwh', np.nan),
            'current_capacity_ah': r.get('current_capacity_ah', np.nan),
            'current_capacity_kwh': r.get('current_capacity_kwh', np.nan),
            'eol_capacity_ah': r.get('eol_capacity_ah', np.nan),
            'charging_events_count': r.get('charging_events_count', np.nan),
            'equivalent_full_cycles': r.get('equivalent_full_cycles', np.nan),
            'equivalent_full_cycles_epoch': r.get('equivalent_full_cycles_epoch', np.nan),
            'pack_config_guess': r.get('pack_config_guess', ''),
            'pack_score_confidence': r.get('pack_score_confidence', np.nan),
            'pack_options_source': r.get('pack_options_source', ''),
            'pack_series_cells': r.get('pack_series_cells', np.nan),
            'pack_parallel_guess': r.get('pack_parallel_guess', np.nan),
            'pack_voltage_ref_v': r.get('pack_voltage_ref_v', np.nan),
            'pack_q_data_median_ah': r.get('pack_q_data_median_ah', np.nan),
            'pack_q_data_count': r.get('pack_q_data_count', np.nan),
            'pack_config_changed_strong': r.get('pack_config_changed_strong', False),
            'config_epoch_id': r.get('config_epoch_id', 0),
            'km_run_till_date': r.get('km_run_till_date', np.nan),
            'km_to_eol_p10': r.get('km_to_eol_p10', np.nan),
            'km_to_eol_p50': r.get('km_to_eol_p50', np.nan),
            'km_to_eol_p90': r.get('km_to_eol_p90', np.nan),
            'rul_days_p10': r.get('rul_days_p10', np.nan),
            'rul_days_p50': r.get('rul_days_p50', np.nan),
            'rul_days_p90': r.get('rul_days_p90', np.nan),
            'energy_to_eol_kwh_p10': r.get('energy_to_eol_kwh_p10', np.nan),
            'energy_to_eol_kwh_p50': r.get('energy_to_eol_kwh_p50', np.nan),
            'energy_to_eol_kwh_p90': r.get('energy_to_eol_kwh_p90', np.nan),
            'kwh_per_day_hist': r.get('kwh_per_day_hist', np.nan),
            'deliverable_kwh_at_80_model': r.get('deliverable_kwh_at_80_model', np.nan),
            'deliverable_kwh_at_80_nominal': r.get('deliverable_kwh_at_80_nominal', np.nan),
            'phase2_slope_pct_per_axis': r.get('phase2_slope', np.nan),
            'slope_basis': r.get('slope_basis', ''),
            'last_seen_datetime_ist': r.get('last_seen_datetime_ist', pd.NaT),
            'eol_date_p10_ist': r.get('eol_date_p10_ist', pd.NaT),
            'eol_date_p50_ist': r.get('eol_date_p50_ist', pd.NaT),
            'eol_date_p90_ist': r.get('eol_date_p90_ist', pd.NaT),
            'bms_correction_factor': r.get('bms_correction_factor', 1.0),
        })
    pd.DataFrame(summary_rows).sort_values('vehicle_id').to_csv(out_dir / "fleet_summary.csv", index=False)

    # 2) Replacement events
    if replacement_events is None:
        replacement_events = pd.DataFrame()
    replacement_events.to_csv(out_dir / "replacement_events.csv", index=False)

    # 3) Session-level modeled data
    sess_parts = []
    for vid, res in xgb_results.items():
        g = res['sessions'].copy()
        g['vehicle_id'] = vid
        sess_parts.append(g)
    if sess_parts:
        pd.concat(sess_parts, ignore_index=True).to_csv(out_dir / "session_predictions.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "session_predictions.csv", index=False)

    # 4) LSTM trajectory points
    traj_parts = []
    for vid, lr in lstm_results.items():
        lb = int(lr.get('lookback', 0))
        x_all = np.asarray(lr.get('hrlfc_seq', []), dtype=float)
        y_all = np.asarray(lr.get('soh_seq', []), dtype=float)
        y_pred = np.asarray(lr.get('soh_pred', []), dtype=float)
        x_pred = x_all[lb:] if len(x_all) >= lb else np.array([], dtype=float)
        n = min(len(x_pred), len(y_pred))
        if n > 0:
            traj_parts.append(pd.DataFrame({
                'vehicle_id': vid,
                'x_axis': x_pred[:n],
                'soh_pred': y_pred[:n],
            }))
        if len(y_all) > 0:
            traj_parts.append(pd.DataFrame({
                'vehicle_id': vid,
                'x_axis': x_all[:len(y_all)],
                'soh_input': y_all,
            }))
    if traj_parts:
        pd.concat(traj_parts, ignore_index=True).to_csv(out_dir / "lstm_trajectory.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "lstm_trajectory.csv", index=False)

    print(f"\n  CSV exports -> {out_dir}")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def run_pipeline(
    path: str,
    plot_path: str = 'soh_rul_results.png',
    incremental: bool = True,
    state_path: str = None,
    overlap_hours: float = INCREMENTAL_OVERLAP_HOURS,
):
    _validate_runtime_config()

    state_file = _resolve_state_path(plot_path=plot_path, state_path=state_path)
    state = _load_pipeline_state(state_file) if incremental else None
    prev_sessions = None
    since_utc = None
    if state is not None:
        prev_sessions = state.get('sessions')
        since_utc = state.get('last_utc_num')
        n_prev = len(prev_sessions) if isinstance(prev_sessions, pd.DataFrame) else 0
        print(f"  Incremental state loaded: sessions={n_prev:,}, watermark_utc={since_utc}")

    try:
        df_raw = load_and_clean(
            path,
            since_utc=since_utc if incremental else None,
            overlap_sec=max(0.0, float(overlap_hours)) * 3600.0,
        )
    except RuntimeError as e:
        if incremental and str(e).startswith("[INCREMENTAL]"):
            print(f"  {e}")
            print("  No new data to process. Reusing cached outputs from previous run.")
            if state is not None and all(k in state for k in ['xgb_results', 'lstm_results', 'rul_all']):
                xgb_results = state.get('xgb_results', {})
                lstm_results = state.get('lstm_results', {})
                rul_all = state.get('rul_all', {})
                replacement_events = state.get('replacement_events', pd.DataFrame())
                _print_summary_tables(rul_all, replacement_events)
                plot_results(xgb_results, lstm_results, rul_all, plot_path)
                plot_customer_views(xgb_results, lstm_results, rul_all, replacement_events, plot_path)
                export_results_csv(xgb_results, lstm_results, rul_all, replacement_events, plot_path)
                return xgb_results, lstm_results, rul_all
        raise

    runtime_alias_map = _extract_runtime_vehicle_alias_map(df_raw)
    if incremental and isinstance(state, dict) and runtime_alias_map:
        state_remapped = _remap_pipeline_state_vehicle_ids(state, runtime_alias_map)
        if state_remapped is not state:
            state = state_remapped
            prev_sessions = state.get('sessions')
            remapped_cached_ids = sorted({
                _normalize_vehicle_id_text(v)
                for v in runtime_alias_map.values()
                if _normalize_vehicle_id_text(v)
            })
            print(f"    Cached state vehicle IDs aligned to current file identity: {len(remapped_cached_ids)} vehicle(s)")

    sessions_new = build_session_table(df_raw)

    # Per-vehicle processing scope:
    #   - existing vehicle_id in state  -> incremental
    #   - unseen/new vehicle_id         -> from_start
    seen_vehicle_ids = set()
    if incremental and isinstance(prev_sessions, pd.DataFrame) and ('vehicle_id' in prev_sessions.columns):
        seen_vehicle_ids = set(prev_sessions['vehicle_id'].dropna().tolist())

    if 'vehicle_id' in sessions_new.columns:
        if incremental:
            sessions_new['processing_scope'] = sessions_new['vehicle_id'].apply(
                lambda v: 'incremental' if v in seen_vehicle_ids else 'from_start'
            )
        else:
            sessions_new['processing_scope'] = 'from_start'
    else:
        sessions_new['processing_scope'] = 'unknown'

    if incremental and since_utc is not None and np.isfinite(float(since_utc)):
        end_utc = _finite_series(sessions_new.get('end_utc', pd.Series(np.nan, index=sessions_new.index)))
        is_existing = sessions_new['processing_scope'].eq('incremental')
        is_new_vid = sessions_new['processing_scope'].eq('from_start')
        # Existing vehicles: only new sessions after watermark.
        # New vehicles: process from start (do not apply global watermark cutoff).
        new_mask = (is_existing & (end_utc > float(since_utc))) | is_new_vid
        sessions_new = sessions_new.loc[new_mask].copy()
        n_inc = int((sessions_new['processing_scope'] == 'incremental').sum())
        n_new = int((sessions_new['processing_scope'] == 'from_start').sum())
        print(f"    Sessions selected -> incremental={n_inc:,}, from_start={n_new:,}, total={len(sessions_new):,}")

    if incremental:
        sessions = _merge_sessions_cached(prev_sessions if isinstance(prev_sessions, pd.DataFrame) else pd.DataFrame(), sessions_new)
    else:
        sessions = sessions_new

    if len(sessions) == 0:
        raise RuntimeError("No charging sessions available after incremental merge.")

    cached_xgb = state.get('xgb_results', {}) if isinstance(state, dict) else {}
    cached_lstm = state.get('lstm_results', {}) if isinstance(state, dict) else {}
    cached_rul = state.get('rul_all', {}) if isinstance(state, dict) else {}
    cached_repl = state.get('replacement_events', pd.DataFrame()) if isinstance(state, dict) else pd.DataFrame()
    cached_init_map = _extract_cached_init_capacity_map(cached_rul)
    cached_pack_ctx = _extract_cached_pack_context_map(cached_rul)

    if incremental and since_utc is not None and len(sessions_new) == 0 and all(
        isinstance(state, dict) and (k in state) for k in ['xgb_results', 'lstm_results', 'rul_all']
    ):
        print("    No new charging sessions after watermark. Reusing cached outputs.")
        xgb_results = cached_xgb
        lstm_results = cached_lstm
        rul_all = cached_rul
        replacement_events = cached_repl if isinstance(cached_repl, pd.DataFrame) else pd.DataFrame()
    elif incremental and since_utc is not None and len(sessions_new) == 0:
        print("    No new charging sessions after watermark; cached models unavailable, running full rebuild.")
        labeled      = compute_soh_labels(sessions, init_capacity_overrides=cached_init_map, prior_pack_context=cached_pack_ctx)
        xgb_results  = train_xgboost_soh(labeled)
        lstm_results = train_lstm_trajectory(xgb_results, lookback=10)
        rul_all      = compute_all_rul(xgb_results, lstm_results, df_raw, prev_rul_all=cached_rul)
        replacement_events = detect_battery_replacements(xgb_results)

    elif incremental and since_utc is not None and len(sessions_new) > 0:
        touched = set(sessions_new['vehicle_id'].dropna().tolist()) if 'vehicle_id' in sessions_new.columns else set()
        new_vids = set(sessions_new.loc[sessions_new['processing_scope'] == 'from_start', 'vehicle_id'].dropna().tolist()) if 'processing_scope' in sessions_new.columns else set()
        inc_vids = set(sessions_new.loc[sessions_new['processing_scope'] == 'incremental', 'vehicle_id'].dropna().tolist()) if 'processing_scope' in sessions_new.columns else set()
        print(f"    Vehicles selected: total={len(touched)} | incremental={len(inc_vids)} | from_start={len(new_vids)}")

        if not touched:
            print("    No new vehicle sessions found after watermark. Reusing cached outputs.")
            xgb_results = cached_xgb
            lstm_results = cached_lstm
            rul_all = cached_rul
            replacement_events = cached_repl if isinstance(cached_repl, pd.DataFrame) else pd.DataFrame()
        else:
            target_sessions = sessions[sessions['vehicle_id'].isin(list(touched))].copy()
            init_override_touched = {v: cached_init_map[v] for v in touched if v in cached_init_map}
            try:
                prior_ctx_touched = {v: cached_pack_ctx.get(v, {}) for v in touched if v in cached_pack_ctx}
                labeled_new = compute_soh_labels(
                    target_sessions,
                    init_capacity_overrides=init_override_touched,
                    prior_pack_context=prior_ctx_touched,
                )
            except RuntimeError as e:
                if str(e).startswith("[CRITICAL] Pipeline stopped: No valid sessions found after filtering."):
                    print("    No valid new labeled sessions after filtering; reusing cached outputs.")
                    xgb_results = cached_xgb
                    lstm_results = cached_lstm
                    rul_all = cached_rul
                    replacement_events = cached_repl if isinstance(cached_repl, pd.DataFrame) else pd.DataFrame()
                else:
                    raise
            else:
                xgb_new = train_xgboost_soh(labeled_new)
                lstm_new = train_lstm_trajectory(xgb_new, lookback=10)
                prev_rul_touched = {v: cached_rul.get(v, {}) for v in touched if v in cached_rul}
                rul_new = compute_all_rul(xgb_new, lstm_new, df_raw, prev_rul_all=prev_rul_touched)
                repl_new = detect_battery_replacements(xgb_new)

                xgb_results = dict(cached_xgb)
                xgb_results.update(xgb_new)
                lstm_results = dict(cached_lstm)
                lstm_results.update(lstm_new)
                rul_all = dict(cached_rul)
                rul_all.update(rul_new)

                if isinstance(cached_repl, pd.DataFrame) and len(cached_repl) > 0:
                    replacement_events = pd.concat([cached_repl, repl_new], ignore_index=True, sort=False)
                    dedupe_cols = [c for c in ['vehicle_id', 'event_session_idx', 'event_utc'] if c in replacement_events.columns]
                    if dedupe_cols:
                        replacement_events = replacement_events.drop_duplicates(subset=dedupe_cols, keep='last')
                    replacement_events = replacement_events.sort_values(
                        [c for c in ['vehicle_id', 'event_session_idx'] if c in replacement_events.columns]
                    ).reset_index(drop=True)
                else:
                    replacement_events = repl_new
    else:
        labeled      = compute_soh_labels(
            sessions,
            init_capacity_overrides=cached_init_map if incremental else None,
            prior_pack_context=cached_pack_ctx if incremental else None,
        )
        xgb_results  = train_xgboost_soh(labeled)
        lstm_results = train_lstm_trajectory(xgb_results, lookback=10)
        rul_all      = compute_all_rul(xgb_results, lstm_results, df_raw, prev_rul_all=cached_rul if incremental else None)
        replacement_events = detect_battery_replacements(xgb_results)

    _print_summary_tables(rul_all, replacement_events)
    plot_results(xgb_results, lstm_results, rul_all, plot_path)
    plot_customer_views(xgb_results, lstm_results, rul_all, replacement_events, plot_path)
    export_results_csv(xgb_results, lstm_results, rul_all, replacement_events, plot_path)

    if incremental:
        last_utc = np.nan
        if '_utc_num' in df_raw.columns:
            last_utc = _finite_series(df_raw['_utc_num']).max()
        if not np.isfinite(last_utc) and since_utc is not None:
            last_utc = float(since_utc)
        state_out = {
            'schema_version': 1,
            'saved_at_utc': pd.Timestamp.utcnow(),
            'last_utc_num': float(last_utc) if np.isfinite(last_utc) else since_utc,
            'sessions': sessions,
            'xgb_results': _state_safe_xgb_results(xgb_results),
            'lstm_results': _state_safe_lstm_results(lstm_results),
            'rul_all': rul_all,
            'replacement_events': replacement_events,
        }
        _save_pipeline_state(state_file, state_out)

    return xgb_results, lstm_results, rul_all


if __name__ == '__main__':
    import sys

    script_dir = Path(__file__).resolve().parent
    # Cross-platform default data path. You can override via CLI arg1.
    DEFAULT_DATA_PATH = str(script_dir / "data")

    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    if len(sys.argv) > 2:
        plot_path = sys.argv[2]
    else:
        plot_path = str(script_dir / "soh_rul_results.png")

    profile_name = PROFILE_DEFAULT_NAME
    profile_path = None
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None
    arg6 = sys.argv[6] if len(sys.argv) > 6 else None

    if arg3:
        if str(arg3).lower().endswith('.json'):
            profile_name = 'custom'
            profile_path = arg3
        else:
            profile_name = arg3
    if arg4:
        profile_path = arg4

    incremental_mode = True
    explicit_state_path = None
    if arg5:
        m = str(arg5).strip().lower()
        if m in {'full', 'rebuild', 'noinc', 'nonincremental', 'false', '0'}:
            incremental_mode = False
        elif m in {'inc', 'incremental', 'true', '1'}:
            incremental_mode = True
        else:
            explicit_state_path = arg5
    if arg6:
        explicit_state_path = arg6

    # Single-file default behavior:
    # do not auto-load external JSON config unless user explicitly passes it.

    print("=" * 70)
    print("  EV Fleet SOH + RUL Pipeline  (Combined: Cleaning + Modelling)")
    print("=" * 70)
    print(f"  Data path : {data_path}")
    print(f"  Plot path : {plot_path}")
    print(f"  Profile   : {profile_name}")
    print(f"  Profile JSON : {profile_path if profile_path else 'None'}")
    print(f"  Incremental mode : {'ON' if incremental_mode else 'OFF'}")
    print(f"  State path : {_resolve_state_path(plot_path=plot_path, state_path=explicit_state_path)}")
    print("=" * 70)

    # Helpful cross-platform startup check.
    if not Path(data_path).exists():
        print("[ERROR] Data path not found.")
        print(f"  Provided: {data_path}")
        print("  Usage: python3 ev_pipeline_combined.py <data_path> [plot_path] [profile_name|profile_json] [profile_json] [inc|full|state_path] [state_path]")
        print(f"  Example: python3 {Path(__file__).name} ./data ./soh_rul_results.png conservative")
        raise SystemExit(2)

    apply_config_profile(profile_name=profile_name, profile_path=profile_path)
    run_pipeline(
        data_path,
        plot_path,
        incremental=incremental_mode,
        state_path=explicit_state_path,
        overlap_hours=INCREMENTAL_OVERLAP_HOURS,
    )
