#!/usr/bin/env python3
"""Generate high-quality synthetic pretraining data → local parquet shards.

Shards are stored in HuggingFace-compatible format at corpus/pretrain_v2/data/.
Once a write-enabled HF token is available, upload with:

    huggingface-cli upload avewright/tabula-pretraining-corpus-v2 \
        corpus/pretrain_v2/data data --repo-type dataset

Quality assurance:
  - Domain-realistic column names from 11 domain vocabularies
  - Hard quality gates: no constant cols, no all-null, minority frac > 1%
  - Soft quality gate: RF utility AUC (logged, datasets always kept if hard pass)
  - Concept drift injection on ~20% of datasets
  - MCAR missingness on ~30% of datasets
  - Diverse generators: TreePrior, SCM, GaussianMixture, Polynomial,
    Regression, MixedType_* variants

Target: fill disk to ~42GB used (leaving 8GB headroom on 50GB).
Each shard ~ 2M rows ~ 150-250 MB parquet.

Hardware: 48 CPUs, 247 GB RAM, ~49 GB disk free.
"""
from __future__ import annotations

import gc
import io
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

sys.path.insert(0, ".")

from tabula.data.synthetic import (
    TreePriorGenerator,
    GaussianMixtureGenerator,
    PolynomialGenerator,
    SCMGenerator,
    RegressionSyntheticGenerator,
    MixedTypeGenerator,
    SyntheticDatasetMeta,
)


# ── Output paths ──────────────────────────────────────────────────────
CORPUS_DIR = Path("corpus/pretrain_v2/data")
LOG_PATH = Path("corpus/pretrain_v2/generation_log.jsonl")
MANIFEST_PATH = Path("corpus/pretrain_v2/manifest.json")

# ── Generation parameters ─────────────────────────────────────────────
NUM_WORKERS = 40          # Use 40 of 48 CPUs
ROWS_PER_SHARD = 2_000_000
MAX_DISK_GB = 48          # Stop when total disk usage reaches this
DISK_CHECK_INTERVAL = 5   # Check disk every N shards

# ── HF Upload settings ────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "avewright/tabula-pretraining-corpus-v2"
HF_UPLOAD = True          # Upload each shard to HF and delete local copy

MIN_FEATURES = 4
MAX_FEATURES = 64
MIN_SAMPLES = 500
MAX_SAMPLES = 50_000

METHODS_TIER1 = [
    "TreePrior", "SCM", "GaussianMixture", "Polynomial", "Regression",
]
METHOD_WEIGHTS = [0.25, 0.20, 0.25, 0.20, 0.10]

TASK_TYPES = ["binary", "binary", "multiclass", "regression"]  # 50% binary


# ── Domain vocabulary (11 domains) ────────────────────────────────────
DOMAIN_VOCAB = {
    "finance": [
        "income", "age", "debt_ratio", "credit_score", "loan_amount",
        "employment_years", "balance", "expenses", "interest_rate", "default",
        "assets", "liabilities", "revenue", "margin", "tax_rate",
        "net_worth", "monthly_payment", "annual_income", "savings_rate", "insurance_premium",
        "mortgage_balance", "credit_limit", "utilization_ratio", "payment_history", "delinquency_count",
        "bankruptcy_flag", "collection_count", "inquiry_count", "account_age_months", "open_accounts",
        "revolving_balance", "installment_balance", "total_debt", "debt_to_income",
        "monthly_expenses", "rent_amount", "dividend_yield", "stock_return", "bond_rating",
        "portfolio_value", "risk_tolerance", "sharpe_ratio", "volatility",
        "market_cap", "price_earnings", "book_value", "earnings_per_share", "return_on_equity",
        "return_on_assets", "operating_margin", "gross_margin", "free_cash_flow",
        "ebitda", "current_ratio", "quick_ratio", "inventory_turnover",
        "working_capital", "capital_expenditure", "depreciation",
        "total_equity", "total_assets", "total_liabilities", "long_term_debt",
        "interest_coverage", "leverage_ratio", "payout_ratio", "beta",
    ],
    "health": [
        "bmi", "age", "blood_pressure", "cholesterol", "glucose", "smoking",
        "alcohol_units", "exercise_days", "family_history", "diagnosis",
        "heart_rate", "creatinine", "hemoglobin", "white_cell_count",
        "systolic_bp", "diastolic_bp", "pulse_rate", "respiratory_rate", "oxygen_saturation",
        "temperature_c", "weight_kg", "height_cm", "waist_circumference",
        "body_fat_pct", "lean_mass", "bone_density", "vitamin_d", "calcium_level",
        "potassium_level", "sodium_level", "albumin", "bilirubin",
        "triglycerides", "hdl_cholesterol", "ldl_cholesterol",
        "total_protein", "a1c_level", "fasting_glucose", "insulin_level",
        "thyroid_tsh", "iron_level", "ferritin",
        "platelet_count", "red_cell_count", "esr_rate", "crp_level",
        "troponin", "lactate", "gfr_estimated",
    ],
    "ecommerce": [
        "price", "quantity", "discount", "return_rate", "category", "rating",
        "reviews", "shipping_days", "refund", "revenue", "cart_size",
        "session_minutes", "clicks", "conversion", "churn",
        "page_views", "bounce_rate", "time_on_page", "scroll_depth",
        "add_to_cart_rate", "checkout_rate", "abandonment_rate", "avg_order_value",
        "customer_lifetime_value", "purchase_frequency", "recency_days",
        "rfm_score", "loyalty_tier", "coupon_used", "discount_pct", "shipping_cost",
        "total_amount", "product_weight", "inventory_level",
        "seller_rating", "competitor_price", "search_rank",
        "click_through_rate", "cost_per_click", "ad_spend",
        "organic_traffic_pct", "email_open_rate", "email_click_rate",
        "support_tickets", "csat_score", "nps_score", "review_sentiment",
    ],
    "iot": [
        "temperature", "humidity", "pressure", "vibration", "voltage", "current",
        "uptime_hours", "error_rate", "latency_ms", "throughput_gbps",
        "packet_loss", "cpu_pct", "mem_pct", "disk_iops", "fan_rpm",
        "power_watts", "energy_kwh", "frequency_hz", "signal_strength_dbm",
        "snr_db", "bandwidth_mbps", "connection_count",
        "active_sessions", "cache_hit_ratio", "load_average_1m",
        "thread_count", "response_time_p50", "response_time_p95",
        "request_rate_rps", "error_count", "uptime_pct", "mtbf_hours",
        "sensor_accuracy", "sampling_rate_hz", "duty_cycle_pct",
        "acceleration_g", "flow_rate_lpm", "battery_voltage", "charge_pct",
        "ambient_temp_c", "wind_speed_mps",
    ],
    "hr": [
        "tenure_years", "salary", "performance_score", "overtime_hours",
        "absences", "promotions", "department_id", "role_level",
        "satisfaction", "attrition", "training_hours", "team_size",
        "remote_days", "bonus_pct", "peer_rating",
        "manager_rating", "goal_completion_pct", "project_count",
        "years_experience", "education_level", "certification_count",
        "interview_score", "engagement_score", "burnout_risk",
        "commute_minutes", "meeting_hours_weekly", "email_volume_daily",
        "task_completion_rate", "total_compensation", "pto_days_used",
        "health_risk_score", "expense_report_total",
    ],
    "science": [
        "wavelength", "intensity", "mass", "velocity", "acceleration",
        "temperature_k", "pressure_pa", "volume", "concentration", "ph",
        "reaction_time", "yield_pct", "purity", "entropy", "energy_kj",
        "molar_mass", "density_gcm3", "viscosity_pa_s", "surface_tension",
        "melting_point_k", "boiling_point_k", "heat_capacity",
        "activation_energy", "equilibrium_constant", "gibbs_free_energy",
        "band_gap_ev", "absorbance", "transmittance",
        "peak_area", "peak_height", "sample_concentration",
        "p_value", "t_statistic", "f_statistic", "effect_size",
        "variance_explained", "silhouette_score",
    ],
    "logistics": [
        "weight_kg", "distance_km", "transit_days", "on_time", "carrier_id",
        "cost_usd", "damage_rate", "warehouse_id", "load_factor",
        "fuel_consumption", "route_id", "customs_flag", "return_flag",
        "package_count", "shipment_volume_m3", "shipment_weight_kg",
        "origin_lat", "origin_lon", "dest_lat", "dest_lon",
        "road_distance_km", "loading_time_min", "unloading_time_min",
        "temperature_controlled", "hazmat_flag", "insurance_value",
        "freight_charge", "total_charge", "cost_per_kg",
        "order_accuracy_pct", "inventory_turns", "lead_time_days",
        "claim_count", "eta_variance_hours",
    ],
    "education": [
        "gpa", "credits_completed", "attendance_rate", "assignment_score",
        "exam_score", "quiz_score", "lab_score", "project_score",
        "midterm_grade", "final_grade", "cumulative_gpa",
        "sat_math", "sat_verbal", "study_hours_weekly",
        "course_load", "class_size", "instructor_rating",
        "dropout_risk_score", "financial_aid_amount",
        "extracurricular_count", "internship_count",
    ],
    "manufacturing": [
        "cycle_time_sec", "throughput_units", "yield_rate_pct", "defect_rate",
        "scrap_rate", "rework_rate", "first_pass_yield", "oee_pct",
        "downtime_minutes", "changeover_time_min", "run_time_hours",
        "machine_age_years", "mtbf_machine_hours", "spare_parts_inventory",
        "labor_cost_per_unit", "energy_cost_per_unit", "total_cost_per_unit",
        "batch_size", "raw_material_quality", "temperature_setpoint",
        "tool_wear_pct", "surface_roughness", "hardness", "tensile_strength",
        "weight_deviation_g", "shelf_life_days",
    ],
    "environment": [
        "air_temperature_c", "relative_humidity_pct", "wind_speed_ms",
        "atmospheric_pressure_hpa", "precipitation_mm", "cloud_cover_pct",
        "solar_radiation_wm2", "uv_index_env", "soil_moisture_pct",
        "river_discharge_m3s", "dissolved_oxygen_mgl", "turbidity_env_ntu",
        "pm25_ugm3", "pm10_ugm3", "ozone_ppb", "no2_ppb", "co2_ppm_env",
        "aqi_index", "noise_level_db", "biodiversity_index", "ndvi_value",
        "deforestation_rate", "fire_risk_index", "drought_index",
    ],
    "telecom": [
        "call_duration_sec", "call_count", "sms_count", "data_usage_mb",
        "monthly_charge", "contract_months", "tenure_months", "churn_flag",
        "arpu", "data_plan_gb", "dropped_call_rate",
        "download_speed_mbps", "upload_speed_mbps", "ping_ms",
        "signal_strength_bars", "device_age_months", "device_price",
        "streaming_hours", "complaint_count", "support_call_count",
        "csat_score_telecom", "nps_score_telecom", "overage_charge",
    ],
}
DOMAINS = list(DOMAIN_VOCAB.keys())


# ── Fixed-width schema ────────────────────────────────────────────────
# All datasets share: feat_0..feat_{MAX_FEATURES-1}, target, _source_meta
# Domain-realistic names are stored inside _source_meta JSON.
# This avoids the column-union NaN explosion when concatenating.
FEAT_COLS = [f"feat_{i}" for i in range(MAX_FEATURES)]


# ── Worker function (runs in child process) ───────────────────────────
def _generate_one(args: dict) -> dict | None:
    """Generate one synthetic dataset with quality gates.

    Returns a dict with parquet bytes in FIXED-WIDTH schema:
      feat_0..feat_63, target, _source_meta
    Unused feature slots are NaN (but only ceil(n_features/64) of columns).
    """
    try:
        seed = args["seed"]
        rng = np.random.default_rng(seed)

        n_features = int(rng.integers(MIN_FEATURES, MAX_FEATURES + 1))
        n_samples = int(rng.integers(MIN_SAMPLES, MAX_SAMPLES + 1))
        method = str(rng.choice(METHODS_TIER1, p=METHOD_WEIGHTS))
        task_type = str(rng.choice(TASK_TYPES))
        n_classes = (
            2 if task_type == "binary"
            else int(rng.integers(3, 10)) if task_type == "multiclass"
            else 2
        )

        # Build generator
        gen = _build_generator(method, n_samples, n_features, n_classes, task_type, rng)
        df, meta = gen.generate(seed=seed)

        # ── Pick domain names (stored in metadata, not column names) ─
        domain = str(rng.choice(DOMAINS))
        feature_cols_orig = [c for c in df.columns if c not in ("target", "_source_meta")]
        actual_n = min(len(feature_cols_orig), MAX_FEATURES)

        # Domain names for metadata
        vocab = list(DOMAIN_VOCAB.get(domain, []))
        rng.shuffle(vocab)
        domain_names = vocab[:actual_n]
        while len(domain_names) < actual_n:
            domain_names.append(f"feature_{len(domain_names)}")

        # ── Extract features into fixed-width numpy array ────────
        feat_vals = df[feature_cols_orig[:actual_n]].values.astype(np.float32)
        target_vals = df["target"].values.astype(np.float32)
        n_rows = len(df)

        # ── Missingness injection (~30%) ─────────────────────────
        missingness_rate = 0.0
        if rng.random() < 0.30:
            missingness_rate = float(rng.uniform(0.02, 0.15))
            mask = rng.random(size=feat_vals.shape) < missingness_rate
            feat_vals[mask] = np.nan

        # ── Concept drift injection (~20%) ───────────────────────
        drift_applied = False
        if rng.random() < 0.20 and n_rows >= 200:
            mid = n_rows // 2
            shift = rng.normal(0, 1.0, size=actual_n).astype(np.float32)
            feat_vals[mid:] += shift
            drift_applied = True

        # ── Hard quality gates ───────────────────────────────────
        # Constant columns
        for i in range(actual_n):
            col = feat_vals[:, i]
            non_nan = col[~np.isnan(col)]
            if len(non_nan) <= 1 or np.nanstd(non_nan) < 1e-10:
                return {"status": "gate_fail", "notes": f"constant col {i}"}

        # Target needs >= 2 classes for classification
        if task_type in ("binary", "multiclass"):
            uniq = np.unique(target_vals[~np.isnan(target_vals)])
            if len(uniq) < 2:
                return {"status": "gate_fail", "notes": "target < 2 classes"}
            # Minority fraction
            _, counts = np.unique(target_vals[~np.isnan(target_vals)], return_counts=True)
            minority = counts.min() / counts.sum()
            if minority < 0.01:
                return {"status": "gate_fail", "notes": f"minority={minority:.4f}"}

        # Minimum rows
        if n_rows < 100:
            return {"status": "gate_fail", "notes": f"only {n_rows} rows"}

        # ── Soft: quick RF utility (skip for speed if >10K rows) ─
        utility_auc = 0.0
        if n_rows <= 10_000 and actual_n >= 2:
            try:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.model_selection import cross_val_score
                X = np.nan_to_num(feat_vals, nan=0.0)
                y = target_vals
                if task_type == "regression":
                    clf = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0, n_jobs=1)
                    scores = cross_val_score(clf, X, y, cv=2, scoring="r2", n_jobs=1)
                else:
                    clf = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0, n_jobs=1)
                    try:
                        scores = cross_val_score(clf, X, y, cv=2, scoring="roc_auc", n_jobs=1)
                    except Exception:
                        scores = cross_val_score(clf, X, y, cv=2, scoring="accuracy", n_jobs=1)
                utility_auc = float(scores.mean())
            except Exception:
                pass

        # ── Build fixed-width DataFrame ──────────────────────────
        # Pad feature array to MAX_FEATURES columns
        padded = np.full((n_rows, MAX_FEATURES), np.nan, dtype=np.float32)
        padded[:, :actual_n] = feat_vals

        out = pd.DataFrame(padded, columns=FEAT_COLS)
        out["target"] = target_vals

        # ── Source metadata ──────────────────────────────────────
        source_meta = {
            "generator": meta.generator,
            "task_type": meta.task_type,
            "n_features": actual_n,
            "n_classes": meta.n_classes,
            "n_samples": n_rows,
            "domain": domain,
            "feature_names": domain_names,
            "seed": seed,
            "method": method,
            "missingness_rate": missingness_rate,
            "concept_drift": drift_applied,
            "utility_auc": utility_auc,
        }
        out["_source_meta"] = json.dumps(source_meta)

        buf = io.BytesIO()
        out.to_parquet(buf, index=False, engine="pyarrow")
        return {
            "status": "ok",
            "parquet_bytes": buf.getvalue(),
            "n_rows": n_rows,
            "n_features": actual_n,
            "generator": method,
            "task_type": meta.task_type,
            "domain": domain,
            "utility_auc": utility_auc,
        }
    except Exception as e:
        return {"status": "error", "notes": str(e)[:300]}


def _build_generator(method, n_samples, n_features, n_classes, task_type, rng):
    """Build a synthetic data generator."""
    if method == "TreePrior":
        return TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    elif method == "SCM":
        return SCMGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    elif method == "GaussianMixture":
        strategy = str(rng.choice(["linear", "quadratic", "tree"]))
        return GaussianMixtureGenerator(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            label_strategy=strategy,
        )
    elif method == "Polynomial":
        degree = int(rng.integers(2, 5))
        return PolynomialGenerator(n_samples=n_samples, n_features=n_features,
                                    n_classes=n_classes, degree=degree)
    elif method == "Regression":
        resp = str(rng.choice(["linear", "additive", "interaction"]))
        return RegressionSyntheticGenerator(
            n_samples=n_samples, n_features=n_features,
            response_type=resp, noise_std=float(rng.uniform(0.01, 0.5)),
        )
    elif method.startswith("MixedType_"):
        base = method.replace("MixedType_", "")
        base_gen = _build_generator(base, n_samples, n_features, n_classes, task_type, rng)
        return MixedTypeGenerator(base_gen)
    else:
        return TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)


def _apply_domain_names(df, meta, rng, domain):
    """Kept for reference but not used — domain names now stored in _source_meta."""
    pass


def _quality_gates(df, task_type):
    """Kept for reference but not used — quality logic is inlined in worker."""
    pass


# ── Shard builder ─────────────────────────────────────────────────────
def build_shard(shard_idx: int, base_seed: int) -> tuple[Path, int, dict]:
    """Generate one shard of ~ROWS_PER_SHARD rows."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / f"train-{shard_idx:05d}.parquet"

    collected = []
    total_rows = 0
    total_datasets = 0
    gate_fails = 0
    errors = 0
    gen_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    utility_aucs: list[float] = []
    seed_offset = base_seed

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        while total_rows < ROWS_PER_SHARD:
            remaining = ROWS_PER_SHARD - total_rows
            wave_size = min(300, max(30, remaining // 5000))
            futures = {}
            for _ in range(wave_size):
                seed_offset += 1
                f = executor.submit(_generate_one, {"seed": seed_offset})
                futures[f] = seed_offset

            for f in as_completed(futures):
                try:
                    result = f.result()
                except Exception:
                    errors += 1
                    continue

                if result is None:
                    errors += 1
                    continue

                if result["status"] == "gate_fail":
                    gate_fails += 1
                    continue

                if result["status"] == "error":
                    errors += 1
                    continue

                buf = io.BytesIO(result["parquet_bytes"])
                df = pd.read_parquet(buf)
                collected.append(df)
                total_rows += result["n_rows"]
                total_datasets += 1

                gen = result["generator"]
                gen_counts[gen] = gen_counts.get(gen, 0) + 1
                tt = result["task_type"]
                task_counts[tt] = task_counts.get(tt, 0) + 1
                dom = result["domain"]
                domain_counts[dom] = domain_counts.get(dom, 0) + 1
                if result.get("utility_auc", 0) > 0:
                    utility_aucs.append(result["utility_auc"])

                if total_rows >= ROWS_PER_SHARD:
                    break

    if not collected:
        raise RuntimeError(f"Shard {shard_idx}: 0 datasets generated")

    combined = pd.concat(collected, ignore_index=True)

    combined.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = out_path.stat().st_size / (1024 * 1024)

    stats = {
        "rows": len(combined),
        "datasets": total_datasets,
        "gate_fails": gate_fails,
        "errors": errors,
        "size_mb": round(size_mb, 1),
        "generators": gen_counts,
        "task_types": task_counts,
        "domains": domain_counts,
        "mean_utility_auc": round(float(np.mean(utility_aucs)), 4) if utility_aucs else 0.0,
    }

    del collected, combined
    gc.collect()

    return out_path, total_rows, stats


def disk_used_gb() -> float:
    total, used, free = shutil.disk_usage("/")
    return used / (1024**3)


def corpus_size_gb() -> float:
    total = 0
    if CORPUS_DIR.exists():
        for f in CORPUS_DIR.iterdir():
            if f.is_file():
                total += f.stat().st_size
    return total / (1024**3)


# ── Main loop ─────────────────────────────────────────────────────────
def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine starting shard from generation log (not local files, since we delete after upload)
    existing_local = sorted(CORPUS_DIR.glob("train-*.parquet"))
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            log_entries = [json.loads(l) for l in f if l.strip()]
        start_shard = max((e["shard_idx"] for e in log_entries), default=-1) + 1
        estimated_rows = sum(e.get("rows", 0) for e in log_entries)
    else:
        start_shard = len(existing_local)
        estimated_rows = start_shard * ROWS_PER_SHARD

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION → HUGGINGFACE" if HF_UPLOAD else "SYNTHETIC DATA GENERATION — LOCAL")
    print(f"Output: {CORPUS_DIR}")
    print(f"HF repo: {HF_REPO}" if HF_UPLOAD else "HF upload: disabled")
    print(f"Shard size: ~{ROWS_PER_SHARD:,} rows")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Disk limit: {MAX_DISK_GB} GB used")
    print(f"Starting shard: {start_shard} (~{estimated_rows:,} rows so far)")
    print(f"Current disk: {disk_used_gb():.1f} GB used")
    print(f"Corpus size: {corpus_size_gb():.2f} GB")
    print("=" * 70)

    total_rows = estimated_rows
    global_start = time.time()
    shard_idx = start_shard

    while True:
        if shard_idx % DISK_CHECK_INTERVAL == 0 or shard_idx == start_shard:
            used = disk_used_gb()
            if used >= MAX_DISK_GB:
                print(f"\nDisk limit reached: {used:.1f} GB >= {MAX_DISK_GB} GB")
                break

        shard_start = time.time()
        base_seed = shard_idx * 1_000_000

        print(f"\n{'='*60}")
        print(f"SHARD {shard_idx:05d} | Rows so far: {total_rows:,} | "
              f"Corpus: {corpus_size_gb():.1f} GB | Disk: {disk_used_gb():.1f} GB")

        try:
            path, n_rows, stats = build_shard(shard_idx, base_seed)
            gen_time = time.time() - shard_start

            total_rows += n_rows
            elapsed = time.time() - global_start
            rate = total_rows / max(elapsed, 1)

            print(f"  {n_rows:,} rows | {stats['datasets']} datasets | "
                  f"{stats['size_mb']:.0f} MB | {gen_time:.0f}s ({n_rows/gen_time:,.0f} rows/s)")
            print(f"  Generators: {stats['generators']}")
            print(f"  Tasks: {stats['task_types']}")
            print(f"  Domains: {dict(sorted(stats['domains'].items()))}")
            print(f"  Quality: mean_utility={stats['mean_utility_auc']:.4f} | "
                  f"gate_fails={stats['gate_fails']} | errors={stats['errors']}")
            print(f"  Overall: {total_rows:,} rows | {rate:,.0f} rows/s avg")

            with open(LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "shard_idx": shard_idx,
                    "n_rows": n_rows,
                    **stats,
                    "gen_time_s": round(gen_time, 1),
                    "total_rows": total_rows,
                    "corpus_gb": round(corpus_size_gb(), 2),
                }) + "\n")

            # Upload to HF and delete local copy
            if HF_UPLOAD:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=HF_TOKEN)
                    upload_start = time.time()
                    api.upload_file(
                        path_or_fileobj=str(path),
                        path_in_repo=f"data/{path.name}",
                        repo_id=HF_REPO,
                        repo_type="dataset",
                        commit_message=f"Add {path.name}",
                    )
                    upload_time = time.time() - upload_start
                    size_mb = path.stat().st_size / 1e6
                    print(f"  HF upload: {size_mb:.0f} MB in {upload_time:.0f}s | deleted local")
                    path.unlink()
                except Exception as ue:
                    print(f"  HF upload failed: {ue} — keeping local copy")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

        shard_idx += 1
        gc.collect()

    elapsed = time.time() - global_start
    manifest = {
        "total_rows": total_rows,
        "total_shards": shard_idx,
        "corpus_size_gb": round(corpus_size_gb(), 2),
        "elapsed_hours": round(elapsed / 3600, 2),
        "avg_rows_per_sec": round(total_rows / max(elapsed, 1), 0),
        "generators": METHODS_TIER1,
        "domains": DOMAINS,
        "task_types": list(set(TASK_TYPES)),
        "features_range": f"{MIN_FEATURES}-{MAX_FEATURES}",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print(f"Total: {total_rows:,} rows in {shard_idx} shards")
    print(f"Corpus: {corpus_size_gb():.1f} GB")
    print(f"Elapsed: {elapsed/3600:.1f} hours")
    print(f"Rate: {total_rows/elapsed:,.0f} rows/s")
    print("=" * 70)
    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload avewright/tabula-pretraining-corpus-v2 "
          f"{CORPUS_DIR} data --repo-type dataset")


if __name__ == "__main__":
    main()
