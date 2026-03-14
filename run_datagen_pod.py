#!/usr/bin/env python3
"""Optimized datagen loop for RunPod — RTX A4500 / 12 vCPU / 62 GB RAM.

Hardware-specific optimizations over run_datagen_v2.py:
  - Parallel synthetic generation across 12 vCPUs (ProcessPoolExecutor)
  - Larger bundles: 20-30 datasets per push (62 GB RAM headroom)
  - Bigger datasets: up to 200K rows per dataset (vs 100K)
  - GPU-accelerated Tier-2 generators: CTGAN / TVAE via RTX A4500
  - Concurrent upload: generate next batch while uploading previous
  - Batched quality gates with joblib parallelism on all 12 cores
  - Memory-efficient: explicit gc + torch.cuda.empty_cache between batches

Runs forever. Press Ctrl+C to stop.
Usage:
    source /workspace/venv/bin/activate
    cd /workspace/tabula
    python run_datagen_pod.py
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from tabula.data.env import load_repo_env_file
from tabula.data.synthetic import (
    TreePriorGenerator,
    GaussianMixtureGenerator,
    PolynomialGenerator,
    SCMGenerator,
    RegressionSyntheticGenerator,
    TimeSeriesSyntheticGenerator,
    MixedTypeGenerator,
    SyntheticDatasetMeta,
)

# ---------------------------------------------------------------------------
# Hardware-tuned constants
# ---------------------------------------------------------------------------
NUM_CPUS = min(os.cpu_count() or 4, 12)
# Leave 2 cores for upload / OS — use the rest for generation
GENERATION_WORKERS = max(1, NUM_CPUS - 2)
QUALITY_GATE_JOBS = NUM_CPUS  # sklearn n_jobs for RF cross-val

# Larger bundles — 62 GB RAM can handle 20-30 datasets easily
MIN_DATASETS_PER_BUNDLE = 20
MAX_DATASETS_PER_BUNDLE = 30

# Bigger individual datasets — RAM allows it
SAMPLE_SIZES = [2_000, 5_000, 10_000, 10_000, 50_000, 50_000, 100_000, 100_000, 200_000]

# Max total rows per HF push (avoid HF upload timeouts)
MAX_ROWS_PER_PUSH = 1_000_000

CORPUS_REPO = "avewright/tabula-pretraining-corpus"
LOG_FILE = Path("datagen_log.tsv")
LOG_COLUMNS = [
    "batch_id", "timestamp", "source_type", "method", "source_id",
    "n_datasets", "total_rows", "n_features_range", "utility_auc",
    "hub_repo", "status", "notes",
]

# ---------------------------------------------------------------------------
# Tier-2 GPU availability (lazy-detected)
# ---------------------------------------------------------------------------
_TIER2_AVAILABLE: dict[str, bool] = {}


def _check_tier2() -> dict[str, bool]:
    """Detect which Tier-2 GPU generators are importable."""
    global _TIER2_AVAILABLE
    if _TIER2_AVAILABLE:
        return _TIER2_AVAILABLE

    import importlib
    for pkg, key in [("ctgan", "CTGAN"), ("ctgan", "TVAE")]:
        try:
            importlib.import_module(pkg)
            _TIER2_AVAILABLE[key] = True
        except ImportError:
            _TIER2_AVAILABLE[key] = False

    # Check GPU availability
    try:
        import torch
        if not torch.cuda.is_available():
            # No GPU — disable all Tier-2
            for k in list(_TIER2_AVAILABLE.keys()):
                _TIER2_AVAILABLE[k] = False
    except ImportError:
        for k in list(_TIER2_AVAILABLE.keys()):
            _TIER2_AVAILABLE[k] = False

    return _TIER2_AVAILABLE


# ---------------------------------------------------------------------------
# DOMAIN_VOCAB — identical to v2 (imported in full for self-containment)
# ---------------------------------------------------------------------------
DOMAIN_VOCAB = {
    "finance": [
        "income", "age", "debt_ratio", "credit_score", "loan_amount",
        "employment_years", "balance", "expenses", "interest_rate", "default",
        "assets", "liabilities", "revenue", "margin", "tax_rate",
        "net_worth", "monthly_payment", "annual_income", "savings_rate", "insurance_premium",
        "mortgage_balance", "credit_limit", "utilization_ratio", "payment_history", "delinquency_count",
        "bankruptcy_flag", "collection_count", "inquiry_count", "account_age_months", "open_accounts",
        "closed_accounts", "revolving_balance", "installment_balance", "total_debt", "debt_to_income",
        "monthly_expenses", "rent_amount", "dividend_yield", "stock_return", "bond_rating",
        "portfolio_value", "investment_horizon", "risk_tolerance", "sharpe_ratio", "volatility",
        "market_cap", "price_earnings", "book_value", "earnings_per_share", "return_on_equity",
        "return_on_assets", "operating_margin", "gross_margin", "free_cash_flow", "enterprise_value",
        "ebitda", "current_ratio", "quick_ratio", "inventory_turnover", "receivables_turnover",
        "working_capital", "capital_expenditure", "depreciation", "amortization", "goodwill",
        "intangible_assets", "total_equity", "total_assets", "total_liabilities", "long_term_debt",
        "short_term_debt", "interest_coverage", "leverage_ratio", "payout_ratio", "beta",
        "alpha", "tracking_error", "information_ratio", "max_drawdown", "var_95",
        "expected_shortfall", "correlation", "covariance", "duration", "convexity",
        "yield_to_maturity", "coupon_rate", "face_value", "par_value", "accrued_interest",
        "bid_price", "ask_price", "spread", "volume_traded", "open_interest",
        "implied_volatility", "delta", "gamma", "theta", "vega",
        "strike_price", "expiry_days", "option_premium", "intrinsic_value", "time_value",
        "margin_requirement", "collateral_value", "haircut", "default_probability", "recovery_rate",
    ],
    "health": [
        "bmi", "age", "blood_pressure", "cholesterol", "glucose", "smoking",
        "alcohol_units", "exercise_days", "family_history", "diagnosis",
        "heart_rate", "creatinine", "hemoglobin", "white_cell_count", "fever",
        "systolic_bp", "diastolic_bp", "pulse_rate", "respiratory_rate", "oxygen_saturation",
        "temperature_c", "weight_kg", "height_cm", "waist_circumference", "hip_circumference",
        "body_fat_pct", "lean_mass", "bone_density", "vitamin_d", "calcium_level",
        "potassium_level", "sodium_level", "chloride_level", "bicarbonate", "blood_urea_nitrogen",
        "albumin", "bilirubin", "alkaline_phosphatase", "ast_level", "alt_level",
        "ggt_level", "ldh_level", "uric_acid", "triglycerides", "hdl_cholesterol",
        "ldl_cholesterol", "vldl_cholesterol", "total_protein", "globulin", "a1c_level",
        "fasting_glucose", "insulin_level", "homa_ir", "c_peptide", "cortisol",
        "thyroid_tsh", "free_t4", "free_t3", "prolactin", "testosterone",
        "estradiol", "progesterone", "fsh_level", "lh_level", "dhea_level",
        "iron_level", "ferritin", "transferrin", "tibc", "reticulocyte_count",
        "platelet_count", "red_cell_count", "mean_cell_volume", "mch", "mchc",
        "rdw", "mpv", "esr_rate", "crp_level", "procalcitonin",
        "d_dimer", "fibrinogen", "inr", "pt_time", "aptt",
        "troponin", "bnp_level", "ck_mb", "myoglobin", "lactate",
        "pco2", "po2", "ph_blood", "base_excess", "anion_gap",
        "gfr_estimated", "urine_protein", "urine_creatinine", "microalbumin", "specific_gravity",
        "wbc_differential", "neutrophil_pct", "lymphocyte_pct", "monocyte_pct", "eosinophil_pct",
    ],
    "ecommerce": [
        "price", "quantity", "discount", "return_rate", "category", "rating",
        "reviews", "shipping_days", "refund", "revenue", "cart_size",
        "session_minutes", "clicks", "conversion", "churn",
        "page_views", "bounce_rate", "exit_rate", "time_on_page", "scroll_depth",
        "add_to_cart_rate", "checkout_rate", "abandonment_rate", "avg_order_value", "customer_lifetime_value",
        "purchase_frequency", "recency_days", "monetary_value", "rfm_score", "loyalty_tier",
        "promotion_response", "coupon_used", "discount_pct", "shipping_cost", "handling_fee",
        "tax_amount", "total_amount", "payment_method_id", "installment_count", "credit_card_flag",
        "product_weight", "product_volume", "product_age_days", "inventory_level", "reorder_point",
        "stockout_days", "backorder_count", "supplier_lead_time", "cost_of_goods", "gross_profit",
        "margin_pct", "markup_pct", "seller_rating", "seller_age_months", "seller_volume",
        "competitor_price", "price_difference", "price_elasticity", "demand_forecast", "seasonal_index",
        "trend_component", "search_rank", "impression_count", "click_through_rate", "cost_per_click",
        "ad_spend", "roas", "organic_traffic_pct", "referral_traffic_pct", "social_traffic_pct",
        "email_open_rate", "email_click_rate", "unsubscribe_rate", "sms_response_rate", "push_notification_ctr",
        "app_install_source", "app_session_count", "app_crash_rate", "load_time_ms", "error_rate",
        "support_tickets", "resolution_time_hours", "csat_score", "nps_score", "review_sentiment",
        "wishlist_count", "comparison_count", "share_count", "gift_flag", "repeat_customer_flag",
        "cross_sell_count", "upsell_count", "bundle_discount", "membership_flag", "subscription_months",
        "free_trial_flag", "cancellation_reason_id", "win_back_offer", "reactivation_flag", "cohort_week",
    ],
    "iot": [
        "temperature", "humidity", "pressure", "vibration", "voltage", "current",
        "uptime_hours", "error_rate", "latency_ms", "throughput_gbps",
        "packet_loss", "cpu_pct", "mem_pct", "disk_iops", "fan_rpm",
        "power_watts", "energy_kwh", "frequency_hz", "resistance_ohm", "capacitance_uf",
        "signal_strength_dbm", "noise_floor_dbm", "snr_db", "bit_error_rate", "frame_error_rate",
        "retransmit_count", "jitter_ms", "round_trip_ms", "bandwidth_mbps", "connection_count",
        "active_sessions", "queue_depth", "buffer_occupancy", "cache_hit_ratio", "page_fault_rate",
        "context_switch_rate", "interrupt_rate", "io_wait_pct", "load_average_1m", "load_average_5m",
        "thread_count", "process_count", "file_descriptor_count", "socket_count", "tcp_retransmits",
        "dns_lookup_ms", "tls_handshake_ms", "http_request_count", "http_error_rate", "response_time_p50",
        "response_time_p95", "response_time_p99", "request_rate_rps", "concurrent_users", "error_count",
        "warning_count", "critical_alert_count", "uptime_pct", "mtbf_hours", "mttr_hours",
        "availability_pct", "reliability_score", "firmware_version_num", "patch_level", "config_drift_score",
        "sensor_accuracy", "calibration_offset", "measurement_noise", "sampling_rate_hz", "duty_cycle_pct",
        "motor_speed_rpm", "torque_nm", "acceleration_g", "displacement_mm", "velocity_mps",
        "flow_rate_lpm", "level_pct", "ph_value", "conductivity_us", "turbidity_ntu",
        "dissolved_oxygen", "co2_ppm", "particulate_matter", "radiation_usv", "magnetic_field_ut",
        "light_lux", "sound_db", "motion_detected", "door_open_count", "occupancy_count",
        "battery_voltage", "charge_pct", "solar_irradiance", "wind_speed_mps", "rain_mm",
        "ambient_temp_c", "dew_point_c", "heat_index", "uv_index", "barometric_trend",
    ],
    "hr": [
        "tenure_years", "salary", "performance_score", "overtime_hours",
        "absences", "promotions", "department_id", "role_level",
        "satisfaction", "attrition", "training_hours", "team_size",
        "remote_days", "bonus_pct", "peer_rating",
        "manager_rating", "self_rating", "goal_completion_pct", "project_count", "deadline_met_pct",
        "communication_score", "leadership_score", "technical_score", "collaboration_score", "initiative_score",
        "years_experience", "education_level", "certification_count", "skill_count", "language_count",
        "interview_score", "assessment_score", "background_check_flag", "reference_score", "offer_amount",
        "signing_bonus", "relocation_flag", "start_date_delay_days", "onboarding_score", "first_review_score",
        "probation_passed", "first_year_retention", "internal_transfer_count", "lateral_move_count", "demotion_flag",
        "grievance_count", "disciplinary_count", "warning_count", "suspension_days", "termination_risk_score",
        "engagement_score", "pulse_survey_score", "eNPS", "burnout_risk", "work_life_balance_score",
        "commute_minutes", "office_days_per_week", "meeting_hours_weekly", "email_volume_daily", "slack_messages_daily",
        "focus_time_hours", "interruption_count", "task_completion_rate", "velocity_points", "sprint_participation",
        "code_review_count", "bug_fix_count", "feature_delivery_count", "documentation_pages", "mentoring_hours",
        "hiring_referral_count", "diversity_index", "pay_equity_ratio", "market_ratio", "compa_ratio",
        "total_compensation", "equity_grant_value", "vesting_pct", "pto_days_used", "sick_days_used",
        "fmla_days", "parental_leave_weeks", "sabbatical_flag", "benefits_enrollment_score", "hsa_contribution",
        "retirement_contribution_pct", "employer_match_pct", "stock_purchase_pct", "wellness_points", "gym_visits_monthly",
        "health_risk_score", "ergonomic_assessment_score", "safety_incident_count", "near_miss_count", "training_compliance_pct",
        "expense_report_total", "travel_days_quarterly", "conference_attendance", "patent_count", "publication_count",
    ],
    "science": [
        "wavelength", "intensity", "mass", "velocity", "acceleration",
        "temperature_k", "pressure_pa", "volume", "concentration", "ph",
        "reaction_time", "yield_pct", "purity", "entropy", "energy_kj",
        "molar_mass", "density_gcm3", "viscosity_pa_s", "surface_tension", "refractive_index",
        "melting_point_k", "boiling_point_k", "heat_capacity", "thermal_conductivity", "diffusion_coefficient",
        "activation_energy", "rate_constant", "equilibrium_constant", "gibbs_free_energy", "enthalpy",
        "bond_length", "bond_angle", "dihedral_angle", "electronegativity", "ionization_energy",
        "electron_affinity", "atomic_radius", "covalent_radius", "van_der_waals_radius", "crystal_lattice",
        "band_gap_ev", "conductance", "resistivity", "permittivity", "permeability",
        "magnetic_moment", "spin_state", "orbital_energy", "photon_energy", "luminescence",
        "fluorescence_lifetime", "quantum_yield", "absorbance", "transmittance", "optical_density",
        "peak_wavelength", "fwhm", "spectral_resolution", "signal_to_noise", "baseline_offset",
        "retention_time", "peak_area", "peak_height", "plate_count", "resolution_factor",
        "selectivity_factor", "capacity_factor", "dead_volume", "flow_rate_mlmin", "column_pressure",
        "injection_volume", "sample_concentration", "dilution_factor", "recovery_pct", "rsd_pct",
        "lod_ppm", "loq_ppm", "linearity_r2", "intercept", "slope",
        "residual_error", "chi_squared", "p_value", "t_statistic", "f_statistic",
        "degrees_freedom", "confidence_interval", "standard_error", "effect_size", "power",
        "sample_size", "replicate_count", "block_count", "treatment_level", "control_mean",
        "treatment_mean", "fold_change", "log2_ratio", "normalized_expression", "z_score",
        "principal_component_1", "principal_component_2", "variance_explained", "silhouette_score", "cluster_id",
    ],
    "logistics": [
        "weight_kg", "distance_km", "transit_days", "on_time", "carrier_id",
        "cost_usd", "damage_rate", "warehouse_id", "load_factor",
        "fuel_consumption", "route_id", "customs_flag", "return_flag",
        "priority", "fragile",
        "package_count", "pallet_count", "container_count", "shipment_volume_m3", "shipment_weight_kg",
        "origin_lat", "origin_lon", "dest_lat", "dest_lon", "great_circle_km",
        "road_distance_km", "sea_distance_nm", "air_distance_km", "segments_count", "transshipment_count",
        "pickup_window_hours", "delivery_window_hours", "dwell_time_hours", "loading_time_min", "unloading_time_min",
        "dock_appointment_flag", "cross_dock_flag", "consolidation_flag", "deconsolidation_flag", "last_mile_flag",
        "temperature_controlled", "hazmat_flag", "oversized_flag", "stackable_flag", "insurance_value",
        "declared_value", "duty_amount", "tariff_code", "origin_country_id", "dest_country_id",
        "free_trade_zone", "bonded_warehouse", "inspection_flag", "quarantine_days", "documentation_complete",
        "bill_of_lading_type", "incoterm_id", "payment_term_days", "invoice_amount", "freight_charge",
        "surcharge_fuel", "surcharge_peak", "accessorial_charge", "total_charge", "cost_per_kg",
        "cost_per_km", "revenue_per_shipment", "margin_per_shipment", "utilization_pct", "fill_rate_pct",
        "order_accuracy_pct", "pick_error_rate", "cycle_count_accuracy", "inventory_turns", "days_on_hand",
        "safety_stock_days", "reorder_quantity", "lead_time_days", "supplier_reliability_pct", "defect_rate_ppm",
        "claim_count", "claim_amount", "pod_received_flag", "eta_variance_hours", "schedule_adherence_pct",
        "fleet_size", "vehicle_age_years", "maintenance_cost", "fuel_efficiency_kml", "co2_emission_kg",
        "driver_score", "hours_of_service_remaining", "rest_stop_count", "speed_violation_count", "incident_count",
    ],
    "education": [
        "gpa", "credits_completed", "credits_attempted", "attendance_rate", "assignment_score",
        "exam_score", "quiz_score", "lab_score", "project_score", "participation_score",
        "midterm_grade", "final_grade", "cumulative_gpa", "semester_gpa", "class_rank",
        "sat_math", "sat_verbal", "act_composite", "ap_courses_taken", "ap_average_score",
        "study_hours_weekly", "tutoring_sessions", "office_hours_visits", "library_visits_weekly", "online_login_count",
        "forum_posts", "assignment_submissions", "late_submissions", "missing_assignments", "extra_credit_points",
        "course_load", "major_courses", "elective_courses", "prerequisite_gpa", "department_avg_gpa",
        "class_size", "student_faculty_ratio", "instructor_rating", "course_difficulty", "course_satisfaction",
        "dropout_risk_score", "retention_probability", "graduation_probability", "time_to_degree_semesters", "transfer_credits",
        "financial_aid_amount", "scholarship_amount", "loan_amount_education", "work_study_hours", "family_income_bracket",
        "first_generation_flag", "international_flag", "distance_from_campus_km", "housing_type_id", "meal_plan_flag",
        "extracurricular_count", "sport_participation", "volunteer_hours", "leadership_positions", "research_hours",
        "internship_count", "career_center_visits", "job_offers_count", "starting_salary", "employer_satisfaction",
        "alumni_engagement_score", "recommendation_strength", "portfolio_score", "thesis_grade", "defense_score",
        "publication_count_student", "conference_presentations", "award_count", "honor_society_flag", "dean_list_count",
        "academic_probation_flag", "suspension_flag_edu", "conduct_violation_count", "accommodation_flag", "counseling_visits",
        "wellness_score_student", "stress_level", "sleep_hours", "exercise_frequency", "social_connection_score",
        "technology_proficiency", "digital_literacy_score", "typing_speed_wpm", "programming_skill_level", "data_literacy_score",
    ],
    "manufacturing": [
        "cycle_time_sec", "throughput_units", "yield_rate_pct", "defect_rate", "scrap_rate",
        "rework_rate", "first_pass_yield", "oee_pct", "availability_rate", "performance_rate",
        "quality_rate", "downtime_minutes", "changeover_time_min", "setup_time_min", "run_time_hours",
        "planned_production", "actual_production", "production_variance", "utilization_rate", "capacity_pct",
        "machine_age_years", "maintenance_interval_hours", "last_maintenance_days_ago", "failure_mode_id", "severity_score",
        "occurrence_score", "detection_score", "rpn_score", "mtbf_machine_hours", "mttr_machine_hours",
        "spare_parts_inventory", "parts_cost", "labor_cost_per_unit", "energy_cost_per_unit", "total_cost_per_unit",
        "batch_size", "lot_number_id", "raw_material_quality", "incoming_inspection_score", "supplier_score",
        "temperature_setpoint", "temperature_actual", "pressure_setpoint_mfg", "pressure_actual_mfg", "speed_setpoint_rpm",
        "speed_actual_rpm", "feed_rate", "tool_wear_pct", "tool_life_remaining", "surface_roughness",
        "dimension_tolerance", "position_tolerance", "roundness", "flatness", "parallelism",
        "perpendicularity", "concentricity", "runout", "hardness", "tensile_strength",
        "elongation_pct", "impact_resistance", "fatigue_life_cycles", "corrosion_rate", "coating_thickness",
        "moisture_content_pct", "particle_size_um", "viscosity_mfg", "ph_level_mfg", "color_delta_e",
        "weight_deviation_g", "volume_deviation_ml", "seal_integrity_score", "leak_test_result", "functional_test_score",
        "visual_inspection_score", "xray_inspection_score", "ultrasonic_test_score", "packaging_integrity", "label_accuracy",
        "warehouse_temperature_c", "warehouse_humidity_pct", "shelf_life_days", "expiry_risk_score", "traceability_score",
        "environmental_impact_score", "waste_generated_kg", "water_usage_liters", "power_consumption_kwh", "carbon_footprint_kg",
    ],
    "environment": [
        "air_temperature_c", "water_temperature_c", "soil_temperature_c", "relative_humidity_pct", "absolute_humidity",
        "dew_point_temperature", "wind_speed_ms", "wind_direction_deg", "wind_gust_ms", "atmospheric_pressure_hpa",
        "sea_level_pressure", "precipitation_mm", "snowfall_cm", "snow_depth_cm", "rainfall_intensity",
        "cloud_cover_pct", "visibility_km", "solar_radiation_wm2", "uv_index_env", "sunshine_hours",
        "evapotranspiration_mm", "soil_moisture_pct", "groundwater_level_m", "river_discharge_m3s", "lake_level_m",
        "ocean_salinity_psu", "ocean_ph", "dissolved_oxygen_mgl", "turbidity_env_ntu", "chlorophyll_a_ugl",
        "nitrogen_total_mgl", "phosphorus_total_mgl", "bod_mgl", "cod_mgl", "tss_mgl",
        "pm25_ugm3", "pm10_ugm3", "ozone_ppb", "no2_ppb", "so2_ppb",
        "co_ppm_env", "voc_ppb", "methane_ppb", "co2_ppm_env", "aqi_index",
        "noise_level_db", "light_pollution_index", "electromagnetic_field", "radon_bqm3", "asbestos_fibers",
        "lead_content_ppb", "mercury_content_ppb", "arsenic_content_ppb", "cadmium_content_ppb", "chromium_content_ppb",
        "pesticide_residue", "herbicide_residue", "microplastic_count", "coliform_count", "e_coli_count",
        "biodiversity_index", "species_richness", "species_evenness", "habitat_area_km2", "fragmentation_index",
        "ndvi_value", "evi_value", "lai_value", "canopy_cover_pct", "tree_density",
        "deforestation_rate", "erosion_rate_mm", "sediment_load", "wetland_area_pct", "impervious_surface_pct",
        "carbon_stock_tc", "sequestration_rate", "fire_risk_index", "drought_index", "flood_risk_score",
        "seismic_activity", "volcanic_alert_level", "tsunami_risk", "landslide_susceptibility", "permafrost_depth",
    ],
    "telecom": [
        "call_duration_sec", "call_count", "sms_count", "data_usage_mb", "monthly_charge",
        "contract_months", "tenure_months", "churn_flag", "customer_lifetime_months", "arpu",
        "plan_type_id", "data_plan_gb", "voice_plan_minutes", "roaming_usage_mb", "international_calls",
        "dropped_call_rate", "call_setup_time_ms", "voice_quality_mos", "data_throughput_mbps", "download_speed_mbps",
        "upload_speed_mbps", "ping_ms", "jitter_ms_telecom", "packet_loss_pct", "availability_pct_telecom",
        "coverage_score", "signal_strength_bars", "handover_count", "cell_tower_distance_m", "frequency_band",
        "technology_generation", "device_age_months", "device_price", "screen_time_hours", "app_usage_hours",
        "streaming_hours", "gaming_hours", "social_media_hours", "browsing_hours", "email_data_mb",
        "complaint_count", "support_call_count", "resolution_time_hours_telecom", "csat_score_telecom", "nps_score_telecom",
        "billing_issue_count", "payment_delay_days", "auto_pay_flag", "paperless_billing_flag", "bundle_count",
        "add_on_count", "upgrade_count", "downgrade_count", "win_back_count", "referral_count",
        "loyalty_points", "reward_redemption_count", "promotion_count", "discount_total", "credit_total",
        "overage_charge", "late_fee", "activation_fee", "device_installment", "trade_in_value",
        "family_plan_members", "shared_data_pct", "hotspot_usage_mb", "wifi_calling_pct", "volte_pct",
        "tower_load_pct", "backhaul_utilization", "spectrum_efficiency", "energy_per_bit", "site_rental_cost",
        "maintenance_cost_telecom", "capex_per_subscriber", "opex_per_subscriber", "revenue_per_tower", "margin_per_subscriber",
    ],
}

# Method rotation — includes Tier-2 GPU methods
SYNTHETIC_METHODS_TIER1 = [
    "TreePrior",
    "SCM",
    "GaussianMixture",
    "Polynomial",
    "Regression",
    "MixedType_TreePrior",
    "MixedType_SCM",
    "MixedType_GaussianMixture",
]

SYNTHETIC_METHODS_TIER2 = [
    "CTGAN",
    "TVAE",
]

REAL_SOURCES = ["pmlb", "openml", "huggingface", "kaggle"]

# Domain classifier keywords
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "finance": ["income", "loan", "credit", "debt", "bank", "stock", "price",
                "mortgage", "insurance", "portfolio", "interest", "tax", "revenue",
                "profit", "salary", "wage", "financial", "money", "invest", "asset"],
    "health": ["bmi", "blood", "cholesterol", "glucose", "diagnosis", "patient",
               "medical", "hospital", "disease", "symptom", "drug", "treatment",
               "clinical", "heart", "cancer", "health", "mortality", "survival"],
    "ecommerce": ["price", "cart", "purchase", "customer", "product", "order",
                  "review", "rating", "shop", "store", "retail", "ecommerce",
                  "conversion", "churn", "subscription", "sell", "buy"],
    "iot": ["sensor", "temperature", "humidity", "voltage", "current", "vibration",
            "device", "signal", "iot", "telemetry", "firmware", "uptime", "latency"],
    "hr": ["employee", "attrition", "tenure", "satisfaction", "performance",
           "promotion", "department", "hiring", "turnover", "workforce", "hr",
           "staff", "recruit", "engagement"],
    "science": ["experiment", "species", "specimen", "chemical", "compound",
                "reaction", "wavelength", "isotope", "gene", "protein", "molecule",
                "scientific", "biology", "physics", "chemistry", "ecology"],
    "logistics": ["shipment", "delivery", "warehouse", "route", "freight",
                  "cargo", "transport", "logistics", "supply", "fleet", "tracking"],
    "education": ["student", "grade", "gpa", "course", "school", "university",
                  "education", "exam", "score", "academic", "teacher", "class"],
    "manufacturing": ["defect", "yield", "machine", "assembly", "quality",
                      "production", "factory", "manufacturing", "cycle_time",
                      "inspection", "tolerance", "batch"],
    "environment": ["climate", "weather", "pollution", "emission", "ozone",
                    "rainfall", "forest", "water", "air_quality", "environmental",
                    "soil", "ecosystem", "carbon", "biodiversity"],
    "telecom": ["call", "subscriber", "roaming", "bandwidth", "network",
                "telecom", "cellular", "mobile", "broadband", "spectrum"],
}


def classify_domain(df: pd.DataFrame, dataset_name: str = "") -> str:
    text_blob = " ".join(df.columns.str.lower().tolist())
    text_blob += " " + dataset_name.lower().replace("/", " ").replace("-", " ").replace("_", " ")
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_blob)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] > 0 else "unknown"


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------
class _TimeoutError(Exception):
    pass


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=180):
    kwargs = kwargs or {}
    result = [None]
    error = [None]
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = str(e)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    if t.is_alive():
        return None, f"Timeout after {timeout_seconds}s"
    if error[0]:
        return None, error[0]
    return result[0], None


# ---------------------------------------------------------------------------
# Env / Auth
# ---------------------------------------------------------------------------
def load_hf_token() -> str:
    env = load_repo_env_file()
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN not found in .env or environment")
    return token


# ---------------------------------------------------------------------------
# Log management
# ---------------------------------------------------------------------------
def init_log():
    if not LOG_FILE.exists():
        LOG_FILE.write_text("\t".join(LOG_COLUMNS) + "\n", encoding="utf-8")


def read_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_FILE, sep="\t", dtype=str, keep_default_na=False)


def append_log_row(row: dict):
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row.get(c, "") for c in LOG_COLUMNS])


def get_already_ingested() -> set[str]:
    log = read_log()
    if log.empty:
        return set()
    success = log[log["status"] == "success"]
    result = set()
    for sid in success["source_id"].tolist():
        for part in str(sid).split("|"):
            part = part.strip()
            if part:
                result.add(part)
    return result


def next_batch_id(log: pd.DataFrame) -> str:
    existing_nums = set()
    for bid in log["batch_id"]:
        try:
            existing_nums.add(int(bid.split("_")[1]))
        except (IndexError, ValueError):
            pass
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for item in api.list_repo_tree(CORPUS_REPO, repo_type="dataset"):
            name = getattr(item, "path", "") or getattr(item, "rfilename", "") or str(item)
            if "datagen_" in name:
                try:
                    num_str = name.split("datagen_")[1].split("/")[0]
                    existing_nums.add(int(num_str))
                except (IndexError, ValueError):
                    pass
    except Exception:
        pass
    n = max(existing_nums) + 1 if existing_nums else 1
    return f"datagen_{n:03d}"


# ---------------------------------------------------------------------------
# Quality gates — parallelized RF on all 12 cores
# ---------------------------------------------------------------------------
def run_quality_gates(df: pd.DataFrame, task_type: str) -> tuple[bool, float, str]:
    notes = []
    target = df["target"]
    features = df.drop(columns=["target", "_source_meta"], errors="ignore")
    numeric_features = features.select_dtypes(include=[np.number])

    # HARD gates
    for col in numeric_features.columns:
        if numeric_features[col].nunique() <= 1:
            return False, 0.0, f"gate_fail: constant column {col}"

    for col in features.columns:
        if features[col].isna().all():
            return False, 0.0, f"gate_fail: all-null column {col}"

    if task_type in ("binary", "multiclass"):
        if target.nunique() < 2:
            return False, 0.0, "gate_fail: target has < 2 classes"
        class_counts = target.value_counts()
        minority_frac = class_counts.min() / len(target)
        if minority_frac < 0.01:
            return False, 0.0, f"gate_fail: minority_frac={minority_frac:.4f}"

    dupe_frac = df.duplicated().sum() / len(df)
    if dupe_frac >= 0.50:
        return False, 0.0, f"gate_fail: dupe_frac={dupe_frac:.2f}"

    if len(df) < 100:
        return False, 0.0, f"gate_fail: only {len(df)} rows"

    # SOFT: Quick RF utility — leverage all CPUs
    utility_auc = 0.0
    try:
        if len(df) >= 30:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import OrdinalEncoder

            X_num = numeric_features.values if numeric_features.shape[1] >= 1 else np.empty((len(df), 0))
            cat_features = features.select_dtypes(include=["object", "string", "category"])
            if cat_features.shape[1] > 0:
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_cat = oe.fit_transform(cat_features.fillna("__NA__"))
                X = np.hstack([X_num, X_cat])
            else:
                X = X_num
            if X.shape[1] < 1:
                notes.append("SOFT_WARN: no usable features")
                return True, utility_auc, "; ".join(notes)

            y = target.values
            X = np.nan_to_num(X, nan=0.0)

            # Use all CPUs for RF and cross-val
            if task_type == "regression":
                clf = RandomForestRegressor(
                    n_estimators=50, max_depth=5, random_state=0,
                    n_jobs=QUALITY_GATE_JOBS,
                )
                scores = cross_val_score(clf, X, y, cv=3, scoring="r2",
                                         n_jobs=min(3, QUALITY_GATE_JOBS))
                utility_auc = float(scores.mean())
                if utility_auc <= 0.0:
                    notes.append(f"SOFT_WARN: R2={utility_auc:.3f}")
            else:
                clf = RandomForestClassifier(
                    n_estimators=50, max_depth=5, random_state=0,
                    n_jobs=QUALITY_GATE_JOBS,
                )
                n_classes = target.nunique()
                scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
                try:
                    scores = cross_val_score(clf, X, y, cv=3, scoring=scoring,
                                             n_jobs=min(3, QUALITY_GATE_JOBS))
                    utility_auc = float(scores.mean())
                except Exception:
                    scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy",
                                             n_jobs=min(3, QUALITY_GATE_JOBS))
                    utility_auc = float(scores.mean())
                if utility_auc < 0.55:
                    notes.append(f"SOFT_WARN: AUC={utility_auc:.3f}")
    except Exception as e:
        notes.append(f"SOFT_WARN: utility_check_error: {e}")

    # SOFT: Max pairwise correlation
    try:
        if numeric_features.shape[1] >= 2:
            cols_to_check = numeric_features.columns
            if len(cols_to_check) > 50:
                cols_to_check = numeric_features.columns[
                    np.random.choice(len(numeric_features.columns), 50, replace=False)
                ]
            corr_matrix = numeric_features[cols_to_check].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr = corr_matrix.max().max()
            if max_corr > 0.99:
                notes.append(f"SOFT_WARN: max_corr={max_corr:.3f}")
    except Exception:
        pass

    return True, utility_auc, "; ".join(notes)


# ---------------------------------------------------------------------------
# Target normalization helpers (from v2)
# ---------------------------------------------------------------------------
def _normalize_target_series(target: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(target) or target.dtype == object or str(target.dtype) == "category":
        from sklearn.preprocessing import LabelEncoder
        non_null = target.notna()
        if not non_null.any():
            return target
        le = LabelEncoder()
        values = le.fit_transform(target.loc[non_null].astype(str))
        result = pd.array([np.nan] * len(target), dtype="Float64")
        result[non_null.values] = values
        return pd.Series(result, index=target.index, name=target.name)
    return target


def _infer_task_type_from_target(target: pd.Series) -> str:
    non_null = target.dropna()
    if non_null.empty:
        return "regression"
    if pd.api.types.is_string_dtype(target) or target.dtype == object or str(target.dtype) == "category":
        n_classes = non_null.astype(str).nunique()
        return "binary" if n_classes == 2 else "multiclass"
    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        n_classes = non_null.astype(str).nunique()
        return "binary" if n_classes == 2 else "multiclass"
    unique_count = numeric.nunique()
    if unique_count <= 20:
        return "binary" if unique_count == 2 else "multiclass"
    return "regression"


_TARGET_CANDIDATES = [
    "target", "class", "label", "y", "outcome", "result",
    "diagnosis", "default", "survived", "income", "attrition",
    "churn", "species", "quality", "grade", "score",
    "status", "type", "category", "response",
]


def _rank_target_candidates(df: pd.DataFrame) -> list[str]:
    scores: dict[str, float] = {}
    lc = {c.lower(): c for c in df.columns}
    for i, name in enumerate(_TARGET_CANDIDATES):
        if name in lc:
            scores[lc[name]] = 100 - i
    last_col = df.columns[-1]
    scores.setdefault(last_col, 0)
    scores[last_col] += 10
    for col in df.columns:
        nunique = df[col].nunique()
        ratio = nunique / max(len(df), 1)
        if 2 <= nunique <= 20:
            scores.setdefault(col, 0)
            scores[col] += 5
        elif ratio < 0.05 and nunique >= 2:
            scores.setdefault(col, 0)
            scores[col] += 3
    return sorted(scores, key=lambda c: scores[c], reverse=True)


def normalize_real_dataset(
    df: pd.DataFrame,
    *,
    preferred_target: str | None = None,
    fallback_candidates: list[str] | None = None,
    source_label: str = "dataset",
) -> tuple[pd.DataFrame | None, dict[str, str] | None]:
    candidate_order: list[str] = []
    if preferred_target and preferred_target in df.columns:
        candidate_order.append(preferred_target)
    for candidate in fallback_candidates or []:
        if candidate in df.columns and candidate not in candidate_order:
            candidate_order.append(candidate)
    if "target" in df.columns and "target" not in candidate_order:
        candidate_order.append("target")
    if not candidate_order:
        ranked = _rank_target_candidates(df)
        candidate_order = ranked
    if not candidate_order:
        return None, {"reason": f"{source_label}: could not determine target column"}
    target_col = candidate_order[0]
    normalized = df.copy()
    if target_col != "target":
        normalized = normalized.rename(columns={target_col: "target"})
    normalized["target"] = _normalize_target_series(normalized["target"])
    task_type = _infer_task_type_from_target(normalized["target"])
    return normalized, {"target_column": target_col, "task_type": task_type}


# ---------------------------------------------------------------------------
# Domain naming
# ---------------------------------------------------------------------------
def _apply_domain_names_targeted(
    df: pd.DataFrame,
    meta: SyntheticDatasetMeta,
    rng: np.random.Generator,
    target_domain: str,
) -> tuple[pd.DataFrame, str]:
    vocab = list(DOMAIN_VOCAB[target_domain])
    rng.shuffle(vocab)
    feature_cols = [c for c in df.columns if c not in ("target", "_source_meta")]
    n_features = len(feature_cols)

    name_pool = vocab[:n_features]
    if len(name_pool) < n_features:
        all_names = set(name_pool)
        domains = list(DOMAIN_VOCAB.keys())
        rng.shuffle(domains)
        for d in domains:
            if d == target_domain:
                continue
            extras = list(DOMAIN_VOCAB[d])
            rng.shuffle(extras)
            for name in extras:
                if name not in all_names:
                    name_pool.append(name)
                    all_names.add(name)
                if len(name_pool) >= n_features:
                    break
            if len(name_pool) >= n_features:
                break
        while len(name_pool) < n_features:
            name_pool.append(f"measurement_{len(name_pool)}")

    name_mapping = {old: name_pool[i] for i, old in enumerate(feature_cols)}
    df = df.rename(columns=name_mapping)
    return df, target_domain


# ---------------------------------------------------------------------------
# Generator builders
# ---------------------------------------------------------------------------
def _build_generator(method: str, n_samples: int, n_features: int,
                     n_classes: int, task_type: str, rng: np.random.Generator):
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
        return PolynomialGenerator(
            n_samples=n_samples, n_features=n_features,
            n_classes=n_classes, degree=degree,
        )
    elif method == "Regression":
        resp = str(rng.choice(["linear", "additive", "interaction"]))
        return RegressionSyntheticGenerator(
            n_samples=n_samples, n_features=n_features,
            response_type=resp, noise_std=float(rng.uniform(0.01, 0.5)),
        )
    elif method == "TimeSeries":
        return TimeSeriesSyntheticGenerator(
            n_series=min(n_samples, 2000),
            series_length=int(rng.integers(32, 128)),
            task_type=task_type if task_type != "multiclass" else "binary",
            n_classes=n_classes,
        )
    elif method.startswith("MixedType_"):
        base_method = method.replace("MixedType_", "")
        base_gen = _build_generator(base_method, n_samples, n_features, n_classes, task_type, rng)
        return MixedTypeGenerator(base_gen)
    else:
        return TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)


def _citation_key_for_method(method: str) -> str:
    mapping = {
        "TreePrior": "hollmann2023tabpfn",
        "SCM": "scholkopf2021causal",
        "GaussianMixture": "tabula_internal",
        "Polynomial": "tabula_internal",
        "Regression": "tabula_internal",
        "TimeSeries": "tabula_internal",
        "MixedType_TreePrior": "hollmann2023tabpfn",
        "MixedType_SCM": "scholkopf2021causal",
        "MixedType_GaussianMixture": "tabula_internal",
        "CTGAN": "xu2019ctgan",
        "TVAE": "xu2019ctgan",
    }
    return mapping.get(method, "tabula_internal")


# ---------------------------------------------------------------------------
# Tier-2 GPU generators (CTGAN / TVAE on RTX A4500)
# ---------------------------------------------------------------------------
def _generate_tier2_dataset(
    method: str, n_samples: int, n_features: int, n_classes: int,
    task_type: str, seed: int,
) -> tuple[pd.DataFrame | None, SyntheticDatasetMeta | None, str | None]:
    """Generate a single dataset using a GPU-accelerated Tier-2 method.

    Strategy: generate a small 'template' via Tier-1 TreePrior, then fit
    CTGAN/TVAE on it and sample n_samples synthetic rows. This tricks the
    neural model into learning realistic covariance structure from our prior.
    """
    try:
        rng = np.random.default_rng(seed)
        # 1. Generate template data with TreePrior (fast)
        template_n = min(5000, n_samples)
        template_gen = TreePriorGenerator(
            n_samples=template_n, n_features=n_features, n_classes=n_classes,
        )
        template_df, tmeta = template_gen.generate(seed=seed)

        # Rename columns to avoid generic names during fitting
        feat_cols = [c for c in template_df.columns if c != "target"]
        col_map = {c: f"f{i}" for i, c in enumerate(feat_cols)}
        rev_map = {v: k for k, v in col_map.items()}
        template_df = template_df.rename(columns=col_map)
        fit_cols = list(col_map.values())

        # 2. Fit the neural model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if method == "CTGAN":
            from ctgan import CTGAN
            model = CTGAN(
                epochs=50, batch_size=min(500, template_n),
                verbose=False, cuda=(device == "cuda"),
            )
        elif method == "TVAE":
            from ctgan import TVAE
            model = TVAE(
                epochs=50, batch_size=min(500, template_n),
                verbose=False, cuda=(device == "cuda"),
            )
        else:
            return None, None, f"Unknown Tier-2 method: {method}"

        train_data = template_df[fit_cols + ["target"]]
        discrete_cols = ["target"] if task_type != "regression" else []
        model.fit(train_data, discrete_columns=discrete_cols)

        # 3. Sample
        synth_df = model.sample(n_samples)

        # Restore original column names
        synth_df = synth_df.rename(columns=rev_map)

        # Clean up GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        meta = SyntheticDatasetMeta(
            generator=method,
            n_samples=len(synth_df),
            n_features=len(feat_cols),
            task_type=task_type,
            n_classes=n_classes if task_type != "regression" else None,
            feature_names=feat_cols,
            target_name="target",
            extra={"template_n": template_n, "epochs": 50},
        )
        return synth_df, meta, None

    except Exception as e:
        return None, None, str(e)


# ---------------------------------------------------------------------------
# Single-dataset generation worker (for ProcessPoolExecutor)
# ---------------------------------------------------------------------------
def _generate_one_dataset(args: dict) -> dict | None:
    """Generate a single synthetic dataset. Designed for parallel execution.

    Returns a dict with 'df_bytes' (serialized parquet), 'meta', etc.
    We serialize to parquet bytes to avoid pickling large DataFrames across processes.
    """
    try:
        method = args["method"]
        seed = args["seed"]
        n_samples = args["n_samples"]
        n_features = args["n_features"]
        n_classes = args["n_classes"]
        task_type = args["task_type"]
        target_domain = args["target_domain"]
        batch_id = args["batch_id"]
        dataset_idx = args["dataset_idx"]
        missingness_rate = args["missingness_rate"]
        apply_drift = args["apply_drift"]

        rng = np.random.default_rng(seed)

        # Tier-2 methods handled separately (GPU, can't parallelize across processes)
        if method in ("CTGAN", "TVAE"):
            return None  # Handled in main thread

        generator = _build_generator(method, n_samples, n_features, n_classes, task_type, rng)
        df, meta = generator.generate(seed=seed)

        # Apply domain naming
        df, domain = _apply_domain_names_targeted(df, meta, rng, target_domain=target_domain)

        # Inject MCAR missingness (vectorized)
        if missingness_rate > 0.01:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "target"]
            if numeric_cols:
                mask = rng.random(size=(len(df), len(numeric_cols))) < missingness_rate
                vals = df[numeric_cols].values.astype(float)
                vals[mask] = np.nan
                df[numeric_cols] = vals

        # Inject concept drift
        if apply_drift and len(df) >= 200:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "target"]
            if numeric_cols:
                mid = len(df) // 2
                shift = rng.normal(0, 1.0, size=len(numeric_cols))
                vals = df[numeric_cols].values.astype(float)
                vals[mid:] += shift
                df[numeric_cols] = vals

        source_meta = {
            "batch_id": batch_id,
            "source_type": "synthetic",
            "source_id": f"synthetic:{method}",
            "domain": domain,
            "task_type": meta.task_type,
            "license": "Apache-2.0",
            "citation_key": _citation_key_for_method(method),
            "generator": meta.generator,
            "n_samples": meta.n_samples,
            "n_features": meta.n_features,
            "dataset_idx": dataset_idx,
            "seed": seed,
            "missingness_rate": missingness_rate,
            "concept_drift": apply_drift,
        }

        df["_source_meta"] = json.dumps(source_meta)

        # Quality gates
        passed, utility_auc, gate_notes = run_quality_gates(df, meta.task_type)
        if not passed:
            return {"status": "gate_fail", "idx": dataset_idx, "notes": gate_notes}

        source_meta["utility_auc"] = utility_auc
        df["_source_meta"] = json.dumps(source_meta)

        # Serialize to parquet bytes to avoid pickling large DataFrames
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return {
            "status": "ok",
            "idx": dataset_idx,
            "parquet_bytes": buf.getvalue(),
            "meta": source_meta,
            "n_rows": len(df),
            "n_features": meta.n_features,
            "domain": domain,
            "utility_auc": utility_auc,
            "generator": meta.generator,
            "task_type": meta.task_type,
        }
    except Exception as e:
        return {"status": "error", "idx": args.get("dataset_idx", -1), "notes": str(e)[:300]}


# ---------------------------------------------------------------------------
# Parallel synthetic generation
# ---------------------------------------------------------------------------
def generate_synthetic_bundle_parallel(
    method: str, batch_id: str, rng: np.random.Generator,
) -> tuple[list[pd.DataFrame], list[dict]]:
    """Generate 20-30 datasets in parallel using ProcessPoolExecutor."""
    n_datasets = int(rng.integers(MIN_DATASETS_PER_BUNDLE, MAX_DATASETS_PER_BUNDLE + 1))
    domains = list(DOMAIN_VOCAB.keys())
    dataset_domains = [domains[i % len(domains)] for i in range(n_datasets)]
    rng.shuffle(dataset_domains)

    # Prepare work items
    work_items = []
    for i in range(n_datasets):
        seed = int(rng.integers(0, 2**31))
        n_samples = int(rng.choice(SAMPLE_SIZES))
        max_features = len(DOMAIN_VOCAB[dataset_domains[i]])
        n_features = int(rng.integers(3, min(129, max_features + 1)))
        task_type_choice = str(rng.choice(["binary", "multiclass", "regression"], p=[0.5, 0.3, 0.2]))

        if task_type_choice == "binary":
            n_classes = 2
        elif task_type_choice == "multiclass":
            n_classes = int(rng.choice([3, 4, 5, 10]))
            max_safe_classes = max(2, n_samples // 100)
            n_classes = min(n_classes, max_safe_classes)
        else:
            n_classes = 2

        missingness_rate = float(rng.uniform(0.0, 0.15))
        apply_drift = float(rng.random()) < 0.2

        work_items.append({
            "method": method,
            "seed": seed,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "task_type": task_type_choice,
            "target_domain": dataset_domains[i],
            "batch_id": batch_id,
            "dataset_idx": i,
            "missingness_rate": missingness_rate,
            "apply_drift": apply_drift,
        })

    # Tier-2 GPU methods — run sequentially on GPU
    if method in ("CTGAN", "TVAE"):
        return _generate_tier2_bundle(method, work_items, batch_id)

    # Tier-1 methods — parallel across CPUs
    dfs = []
    metas = []
    ok_count = 0
    fail_count = 0

    print(f"  Generating {n_datasets} datasets with {GENERATION_WORKERS} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=GENERATION_WORKERS) as executor:
        futures = {executor.submit(_generate_one_dataset, item): item for item in work_items}

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                fail_count += 1
                continue
            if result["status"] == "ok":
                # Deserialize parquet bytes back to DataFrame
                import io
                buf = io.BytesIO(result["parquet_bytes"])
                df = pd.read_parquet(buf)
                dfs.append(df)
                metas.append(result["meta"])
                ok_count += 1
                print(f"    [{ok_count}/{n_datasets}] {result['generator']} "
                      f"n={result['n_rows']} d={result['n_features']} "
                      f"task={result['task_type']} domain={result['domain']} "
                      f"auc={result['utility_auc']:.3f}", flush=True)
            else:
                fail_count += 1
                print(f"    [SKIP] dataset {result['idx']}: {result.get('notes', '')}", flush=True)

    print(f"  Bundle complete: {ok_count} ok, {fail_count} failed", flush=True)
    return dfs, metas


def _generate_tier2_bundle(
    method: str, work_items: list[dict], batch_id: str,
) -> tuple[list[pd.DataFrame], list[dict]]:
    """Generate a Tier-2 bundle sequentially on GPU."""
    dfs = []
    metas = []
    ok_count = 0
    n_total = len(work_items)

    print(f"  Generating {n_total} Tier-2 ({method}) datasets on GPU...", flush=True)

    for item in work_items:
        df, meta, err = _generate_tier2_dataset(
            method=method,
            n_samples=item["n_samples"],
            n_features=item["n_features"],
            n_classes=item["n_classes"],
            task_type=item["task_type"],
            seed=item["seed"],
        )
        if err:
            print(f"    [ERR] dataset {item['dataset_idx']}: {err[:200]}", flush=True)
            continue
        if df is None or meta is None:
            continue

        rng = np.random.default_rng(item["seed"])
        df, domain = _apply_domain_names_targeted(df, meta, rng, target_domain=item["target_domain"])

        # Missingness
        if item["missingness_rate"] > 0.01:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "target"]
            if numeric_cols:
                mask = rng.random(size=(len(df), len(numeric_cols))) < item["missingness_rate"]
                vals = df[numeric_cols].values.astype(float)
                vals[mask] = np.nan
                df[numeric_cols] = vals

        source_meta = {
            "batch_id": batch_id,
            "source_type": "synthetic",
            "source_id": f"synthetic:{method}",
            "domain": domain,
            "task_type": meta.task_type,
            "license": "Apache-2.0",
            "citation_key": _citation_key_for_method(method),
            "generator": meta.generator,
            "n_samples": meta.n_samples,
            "n_features": meta.n_features,
            "dataset_idx": item["dataset_idx"],
            "seed": item["seed"],
            "missingness_rate": item["missingness_rate"],
            "concept_drift": False,
        }
        df["_source_meta"] = json.dumps(source_meta)

        passed, utility_auc, gate_notes = run_quality_gates(df, meta.task_type)
        if not passed:
            print(f"    [SKIP] dataset {item['dataset_idx']}: {gate_notes}", flush=True)
            continue

        source_meta["utility_auc"] = utility_auc
        df["_source_meta"] = json.dumps(source_meta)

        dfs.append(df)
        metas.append(source_meta)
        ok_count += 1
        print(f"    [{ok_count}/{n_total}] {method} n={len(df)} d={meta.n_features} "
              f"domain={domain} auc={utility_auc:.3f}", flush=True)

        # Free GPU memory between datasets
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return dfs, metas


# ---------------------------------------------------------------------------
# Real dataset ingestion (same as v2 — OpenML, PMLB, HF, Kaggle)
# ---------------------------------------------------------------------------
def discover_openml_batch(already_ingested: set[str], batch_size: int = 8) -> list[dict]:
    from tabula.data.openml import search_openml_datasets
    candidates = []
    for offset in range(0, 300, 50):
        if len(candidates) >= batch_size * 3:
            break
        try:
            results = search_openml_datasets(
                min_instances=100, max_instances=500000,
                max_features=200, limit=50, offset=offset,
            )
            for r in results:
                sid = f"openml:{r.dataset_id}"
                if sid not in already_ingested:
                    candidates.append({
                        "dataset_id": r.dataset_id,
                        "name": r.name,
                        "source_id": sid,
                        "n_instances": r.n_instances,
                        "n_features": r.n_features,
                        "n_classes": r.n_classes,
                    })
        except Exception as e:
            print(f"  OpenML search error at offset={offset}: {e}")
    rng = np.random.default_rng()
    rng.shuffle(candidates)
    return candidates[:batch_size]


def ingest_openml_dataset(info: dict) -> tuple[pd.DataFrame | None, dict | None]:
    from tabula.data.openml import fetch_openml_dataset
    import tempfile
    dataset_id = info["dataset_id"]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = fetch_openml_dataset(
                dataset_id=dataset_id, output_root=tmpdir, max_rows=100000,
            )
            csv_path = raw_dir / "train.csv"
            if not csv_path.exists():
                return None, None
            df = pd.read_csv(csv_path)
            manifest_path = raw_dir / "dataset_manifest.json"
            preferred_target = None
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                preferred_target = manifest.get("target_column")
        if len(df) < 100:
            return None, None
        normalized, inferred = normalize_real_dataset(
            df, preferred_target=preferred_target,
            fallback_candidates=["class", "target", "label", "y"],
            source_label=f"openml:{dataset_id}",
        )
        if normalized is None or inferred is None:
            return None, None
        df = normalized
        task_type = inferred["task_type"]
        citation = _fetch_openml_citation(dataset_id)
        domain = classify_domain(df, info.get("name", ""))
        source_meta = {
            "source_type": "real",
            "source_id": f"openml:{dataset_id}",
            "domain": domain,
            "task_type": task_type,
            "license": "CC-BY 4.0",
            "citation_key": f"openml_{dataset_id}",
            "original_source": f"https://openml.org/d/{dataset_id}",
            "citation": citation,
            "name": info.get("name", ""),
            "original_target_column": inferred["target_column"],
        }
        return df, source_meta
    except Exception as e:
        print(f"  [ERR] openml:{dataset_id}: {e}")
        return None, None


def _fetch_openml_citation(dataset_id: int) -> str:
    try:
        import urllib.request
        url = f"https://api.openml.org/json/data/{dataset_id}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ds = data.get("data_set_description", {})
        citation = ds.get("citation", "")
        paper_url = ds.get("paper_url", "")
        if citation:
            return citation
        if paper_url:
            return f"@misc{{openml_{dataset_id}, url={{{paper_url}}}}}"
        return f"@misc{{openml_{dataset_id}, url={{https://openml.org/d/{dataset_id}}}}}"
    except Exception:
        return f"@misc{{openml_{dataset_id}, url={{https://openml.org/d/{dataset_id}}}}}"


def discover_pmlb_batch(already_ingested: set[str], batch_size: int = 8) -> list[dict]:
    from tabula.data.pmlb import search_pmlb_datasets
    candidates = []
    for task in ["classification", "regression"]:
        try:
            results = search_pmlb_datasets(
                task=task, min_instances=100, max_instances=100000, max_features=200,
            )
            for r in results:
                sid = f"pmlb:{r.name}"
                if sid not in already_ingested:
                    candidates.append({
                        "name": r.name,
                        "source_id": sid,
                        "task": task,
                        "n_instances": r.n_instances,
                        "n_features": r.n_features,
                        "n_classes": getattr(r, "n_classes", None),
                    })
        except Exception as e:
            print(f"  PMLB search error ({task}): {e}")
    rng = np.random.default_rng()
    rng.shuffle(candidates)
    return candidates[:batch_size]


def ingest_pmlb_dataset(info: dict) -> tuple[pd.DataFrame | None, dict | None]:
    from tabula.data.pmlb import fetch_pmlb_dataset
    import tempfile
    name = info["name"]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = fetch_pmlb_dataset(name=name, output_root=tmpdir, max_rows=100000)
            csv_path = raw_dir / "train.csv"
            if not csv_path.exists():
                return None, None
            df = pd.read_csv(csv_path)
        if len(df) < 100:
            return None, None
        normalized, inferred = normalize_real_dataset(
            df, preferred_target="target" if "target" in df.columns else None,
            fallback_candidates=["target"],
            source_label=f"pmlb:{name}",
        )
        if normalized is None or inferred is None:
            return None, None
        df = normalized
        task_type = inferred["task_type"]
        domain = classify_domain(df, name)
        source_meta = {
            "source_type": "real",
            "source_id": f"pmlb:{name}",
            "domain": domain,
            "task_type": task_type,
            "license": "CC-BY 4.0",
            "citation_key": "Olson2017PMLB",
            "original_source": f"https://epistasislab.github.io/pmlb/profile/{name}.html",
            "citation": "@article{Olson2017PMLB, title={PMLB: A Large Benchmark Suite for Machine Learning Evaluation and Comparison}, author={Olson et al.}, journal={BioData Mining}, volume={10}, number={36}, year={2017}}",
            "name": name,
            "original_target_column": inferred["target_column"],
        }
        return df, source_meta
    except Exception as e:
        print(f"  [ERR] pmlb:{name}: {e}")
        return None, None


def discover_hf_batch(already_ingested: set[str], batch_size: int = 8) -> list[dict]:
    from tabula.data.huggingface import search_huggingface_datasets
    candidates = []
    seen_repos = set()
    for category in ["tabular-classification", "tabular-regression"]:
        for sort in ["downloads", "likes"]:
            try:
                results = search_huggingface_datasets(
                    task_category=category, limit=30, sort=sort,
                )
                for r in results:
                    sid = f"hf:{r.repo_id}"
                    if sid not in already_ingested and r.repo_id not in seen_repos:
                        seen_repos.add(r.repo_id)
                        candidates.append({
                            "repo_id": r.repo_id,
                            "source_id": sid,
                            "category": category,
                            "downloads": getattr(r, "downloads", 0),
                        })
            except Exception as e:
                print(f"  HF search error ({category}/{sort}): {e}")
    rng = np.random.default_rng()
    rng.shuffle(candidates)
    return candidates[:batch_size]


def ingest_hf_dataset(info: dict) -> tuple[pd.DataFrame | None, dict | None]:
    from tabula.data.huggingface import fetch_huggingface_dataset
    import tempfile
    repo_id = info["repo_id"]
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fetch_result = [None]
            def _do_fetch():
                fetch_result[0] = fetch_huggingface_dataset(
                    repo_id=repo_id, output_root=tmpdir, max_rows=100000,
                )
            _, err = run_with_timeout(_do_fetch, timeout_seconds=180)
            if err:
                print(f"  [ERR] hf:{repo_id}: fetch timeout (180s)")
                return None, None
            raw_dir = fetch_result[0]
            if raw_dir is None:
                return None, None
            csv_path = raw_dir / "train.csv"
            if not csv_path.exists():
                return None, None
            df = pd.read_csv(csv_path)
        if len(df) < 100:
            return None, None
        normalized, inferred = normalize_real_dataset(
            df, preferred_target="target" if "target" in df.columns else None,
            fallback_candidates=["target", "label", "class", "y", "income", "survived"],
            source_label=f"hf:{repo_id}",
        )
        if normalized is None or inferred is None:
            return None, None
        df = normalized
        task_type = inferred["task_type"]
        domain = classify_domain(df, repo_id)
        source_meta = {
            "source_type": "real",
            "source_id": f"hf:{repo_id}",
            "domain": domain,
            "task_type": task_type,
            "license": "other",
            "citation_key": f"hf_{repo_id.replace('/', '_')}",
            "original_source": f"https://huggingface.co/datasets/{repo_id}",
            "citation": f"@misc{{hf_{repo_id.replace('/', '_')}, url={{https://huggingface.co/datasets/{repo_id}}}}}",
            "name": repo_id,
            "original_target_column": inferred["target_column"],
        }
        return df, source_meta
    except Exception as e:
        print(f"  [ERR] hf:{repo_id}: {e}")
        return None, None


def _kaggle_available() -> bool:
    try:
        env = load_repo_env_file()
        has_creds = bool(env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"))
        has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        if has_creds or has_env:
            return True
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        return kaggle_json.exists()
    except Exception:
        return False


def discover_kaggle_batch(already_ingested: set[str], batch_size: int = 8) -> list[dict]:
    if not _kaggle_available():
        print("  Kaggle credentials not configured — skipping.", flush=True)
        return []
    try:
        from tabula.data.kaggle import search_kaggle_datasets, configure_kaggle_credentials
        configure_kaggle_credentials()
    except Exception as e:
        print(f"  Kaggle import/auth error: {e}", flush=True)
        return []
    candidates = []
    queries = ["tabular classification", "tabular regression", "structured data", "survey data"]
    rng = np.random.default_rng()
    rng.shuffle(queries)
    for query in queries[:2]:
        try:
            results = search_kaggle_datasets(search=query, sort_by="votes", min_usability_rating=0.8)
            for r in results:
                sid = f"kaggle:{r.slug}"
                if sid not in already_ingested:
                    candidates.append({
                        "slug": r.slug,
                        "source_id": sid,
                        "title": r.title,
                        "name": r.slug,
                        "size_bytes": r.size_bytes,
                        "download_count": r.download_count,
                    })
        except Exception as e:
            print(f"  Kaggle search error ({query}): {e}", flush=True)
    rng.shuffle(candidates)
    return candidates[:batch_size]


def ingest_kaggle_dataset_for_corpus(info: dict) -> tuple[pd.DataFrame | None, dict | None]:
    import tempfile
    slug = info["slug"]
    try:
        from tabula.data.kaggle import download_kaggle_slug, configure_kaggle_credentials
        configure_kaggle_credentials()
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = download_kaggle_slug(slug, output_root=tmpdir, backend="auto")
            csv_files = list(raw_dir.rglob("*.csv"))
            if not csv_files:
                print(f"  [SKIP] kaggle:{slug}: no CSV files found")
                return None, None
            csv_files.sort(key=lambda p: p.stat().st_size, reverse=True)
            df = pd.read_csv(csv_files[0], nrows=100000)
        if len(df) < 100:
            return None, None
        normalized, inferred = normalize_real_dataset(
            df, preferred_target=None,
            fallback_candidates=_TARGET_CANDIDATES,
            source_label=f"kaggle:{slug}",
        )
        if normalized is None or inferred is None:
            return None, None
        df = normalized
        task_type = inferred["task_type"]
        domain = classify_domain(df, slug)
        source_meta = {
            "source_type": "real",
            "source_id": f"kaggle:{slug}",
            "domain": domain,
            "task_type": task_type,
            "license": "other",
            "citation_key": f"kaggle_{slug.replace('/', '_')}",
            "original_source": f"https://www.kaggle.com/datasets/{slug}",
            "citation": f"@misc{{kaggle_{slug.replace('/', '_')}, url={{https://www.kaggle.com/datasets/{slug}}}}}",
            "name": info.get("title", slug),
            "original_target_column": inferred["target_column"],
        }
        return df, source_meta
    except Exception as e:
        print(f"  [ERR] kaggle:{slug}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Parallel real dataset ingestion (fetch multiple datasets concurrently)
# ---------------------------------------------------------------------------
def _ingest_real_one(args: tuple) -> tuple[pd.DataFrame | None, dict | None]:
    """Ingest one real dataset. For use with ThreadPoolExecutor."""
    source, info = args
    if source == "pmlb":
        return ingest_pmlb_dataset(info)
    elif source == "openml":
        return ingest_openml_dataset(info)
    elif source == "huggingface":
        return ingest_hf_dataset(info)
    elif source == "kaggle":
        return ingest_kaggle_dataset_for_corpus(info)
    return None, None


# ---------------------------------------------------------------------------
# HuggingFace push — same as v2
# ---------------------------------------------------------------------------
def push_batch_to_corpus(dfs: list[pd.DataFrame], metas: list[dict], batch_id: str, token: str):
    from datasets import Dataset
    from huggingface_hub import HfApi

    all_cols = set()
    for df in dfs:
        all_cols.update(df.columns)
    aligned = []
    for df in dfs:
        for col in all_cols - set(df.columns):
            df[col] = np.nan
        aligned.append(df)

    total_rows = sum(len(df) for df in aligned)
    if total_rows > MAX_ROWS_PER_PUSH:
        trimmed = []
        running = 0
        for df in aligned:
            if running + len(df) > MAX_ROWS_PER_PUSH:
                remaining = MAX_ROWS_PER_PUSH - running
                if remaining > 100:
                    trimmed.append(df.iloc[:remaining])
                    running += remaining
                break
            trimmed.append(df)
            running += len(df)
        aligned = trimmed
        print(f"  NOTE: Capped batch to {running:,} rows (from {total_rows:,})", flush=True)

    combined = pd.concat(aligned, ignore_index=True)
    if "_source_meta" not in combined.columns:
        combined["_source_meta"] = ""

    obj_cols = combined.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        combined[col] = combined[col].astype(str)

    new_shard = Dataset.from_pandas(combined, preserve_index=False)

    print(f"  Pushing {len(combined):,} rows to {CORPUS_REPO} config={batch_id}...", flush=True)

    def _do_push():
        new_shard.push_to_hub(
            CORPUS_REPO,
            config_name=batch_id,
            token=token,
            commit_message=f"Add batch {batch_id} ({len(combined):,} rows)",
        )

    max_attempts = 3
    last_err = None
    for attempt in range(1, max_attempts + 1):
        _, err = run_with_timeout(_do_push, timeout_seconds=900)  # 15 min for larger pushes
        if err is None:
            last_err = None
            break
        last_err = err
        if attempt < max_attempts:
            wait = 15 * (2 ** (attempt - 1))
            print(f"  Push attempt {attempt}/{max_attempts} failed: {err}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    if last_err:
        raise RuntimeError(f"HF push failed after {max_attempts} attempts: {last_err}")

    try:
        _update_corpus_card(batch_id, token)
    except Exception as e:
        print(f"  WARN: Card update failed: {e}", flush=True)

    print(f"  Push complete: {CORPUS_REPO} / {batch_id}", flush=True)

    # Free memory after push
    del combined, new_shard, aligned
    gc.collect()


def _update_corpus_card(batch_id: str, token: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    config_names_on_hf: set[str] = set()
    try:
        for item in api.list_repo_tree(CORPUS_REPO, repo_type="dataset"):
            name = getattr(item, "path", "") or getattr(item, "rfilename", "") or ""
            clean = name.strip("/").split("/")[0]
            if clean.startswith("datagen_"):
                config_names_on_hf.add(clean)
    except Exception:
        pass
    config_names_on_hf.add(batch_id)

    log = read_log()
    success = log[log["status"] == "success"] if not log.empty else log

    total_rows = 0
    for v in success.get("total_rows", []):
        try:
            total_rows += int(v)
        except (ValueError, TypeError):
            pass
    n_real = int((success["source_type"] == "real").sum()) if not success.empty else 0
    n_synthetic = int((success["source_type"] == "synthetic").sum()) if not success.empty else 0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    seen_batches = set()
    sources_table = "| batch_id | source_type | method | source_id | n_datasets | total_rows | status |\n"
    sources_table += "|----------|-------------|--------|-----------|------------|------------|--------|\n"
    if not success.empty:
        for _, row in success.iterrows():
            bid = row["batch_id"]
            if bid in seen_batches:
                continue
            seen_batches.add(bid)
            sources_table += f"| {bid} | {row['source_type']} | {row['method']} | {row['source_id'][:50]} | {row['n_datasets']} | {row['total_rows']} | {row['status']} |\n"

    config_names = sorted(config_names_on_hf)
    configs_yaml = ""
    for cfg in config_names:
        configs_yaml += f"  - config_name: {cfg}\n"
        configs_yaml += f"    data_files:\n"
        configs_yaml += f"      - split: train\n"
        configs_yaml += f"        path: {cfg}/*\n"

    card = f"""---
license: other
task_categories:
  - tabular-classification
  - tabular-regression
language:
  - en
tags:
  - tabular
  - synthetic
  - real-data
  - pretraining
  - tabpfn
pretty_name: "Tabula Pretraining Corpus"
configs:
{configs_yaml}---

# Tabula Pretraining Corpus

A continuously growing tabular pretraining corpus for the Tabula foundation model
(tabPFN-style in-context learning). Built by an autonomous agent that alternates
between harvesting permissively-licensed real datasets and generating high-quality
synthetic ones.

## Usage

```python
from datasets import load_dataset

# Load a specific batch config
ds = load_dataset("avewright/tabula-pretraining-corpus", name="datagen_001")

# Load all configs
from huggingface_hub import HfApi
api = HfApi()
```

## Stats (auto-updated)

| Metric | Value |
|--------|-------|
| Total rows | {total_rows:,} |
| Real-data batches | {n_real} |
| Synthetic batches | {n_synthetic} |
| Last updated | {timestamp} |

## Schema

Every row has feature columns plus `_source_meta` (JSON string):
- `batch_id`, `source_type`, `source_id`, `domain`, `task_type`, `license`, `citation_key`

## Sources & Citations

{sources_table}

## License

Individual rows carry their own `license` field inside `_source_meta`.
Synthetic rows are Apache 2.0. Real rows carry the original source license.
"""

    try:
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=CORPUS_REPO,
            repo_type="dataset",
            commit_message=f"Update card after batch {batch_id}",
        )
    except Exception as e:
        print(f"  WARN: Could not update corpus card: {e}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_iteration(batch_id: str, dfs: list[pd.DataFrame], logged_total_rows: int):
    actual_rows = sum(len(df) for df in dfs)
    assert actual_rows == logged_total_rows, \
        f"row count mismatch: logged={logged_total_rows}, actual={actual_rows}"

    log = read_log()
    real_success = log[(log["status"] == "success") & (log["source_type"] == "real")]
    if not real_success.empty:
        dupes = real_success["source_id"].duplicated()
        if dupes.any():
            print(f"  WARN: duplicate real source_ids: {real_success.loc[dupes, 'source_id'].tolist()}")

    for df in dfs:
        assert "_source_meta" in df.columns, "missing _source_meta column"

    for df in dfs:
        for col in df.columns:
            if col in ("target", "_source_meta"):
                continue
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                if parts[0] in DOMAIN_VOCAB:
                    print(f"  WARN: overflow column name detected: {col}")


# ---------------------------------------------------------------------------
# Mode decisions & method rotation
# ---------------------------------------------------------------------------
def _build_method_rotation() -> list[str]:
    """Build method rotation including available Tier-2 GPU methods."""
    methods = list(SYNTHETIC_METHODS_TIER1)
    tier2 = _check_tier2()
    for m in SYNTHETIC_METHODS_TIER2:
        if tier2.get(m, False):
            methods.append(m)
    return methods


def decide_mode(log: pd.DataFrame, already_ingested: set[str]) -> str:
    if log.empty:
        return "SYNTHETIC"
    success = log[log["status"] == "success"]
    if success.empty:
        return "SYNTHETIC"

    recent = log.tail(3)
    recent_real_fails = recent[
        (recent["source_type"] == "real") & (recent["status"].isin(["gate_fail", "crash"]))
    ]
    if len(recent_real_fails) >= 2:
        return "SYNTHETIC"

    last_3_success = success.tail(3)
    if (last_3_success["source_type"] == "synthetic").all() and len(last_3_success) >= 3:
        return "REAL"
    elif success.iloc[-1]["source_type"] == "real":
        return "SYNTHETIC"
    return "SYNTHETIC"


def get_next_synthetic_method(log: pd.DataFrame) -> str:
    methods = _build_method_rotation()
    if log.empty:
        return methods[0]

    synth_all = log[log["source_type"] == "synthetic"]
    if synth_all.empty:
        return methods[0]

    last_method = synth_all.iloc[-1]["method"]
    try:
        idx = methods.index(last_method)
    except ValueError:
        idx = -1
    next_method = methods[(idx + 1) % len(methods)]

    recent = log.tail(8)
    fail_count = ((recent["method"] == next_method) & (recent["status"].isin(["gate_fail", "crash"]))).sum()
    if fail_count >= 3:
        next_method = methods[(idx + 2) % len(methods)]

    return next_method


# ---------------------------------------------------------------------------
# Iteration runners
# ---------------------------------------------------------------------------
def run_synthetic_iteration(batch_id: str, method: str, token: str):
    rng = np.random.default_rng()
    print(f"\n{'='*60}", flush=True)
    print(f"SYNTHETIC ITERATION: {batch_id} | method={method} | workers={GENERATION_WORKERS}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.monotonic()
    dfs, metas = generate_synthetic_bundle_parallel(method, batch_id, rng)
    gen_time = time.monotonic() - t0

    if not dfs:
        print("  No datasets passed quality gates!")
        append_log_row({
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": "synthetic",
            "method": method,
            "source_id": f"synthetic:{method}",
            "n_datasets": 0,
            "total_rows": 0,
            "n_features_range": "0",
            "utility_auc": "0.0",
            "hub_repo": CORPUS_REPO,
            "status": "gate_fail",
            "notes": "No datasets passed quality gates",
        })
        return

    total_rows = sum(len(df) for df in dfs)
    feature_ranges = [len([c for c in df.columns if c not in ("target", "_source_meta")]) for df in dfs]
    feat_range_str = f"{min(feature_ranges)}-{max(feature_ranges)}"
    avg_utility = np.mean([m.get("utility_auc", 0) for m in metas])

    t1 = time.monotonic()
    push_batch_to_corpus(dfs, metas, batch_id, token)
    push_time = time.monotonic() - t1

    append_log_row({
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": "synthetic",
        "method": method,
        "source_id": f"synthetic:{method}",
        "n_datasets": len(dfs),
        "total_rows": total_rows,
        "n_features_range": feat_range_str,
        "utility_auc": f"{avg_utility:.3f}",
        "hub_repo": CORPUS_REPO,
        "status": "success",
        "notes": f"domains={set(m['domain'] for m in metas)}; gen={gen_time:.0f}s push={push_time:.0f}s",
    })

    verify_iteration(batch_id, dfs, total_rows)
    print(f"  Iteration {batch_id} complete: {len(dfs)} datasets, {total_rows:,} rows, "
          f"avg_auc={avg_utility:.3f}, gen={gen_time:.0f}s, push={push_time:.0f}s", flush=True)

    # Aggressive memory cleanup between iterations
    del dfs, metas
    gc.collect()


def run_real_iteration(batch_id: str, token: str, already_ingested: set[str]):
    print(f"\n{'='*60}", flush=True)
    print(f"REAL ITERATION: {batch_id}", flush=True)
    print(f"{'='*60}", flush=True)

    log = read_log()
    real_successes = log[(log["status"] == "success") & (log["source_type"] == "real")]
    source_idx = len(real_successes) % len(REAL_SOURCES)
    source = REAL_SOURCES[source_idx]
    print(f"  Real source this iteration: {source}", flush=True)

    # Discover with larger batch sizes (more RAM available)
    batch_size = 8
    candidates = []
    if source == "pmlb":
        candidates = discover_pmlb_batch(already_ingested, batch_size=batch_size)
    elif source == "openml":
        candidates = discover_openml_batch(already_ingested, batch_size=batch_size)
    elif source == "huggingface":
        candidates = discover_hf_batch(already_ingested, batch_size=batch_size)
    elif source == "kaggle":
        candidates = discover_kaggle_batch(already_ingested, batch_size=batch_size)

    if not candidates:
        print(f"  No new {source} datasets found. Trying other sources...", flush=True)
        for alt in REAL_SOURCES:
            if alt == source:
                continue
            if alt == "pmlb":
                candidates = discover_pmlb_batch(already_ingested, batch_size=batch_size)
            elif alt == "openml":
                candidates = discover_openml_batch(already_ingested, batch_size=batch_size)
            elif alt == "huggingface":
                candidates = discover_hf_batch(already_ingested, batch_size=batch_size)
            elif alt == "kaggle":
                candidates = discover_kaggle_batch(already_ingested, batch_size=batch_size)
            if candidates:
                source = alt
                print(f"  Falling back to {source}", flush=True)
                break

    if not candidates:
        print("  No real datasets available from any source.", flush=True)
        return False

    # Parallel fetch real datasets using ThreadPoolExecutor (I/O bound)
    dfs = []
    metas = []
    print(f"  Fetching {len(candidates)} datasets in parallel...", flush=True)

    with ThreadPoolExecutor(max_workers=min(4, len(candidates))) as executor:
        futures = {
            executor.submit(_ingest_real_one, (source, info)): info
            for info in candidates
        }
        for future in as_completed(futures):
            info = futures[future]
            sid = info.get("source_id", "unknown")
            name = info.get("name", info.get("repo_id", "unknown"))
            try:
                df, source_meta = future.result()
            except Exception as e:
                print(f"  [ERR] {sid}: {e}", flush=True)
                continue

            if df is None:
                continue

            passed, utility_auc, gate_notes = run_quality_gates(df, source_meta["task_type"])
            if not passed:
                print(f"  [SKIP] {sid}: {gate_notes}", flush=True)
                continue

            if source_meta["task_type"] == "regression" and utility_auc < -1.0:
                print(f"  [SKIP] {sid}: R2={utility_auc:.3f} too low", flush=True)
                continue
            if source_meta["task_type"] in ("binary", "multiclass") and utility_auc < 0.50:
                print(f"  [SKIP] {sid}: AUC={utility_auc:.3f} too low", flush=True)
                continue

            source_meta["batch_id"] = batch_id
            source_meta["utility_auc"] = utility_auc
            df["_source_meta"] = json.dumps(source_meta)
            dfs.append(df)
            metas.append(source_meta)
            print(f"  [OK] {sid}: n={len(df)} task={source_meta['task_type']} "
                  f"domain={source_meta['domain']} auc={utility_auc:.3f}", flush=True)

    if not dfs:
        print("  No real datasets passed quality gates.", flush=True)
        append_log_row({
            "batch_id": batch_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": "real",
            "method": "real-ingest",
            "source_id": f"{source}:sweep",
            "n_datasets": 0,
            "total_rows": 0,
            "n_features_range": "0",
            "utility_auc": "0.0",
            "hub_repo": CORPUS_REPO,
            "status": "gate_fail",
            "notes": f"No real datasets passed gates (source={source})",
        })
        return False

    total_rows = sum(len(df) for df in dfs)
    feature_ranges = [len([c for c in df.columns if c not in ("target", "_source_meta")]) for df in dfs]
    feat_range_str = f"{min(feature_ranges)}-{max(feature_ranges)}"
    avg_utility = np.mean([m.get("utility_auc", 0) for m in metas])
    source_ids = "|".join(m["source_id"] for m in metas)

    push_batch_to_corpus(dfs, metas, batch_id, token)

    append_log_row({
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_type": "real",
        "method": "real-ingest",
        "source_id": source_ids,
        "n_datasets": len(dfs),
        "total_rows": total_rows,
        "n_features_range": feat_range_str,
        "utility_auc": f"{avg_utility:.3f}",
        "hub_repo": CORPUS_REPO,
        "status": "success",
        "notes": f"sources: {[m.get('name','') for m in metas]}",
    })

    verify_iteration(batch_id, dfs, total_rows)
    print(f"  Iteration {batch_id} complete: {len(dfs)} real datasets, {total_rows:,} rows")

    for m in metas:
        already_ingested.add(m["source_id"])

    del dfs, metas
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 70, flush=True)
    print("TABULA DATAGEN POD — OPTIMIZED FOR RTX A4500 / 12 vCPU / 62 GB", flush=True)
    print(f"Corpus: {CORPUS_REPO}", flush=True)
    print(f"Log: {LOG_FILE}", flush=True)
    print(f"Generation workers: {GENERATION_WORKERS} (of {NUM_CPUS} CPUs)", flush=True)
    print(f"Datasets per bundle: {MIN_DATASETS_PER_BUNDLE}-{MAX_DATASETS_PER_BUNDLE}", flush=True)
    print(f"Max rows per push: {MAX_ROWS_PER_PUSH:,}", flush=True)
    print(f"Domains: {len(DOMAIN_VOCAB)} ({', '.join(DOMAIN_VOCAB.keys())})", flush=True)
    print(f"Min vocab size: {min(len(v) for v in DOMAIN_VOCAB.values())}", flush=True)
    print(f"Total unique column names: {sum(len(v) for v in DOMAIN_VOCAB.values())}", flush=True)

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)", flush=True)
        else:
            print("GPU: Not available (Tier-1 only)", flush=True)
    except ImportError:
        print("GPU: torch not available (Tier-1 only)", flush=True)

    # Tier-2 availability
    tier2 = _check_tier2()
    tier2_str = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in tier2.items())
    print(f"Tier-2 methods: {tier2_str or 'none'}", flush=True)
    print(f"Method rotation: {_build_method_rotation()}", flush=True)
    print("=" * 70, flush=True)

    token = load_hf_token()
    print(f"HF_TOKEN loaded (ends with ...{token[-4:]})", flush=True)

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for attempt in range(5):
        try:
            api.repo_info(CORPUS_REPO, repo_type="dataset")
            print(f"Corpus repo {CORPUS_REPO} exists.", flush=True)
            break
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                print(f"Creating corpus repo {CORPUS_REPO}...", flush=True)
                api.create_repo(CORPUS_REPO, repo_type="dataset", private=False)
                break
            if attempt < 4:
                wait = 10 * (attempt + 1)
                print(f"  Startup network error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise

    init_log()
    log = read_log()
    already_ingested = get_already_ingested()
    print(f"Log has {len(log)} entries, {len(already_ingested)} source_ids already ingested.", flush=True)

    iteration = 0
    while True:
        iteration += 1
        log = read_log()
        already_ingested = get_already_ingested()
        batch_id = next_batch_id(log)

        mode = decide_mode(log, already_ingested)
        current_method = "unknown"
        iter_start = time.monotonic()

        try:
            if mode == "REAL":
                current_method = "real-ingest"
                try:
                    success = run_real_iteration(batch_id, token, already_ingested)
                except Exception as e:
                    print(f"  REAL iteration failed: {e}", flush=True)
                    success = False
                if not success:
                    log = read_log()
                    batch_id = next_batch_id(log)
                    current_method = get_next_synthetic_method(log)
                    run_synthetic_iteration(batch_id, current_method, token)
            else:
                current_method = get_next_synthetic_method(log)
                run_synthetic_iteration(batch_id, current_method, token)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting.", flush=True)
            break
        except Exception:
            tb = traceback.format_exc()
            print(f"\n  CRASH in iteration {batch_id}:\n{tb}", flush=True)
            append_log_row({
                "batch_id": batch_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_type": mode.lower(),
                "method": current_method,
                "source_id": "crash",
                "n_datasets": 0,
                "total_rows": 0,
                "n_features_range": "0",
                "utility_auc": "0.0",
                "hub_repo": CORPUS_REPO,
                "status": "crash",
                "notes": tb[:200],
            })
            continue
        finally:
            elapsed = time.monotonic() - iter_start
            print(f"  [TIMER] Iteration {batch_id} total: {elapsed:.0f}s", flush=True)
            sys.stdout.flush()
            gc.collect()

    print("Datagen loop terminated.", flush=True)


if __name__ == "__main__":
    main()
