"""
Pre-Surgical Hemoglobin-A1c Risk CDS Pipeline
(Epic EHI -> Cosmos -> tiered CDS actions)

- use pre-op serum measured HbA1c
- optionally predict HbA1c (linear) for missing/stale labs using sparse regression (LASSO via adelie)
- fork downstream behavior using NSQIP Simplified Diabetes Surgical Risk Model thresholds:
    * info card only:    a1c < 7.5
    * info card & warning: 7.5 <= a1c <= 8.5
    * pre-operative surgical critical issue EMR workflow block: a1c > 8.5
- emit FHIR artifacts back to epic (CommunicationRequest always; Task only for critical)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from azure.cosmos import CosmosClient
import matplotlib.pyplot as plt


# config
@dataclass(frozen=True)
class Settings:
    cosmos_url: str
    cosmos_key: str
    database_id: str
    container_patient: str
    container_labs: str
    container_vitals: str

    fhir_base: str
    oauth_bearer: str

    preop_date: str  # yyyy-mm-dd
    low_threshold: float   # hba1c low-risk cutoff (default 7.5)
    high_threshold: float  # hba1c high-risk cutoff (default 8.5)

    # legacy single-threshold mode (optional)
    H: Optional[float] = None
    delta: Optional[float] = None

    dry_run: bool
    plot: bool

    training_csv: Optional[str]  # optional: path to historical training data
    enable_prediction: bool      # if true, predict hba1c when missing


def load_settings(args: argparse.Namespace) -> Settings:
    # env vars
    cosmos_url = os.getenv("COSMOS_URL", "").strip()
    cosmos_key = os.getenv("COSMOS_KEY", "").strip()
    database_id = os.getenv("DATABASE_ID", "").strip()

    # azure cosmos db container names
    container_patient = os.getenv("CONTAINER_PATIENT", "PATIENT").strip()
    container_labs = os.getenv("CONTAINER_LABS", "NSQIP_PREOP_LABS").strip()
    container_vitals = os.getenv("CONTAINER_VITALS", "PREOP_VITALS").strip()

    fhir_base = os.getenv("FHIR_BASE", "https://openepic.example.org/fhir").strip()
    oauth_bearer = os.getenv("OAUTH_BEARER", "").strip()

    preop_date = args.preop_date or os.getenv("PREOP_DATE", date.today().isoformat()).strip()

    # thresholds (default model: <7.5 low, 7.5–8.5 medium, >8.5 high)
    low_threshold = float(args.low if getattr(args, "low", None) is not None else os.getenv("HBA1C_LOW_THRESHOLD", "7.5"))
    high_threshold = float(args.high if getattr(args, "high", None) is not None else os.getenv("HBA1C_HIGH_THRESHOLD", "8.5"))

    # FixMe: look to eliminate if possible
    # legacy single-threshold mode (optional; will be ignored if low/high provided)
    H = args.H if getattr(args, "H", None) is not None else (float(os.getenv("HBA1C_THRESHOLD_H", "0")) or None)
    delta = args.delta if getattr(args, "delta", None) is not None else (float(os.getenv("HBA1C_THRESHOLD_DELTA", "0")) or None)
    if not cosmos_url or not cosmos_key or not database_id:
        raise RuntimeError(
            "Missing Cosmos settings. Ensure COSMOS_URL, COSMOS_KEY, DATABASE_ID are set (see .env.example)."
        )

    return Settings(
        cosmos_url=cosmos_url,
        cosmos_key=cosmos_key,
        database_id=database_id,
        container_patient=container_patient,
        container_labs=container_labs,
        container_vitals=container_vitals,
        fhir_base=fhir_base,
        oauth_bearer=oauth_bearer,
        preop_date=preop_date,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        H=H,
        delta=delta,
        dry_run=bool(args.dry_run),
        plot=bool(args.plot),
        training_csv=args.training_csv,
        enable_prediction=bool(args.enable_prediction),
    )


# azure cosmos helper fcns
def cosmos_client(settings: Settings) -> CosmosClient:
    return CosmosClient(settings.cosmos_url, settings.cosmos_key)


def query_container(container, query: str, parameters: Optional[list] = None) -> list:
    if parameters is None:
        parameters = []
    return list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))


def load_preop_cohort(db, settings: Settings) -> pd.DataFrame:
    c_pat = db.get_container_client(settings.container_patient)

    q = """
    SELECT
        c.patient_id,
        c.encounter.id AS encounter_id,
        c.encounter.preop_date AS preop_date,
        c.birthdate,
        c.gender,
        c.surgical_risk
    FROM c
    WHERE c.encounter.preop_date = @preop_date
    """
    rows = query_container(c_pat, q, [{"name": "@preop_date", "value": settings.preop_date}])
    return pd.DataFrame(rows)


def load_a1c_labs(db, settings: Settings) -> pd.DataFrame:
    c_labs = db.get_container_client(settings.container_labs)

    # pull hba1c lab records that exist (across all patients; we filter after merge)
    q = """
    SELECT
        c.patient_id,
        c.labs.a1c.value AS a1c,
        c.labs.a1c.effectiveDateTime AS lab_time
    FROM c
    WHERE IS_DEFINED(c.labs.a1c) AND IS_DEFINED(c.labs.a1c.value)
    """
    rows = query_container(c_labs, q)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # normalize and keep most recent hba1c per patient
    df["lab_time"] = pd.to_datetime(df["lab_time"], errors="coerce", utc=True)
    df["a1c"] = pd.to_numeric(df["a1c"], errors="coerce")
    df = df.dropna(subset=["a1c"])
    df = df.sort_values(["patient_id", "lab_time"]).groupby("patient_id", as_index=False).tail(1)
    return df[["patient_id", "a1c", "lab_time"]]


def load_vitals(db, settings: Settings) -> pd.DataFrame:
    c_vitals = db.get_container_client(settings.container_vitals)

    # if vitals have a date field, filter by preop_date. otherwise, pull recent and filter here
    q = """
    SELECT
        c.patient_id,
        c.vitals.bp.systolic AS sbp,
        c.vitals.bp.diastolic AS dbp,
        c.vitals.bmi AS bmi,
        c.time AS vitals_time
    FROM c
    """
    rows = query_container(c_vitals, q)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["vitals_time"] = pd.to_datetime(df.get("vitals_time"), errors="coerce", utc=True)
    df["sbp"] = pd.to_numeric(df.get("sbp"), errors="coerce")
    df["dbp"] = pd.to_numeric(df.get("dbp"), errors="coerce")
    df["bmi"] = pd.to_numeric(df.get("bmi"), errors="coerce")

    # keep most recent vitals per patient (best-effort)
    if "vitals_time" in df.columns:
        df = df.sort_values(["patient_id", "vitals_time"]).groupby("patient_id", as_index=False).tail(1)

    return df[["patient_id", "sbp", "dbp", "bmi", "vitals_time"]]




# features & sparse regression (optional)
def add_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bd = pd.to_datetime(out.get("birthdate"), errors="coerce")
    today = pd.Timestamp(date.today())
    out["age"] = (today - bd).dt.days / 365.25
    out["age"] = out["age"].round(1)
    return out


def build_design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    minimal design matrix for quick testing...expand later with real feature set
    """
    work = df.copy()

    # categorical -> one-hot
    work["gender"] = work["gender"].fillna("unknown").astype(str).str.lower()
    work["surgical_risk"] = work["surgical_risk"].fillna("unknown").astype(str).str.lower()

    # numeric
    for col in ["age", "sbp", "dbp", "bmi"]:
        work[col] = pd.to_numeric(work.get(col), errors="coerce")

    # one-hot
    X_cat = pd.get_dummies(work[["gender", "surgical_risk"]], dummy_na=False)
    X_num = work[["age", "sbp", "dbp", "bmi"]]
    X = pd.concat([X_num, X_cat], axis=1)

    feature_names = list(X.columns)
    X = X.to_numpy(dtype=float)

    # replace NaNs (simple imputation for smoke-tst)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    return X, feature_names


def fit_predict_a1c_with_adelie(df: pd.DataFrame, training_csv: Optional[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    fit a sparse linear model (LASSO) and predict hba1c for rows with missing a1c.
    priority order:
      1) if training_csv provided: train on it (must contain columns used in build_design_matrix & 'a1c')
      2) else: train on in-scope rows that have 'a1c'
    :param df: only obvious
    :return:
      - predicted_a1c (series aligned to df index; NaN where not predicted)
      - model_info (metadata for debugging)
    """
    try:
        import adelie as ad  # noqa: F401
    except Exception as e:
        raise RuntimeError("adelie is not available in this environment. Install with: pip install adelie") from e

    # build training data frame
    if training_csv:
        train_df = pd.read_csv(training_csv)
    else:
        train_df = df.dropna(subset=["a1c"]).copy()

    if train_df.empty or train_df["a1c"].isna().all():  # nothing to train with
        return pd.Series([np.nan] * len(df), index=df.index), {"trained": False, "reason": "no_training_data"}

    train_df = add_age(train_df)
    X_train, feature_names = build_design_matrix(train_df)
    y_train = pd.to_numeric(train_df["a1c"], errors="coerce").to_numpy(dtype=float)

    # as recommended in docs for performance
    X_train = np.asfortranarray(X_train)

    # cv group elastic net wrapper (with default groups == each feature => lasso-like sparsity)
    import adelie as ad
    model = ad.GroupElasticNet(solver="cv_grpnet")
    model.fit(X_train, y_train)

    # predict only where missing
    pred = pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    mask = df["a1c"].isna()
    if mask.any():
        pred_df = add_age(df.loc[mask].copy())
        X_pred, _ = build_design_matrix(pred_df)
        X_pred = np.asfortranarray(X_pred)
        pred.loc[mask] = model.predict(X_pred)

    # coefficients for explainability
    try:
        coef = np.array(model.coef_).ravel().tolist()
    except Exception:
        coef = []

    info = {
        "trained": True,
        "n_train": int(len(train_df)),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "coef": coef,
    }
    return pred, info


# thresholding / cds tiering
def tier_from_a1c(
    a1c_value: Optional[float],
    low_threshold: float,
    high_threshold: float,
    *,
    legacy_H: Optional[float] = None,
    legacy_delta: Optional[float] = None,
) -> Tuple[str, str, Dict[str, float]]:
    """
    :return:
      tier: info | warning | critical | unknown
      label: Low | Medium | High | Unknown
      metrics: {a1c_used, low_threshold, high_threshold, abs_below_high, abs_above_low, pct_of_band}
    """
    if a1c_value is None or (isinstance(a1c_value, float) and np.isnan(a1c_value)):
        return "unknown", "Unknown", {
            "a1c_used": np.nan,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "abs_below_high": np.nan,
            "abs_above_low": np.nan,
            "pct_of_band": np.nan,
        }

    a1c = float(a1c_value)

    # if caller did not provide explicit low/high, fall back to legacy (H/delta) banding.
    if (low_threshold is None or high_threshold is None) and legacy_H is not None and legacy_delta is not None:
        H = float(legacy_H)
        delta = float(legacy_delta)
        low_threshold = H - delta
        high_threshold = H

    abs_below_high = high_threshold - a1c   # positive => below high cutoff
    abs_above_low = a1c - low_threshold     # positive => above low cutoff
    band = max(high_threshold - low_threshold, 1e-9)
    pct_of_band = (a1c - low_threshold) / band  # 0=at low cutoff, 1=at high cutoff

    if a1c > high_threshold:
        return "critical", "High", {
            "a1c_used": a1c,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "abs_below_high": abs_below_high,
            "abs_above_low": abs_above_low,
            "pct_of_band": pct_of_band,
        }
    if a1c >= low_threshold:
        return "warning", "Medium", {
            "a1c_used": a1c,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "abs_below_high": abs_below_high,
            "abs_above_low": abs_above_low,
            "pct_of_band": pct_of_band,
        }
    return "info", "Low", {
        "a1c_used": a1c,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "abs_below_high": abs_below_high,
        "abs_above_low": abs_above_low,
        "pct_of_band": pct_of_band,
    }




def workflow_action_from_tier(tier: str) -> str:
    if tier == "critical":
        return "Block"
    if tier == "warning":
        return "Warn"
    if tier == "info":
        return "Info"
    return "None"




def build_comm_payload(a1c_used: float, tier: str, metrics: Dict[str, float]) -> str:
    low = float(metrics.get("low_threshold", np.nan))
    high = float(metrics.get("high_threshold", np.nan))
    abs_below_high = float(metrics.get("abs_below_high", np.nan))

    if tier == "critical":
        above_high = a1c_used - high
        return (
            f"Pre-op HbA1c {a1c_used:.2f}% (> {high:.2f}%). **High risk**. "
            f"Stop workflow and require clinical huddle / clearance. "
            f"Above high cutoff by {above_high:+.2f} HbA1c points."
        )
    if tier == "warning":
        return (
            f"Pre-op HbA1c {a1c_used:.2f}% ({low:.2f}–{high:.2f}%). **Medium risk**. "
            f"Trigger Epic warning. {abs_below_high:+.2f} HbA1c points below high cutoff."
        )
    if tier == "info":
        return (
            f"Pre-op HbA1c {a1c_used:.2f}% (< {low:.2f}%). **Low risk**. Info card only. "
            f"{abs_below_high:+.2f} HbA1c points below high cutoff."
        )
    return "Pre-op HbA1c unavailable."




def post_fhir_resource(url: str, headers: Dict[str, str], payload: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)




def send_fhir_alerts(df: pd.DataFrame, settings: Settings) -> None:
    headers = {
        "Authorization": f"Bearer {settings.oauth_bearer}",
        "Content-Type": "application/fhir+json",
    }

    for _, row in df.iterrows():
        action = row.get("workflow_action", "None")
        if action == "None":
            continue

        patient_ref = f"Patient/{row['patient_id']}"
        encounter_ref = f"Encounter/{row['encounter_id']}"

        comm = {
            "resourceType": "CommunicationRequest",
            "status": "active",
            "subject": {"reference": patient_ref},
            "encounter": {"reference": encounter_ref},
            "payload": [{"contentString": row.get("cds_message", "Pre-op risk message")}],
            "reasonCode": [{"text": "Pre-op Glycemic Surgical Risk"}],
        }
        # Priority drives UI salience (Epic behavior depends on local build; this is a best-effort signal).
        if action == "Block":
            comm["priority"] = "stat"
            comm["category"] = [{"text": "Stop-the-line / Clinical huddle"}]
        elif action == "Warn":
            comm["priority"] = "urgent"
            comm["category"] = [{"text": "Pre-op warning"}]
        elif action == "Info":
            comm["priority"] = "routine"
        post_fhir_resource(f"{settings.fhir_base}/CommunicationRequest", headers, comm, settings.dry_run)

        if action == "Block":
            task = {
                "resourceType": "Task",
                "status": "requested",
                "intent": "order",
                "description": "High surgical diabetes risk — HITL sign-off required.",
                "for": {"reference": patient_ref},
                "encounter": {"reference": encounter_ref},
                "priority": "stat",
            }
            post_fhir_resource(f"{settings.fhir_base}/Task", headers, task, settings.dry_run)


# main action
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preop-date", default=None, help="YYYY-MM-DD (defaults to today or PREOP_DATE env)")

    parser.add_argument("--low", type=float, default=None,
                        help="Low-risk HbA1c cutoff (env: HBA1C_LOW_THRESHOLD; default 7.5)")
    parser.add_argument("--high", type=float, default=None,
                        help="High-risk HbA1c cutoff (env: HBA1C_HIGH_THRESHOLD; default 8.5)")

    # Legacy single-threshold mode (deprecated)
    parser.add_argument("--H", type=float, default=None, help="[DEPRECATED] HbA1c action threshold (env: HBA1C_THRESHOLD_H)")
    parser.add_argument("--delta", type=float, default=None, help="[DEPRECATED] Near-threshold band width (env: HBA1C_THRESHOLD_DELTA)")

    parser.add_argument("--dry-run", action="store_true", help="Do not POST to FHIR; still compute outputs")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable matplotlib chart")
    parser.set_defaults(plot=True)

    parser.add_argument("--enable-prediction", action="store_true",
                        help="Predict HbA1c when missing via sparse regression (adelie)")
    parser.add_argument("--training-csv", default=None,
                        help="Optional CSV for training lasso (must include a1c + features)")

    args = parser.parse_args()
    settings = load_settings(args)

    client = cosmos_client(settings)
    db = client.get_database_client(settings.database_id)

    df_pat = load_preop_cohort(db, settings)
    df_labs = load_a1c_labs(db, settings)
    df_vitals = load_vitals(db, settings)

    if df_pat.empty:
        print(f"No pre-op cohort found for preop_date={settings.preop_date}.")
        return 0

    df = df_pat.merge(df_labs, on="patient_id", how="left").merge(df_vitals, on="patient_id", how="left")
    df = add_age(df)

    # optional: predict a1c where missing (smoke-test)
    model_info = {"trained": False}
    if settings.enable_prediction:
        pred_a1c, model_info = fit_predict_a1c_with_adelie(df, settings.training_csv)
        df["a1c_predicted"] = pred_a1c
        # Choose measured if present, else predicted
        df["a1c_used"] = df["a1c"].where(df["a1c"].notna(), df["a1c_predicted"])
    else:
        df["a1c_used"] = df["a1c"]

    # tier assignment
    tiers = df["a1c_used"].apply(lambda v: tier_from_a1c(v, settings.low_threshold, settings.high_threshold, legacy_H=settings.H, legacy_delta=settings.delta))
    df["tier"] = tiers.apply(lambda t: t[0])
    df["risk_category"] = tiers.apply(lambda t: t[1])
    df["tier_metrics"] = tiers.apply(lambda t: t[2])

    df["workflow_action"] = df["tier"].apply(workflow_action_from_tier)
    df["cds_message"] = df.apply(
        lambda r: build_comm_payload(float(r["tier_metrics"]["a1c_used"]) if not np.isnan(r["tier_metrics"]["a1c_used"]) else np.nan,
                                     r["tier"],
                                     r["tier_metrics"]),
        axis=1,
    )

    # emit resulting card back to epic in FHIR format
    if not settings.oauth_bearer and not settings.dry_run:
        raise RuntimeError("OAUTH_BEARER is missing. Set it or run with --dry-run.")
    send_fhir_alerts(df, settings)

    # optional viz
    if settings.plot:
        risk_counts = df["risk_category"].value_counts(dropna=False)
        plt.figure(figsize=(7, 4))
        risk_counts.plot(kind="bar")
        plt.title("Preoperative HbA1c Tier (<7.5 / 7.5–8.5 / >8.5)")
        plt.ylabel("Patient Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # print compact view for COB dev-to-tst sanity check
    cols = ["patient_id", "encounter_id", "preop_date", "a1c", "a1c_used", "tier", "risk_category", "workflow_action"]
    existing = [c for c in cols if c in df.columns]
    print(df[existing].to_string(index=False))

    if settings.enable_prediction:
        print("\n[adelie model_info]")
        print(json.dumps({k: v for k, v in model_info.items() if k != "coef"}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# powered by riparianOne


