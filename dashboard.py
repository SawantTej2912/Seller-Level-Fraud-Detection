"""
Seller-Level Fraud Detection Dashboard
DATA 245 — Spring 2026 | Real-Time Transaction Simulation
Team: Tejas Sawant · Ameya Khond · Prathmesh Mankar
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import os
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from collections import defaultdict

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Header ── */
    .dash-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #1a3a5c 100%);
        padding: 1.4rem 2rem 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 5px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .dash-header h1 {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        letter-spacing: 0.5px;
    }
    .dash-header p {
        color: #8ab4d4;
        font-size: 0.85rem;
        margin: 0;
    }

    /* ── Section titles ── */
    .sec-title {
        color: #4fc3f7;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        border-bottom: 1px solid #1a3a5c;
        padding-bottom: 4px;
        margin-bottom: 8px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d1b2a, #1b2838);
        border: 1px solid #1a3a5c;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] > div { color: #8ab4d4 !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] > div { color: #ffffff !important; font-size: 1.5rem; }
    [data-testid="stMetricDelta"] > div { font-size: 0.75rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background-color: #0d1b2a; }
    [data-testid="stSidebar"] label { color: #8ab4d4 !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: #8ab4d4; font-size: 0.82rem; }

    /* ── Dataframe ── */
    .stDataFrame { font-size: 0.78rem; }

    /* ── Status badge ── */
    .badge-fraud  { background:#e94560; color:#fff; padding:2px 9px; border-radius:12px;
                    font-size:0.72rem; font-weight:700; }
    .badge-legit  { background:#22c55e; color:#fff; padding:2px 9px; border-radius:12px;
                    font-size:0.72rem; font-weight:700; }
    .badge-hi-risk{ background:#e94560; color:#fff; padding:2px 9px; border-radius:12px;
                    font-size:0.72rem; font-weight:700; }
    .badge-ok     { background:#22c55e; color:#fff; padding:2px 9px; border-radius:12px;
                    font-size:0.72rem; font-weight:700; }

    /* ── Setup error box ── */
    .setup-box {
        background: #1b2838; border: 1px solid #e94560; border-radius: 10px;
        padding: 1.5rem 2rem; margin: 2rem 0;
    }
    .setup-box h3 { color: #e94560; margin-top: 0; }
    .setup-box code { background: #0d1b2a; padding: 2px 6px; border-radius: 4px;
                      color: #4fc3f7; font-size: 0.85rem; }
    .setup-box ol li { color: #8ab4d4; margin-bottom: 6px; }

    div[data-testid="stHorizontalBlock"] { gap: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>🛡️ Seller-Level Fraud Detection Dashboard</h1>
  <p>DATA 245 &mdash; Spring 2026 &nbsp;|&nbsp; Real-Time Transaction Simulation
     &nbsp;|&nbsp; Tejas Sawant &middot; Ameya Khond &middot; Prathmesh Mankar</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · Load Artifacts
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRED_FILES = ["model.pkl", "scaler.pkl", "feature_names.json", "config.json"]

@st.cache_resource(show_spinner="⚙️  Loading model artifacts…")
def load_artifacts():
    missing = [f for f in REQUIRED_FILES
               if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        return None, None, None, None, None, None, missing

    model  = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

    with open(os.path.join(BASE_DIR, "feature_names.json")) as fh:
        fn_data = json.load(fh)
    if isinstance(fn_data, list):
        numeric_features = fn_data
        cat_features     = ["channel_web", "merchant_category_te",
                             "country_te", "bin_country_te"]
    else:
        numeric_features = fn_data.get("numeric", [])
        cat_features     = fn_data.get("categorical",
                           ["channel_web", "merchant_category_te",
                            "country_te", "bin_country_te"])

    with open(os.path.join(BASE_DIR, "config.json")) as fh:
        config = json.load(fh)

    # Optional artifacts (graceful fallback)
    te_maps, avg_map = {}, {"__global__": 1.0}

    te_path = os.path.join(BASE_DIR, "target_encoding_maps.json")
    if os.path.exists(te_path):
        with open(te_path) as fh:
            te_maps = json.load(fh)

    avg_path = os.path.join(BASE_DIR, "avg_amount_map.json")
    if os.path.exists(avg_path):
        with open(avg_path) as fh:
            avg_map = json.load(fh)

    artifacts = dict(
        model=model, scaler=scaler,
        numeric_features=numeric_features,
        cat_features=cat_features,
        config=config, te_maps=te_maps, avg_map=avg_map,
    )
    return artifacts, missing


result = load_artifacts()
artifacts, missing_files = result if isinstance(result, tuple) else (result, [])

if missing_files or artifacts is None:
    st.markdown(f"""
    <div class="setup-box">
      <h3>⚠️ Model artifacts not found</h3>
      <p>Missing files: <code>{'</code>, <code>'.join(missing_files or REQUIRED_FILES)}</code></p>
      <ol>
        <li>Open <code>fraud-detection.ipynb</code> in Jupyter</li>
        <li>Run all cells <strong>top to bottom</strong></li>
        <li>Run the last cell (<strong>SETUP: Save All Dashboard Artifacts</strong>)<br>
            — this generates <code>model.pkl</code>, <code>scaler.pkl</code>,
            <code>feature_names.json</code>, <code>config.json</code></li>
        <li>Return here and refresh the page</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model            = artifacts["model"]
scaler           = artifacts["scaler"]
numeric_features = artifacts["numeric_features"]
cat_features     = artifacts["cat_features"]
config           = artifacts["config"]
te_maps          = artifacts["te_maps"]
avg_map          = artifacts["avg_map"]

ALL_FEATURE_NAMES = list(numeric_features) + list(cat_features)
_MODEL_MTIME      = os.path.getmtime(os.path.join(BASE_DIR, "model.pkl"))

THRESHOLD      = float(config.get("threshold",       0.90))
RISK_THRESHOLD = float(config.get("risk_threshold",  0.30))
GLOBAL_FR      = float(config.get("global_fraud_rate", 0.022))

# Fixed simulation pacing (sidebar sliders removed)
SIM_TRANSACTIONS_PER_SEC = 100.0
SIM_BATCH_SIZE           = 10

HIGH_RISK_AVG      = 0.40
HIGH_RISK_FRACTION = 0.20
HIGH_RISK_SLOPE    = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · Session State
# ══════════════════════════════════════════════════════════════════════════════
def _blank_seller():
    return {
        "fraud_probs": [], "tx_count": 0, "flagged_count": 0,
        "avg_risk_score": 0.0, "high_risk_fraction": 0.0,
        "risk_trend_slope": 0.0, "is_high_risk": False,
        "actual_fraud_count": 0,
    }


def init_session_state():
    defaults = {
        "transaction_log":         [],
        "seller_profiles":         defaultdict(_blank_seller),
        "current_index":           0,
        "running":                 False,
        "total_processed":         0,
        "total_fraud_caught":      0,
        "total_high_risk_sellers": 0,
        "df":                      None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · Data Loading
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="📂  Loading transaction data…")
def load_data():
    demo_path = os.path.join(BASE_DIR, "demo_transactions.csv")
    orig_path = os.path.join(BASE_DIR, "transactions 2.csv")

    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
    elif os.path.exists(orig_path):
        df = (pd.read_csv(orig_path)
                .sample(n=2000, random_state=42)
                .reset_index(drop=True))
    else:
        st.error("No transaction data found. "
                 "Place `demo_transactions.csv` or `transactions 2.csv` "
                 "in the same directory as dashboard.py.")
        st.stop()

    # Parse timestamps (do not shuffle — order matters for simulation)
    if "transaction_time" in df.columns:
        df["transaction_time"] = pd.to_datetime(df["transaction_time"], utc=True,
                                                errors="coerce")
    return df


if st.session_state.df is None:
    st.session_state.df = load_data()

sim_df = st.session_state.df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · Seller Risk Update
# ══════════════════════════════════════════════════════════════════════════════
def update_seller_profile(seller_id, fraud_prob, actual_fraud):
    p = st.session_state.seller_profiles[seller_id]
    p["fraud_probs"].append(fraud_prob)
    p["tx_count"]           += 1
    p["actual_fraud_count"] += int(actual_fraud)
    if fraud_prob >= THRESHOLD:
        p["flagged_count"] += 1

    probs = p["fraud_probs"]
    p["avg_risk_score"]     = round(float(np.mean(probs)), 4)
    p["high_risk_fraction"] = round(
        sum(1 for x in probs if x >= RISK_THRESHOLD) / len(probs), 4)

    if len(probs) >= 3:
        xs = np.arange(len(probs), dtype=float)
        p["risk_trend_slope"] = round(float(np.polyfit(xs, probs, 1)[0]), 6)

    p["is_high_risk"] = (
        p["avg_risk_score"]     >= HIGH_RISK_AVG      or
        p["high_risk_fraction"] >= HIGH_RISK_FRACTION or
        p["risk_trend_slope"]   >= HIGH_RISK_SLOPE
    )
    st.session_state.seller_profiles[seller_id] = p


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · Feature Engineering + Scoring
# ══════════════════════════════════════════════════════════════════════════════
def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def build_numeric_vector(row, feats):
    """Build the numeric feature vector that goes through the scaler."""
    out = {}
    for feat in feats:
        if feat == "log_amount":
            out[feat] = np.log1p(_safe_float(row.get("amount", 0)))
        elif feat == "amount_ratio":
            uid  = str(int(_safe_float(row.get("user_id", 0))))
            avg  = float(avg_map.get(uid, avg_map.get("__global__", 1.0)))
            out[feat] = _safe_float(row.get("amount", 0)) / (avg + 1e-6)
        elif feat == "country_match":
            out[feat] = int(str(row.get("country", "")) ==
                            str(row.get("bin_country", "")))
        elif feat == "hour":
            try:
                out[feat] = pd.to_datetime(row["transaction_time"]).hour
            except Exception:
                out[feat] = _safe_float(row.get("hour", 12))
        elif feat == "night_flag":
            h = out.get("hour", _safe_float(row.get("hour", 12)))
            out[feat] = int(int(h) in [22, 23, 0, 1, 2, 3, 4, 5])
        else:
            out[feat] = _safe_float(row.get(feat, 0.0))
    return np.array([[out.get(f, 0.0) for f in feats]], dtype=float)


def build_cat_vector(row, cat_feats):
    """Build the 4 categorical features appended AFTER scaling."""
    out = []
    for feat in cat_feats:
        if feat == "channel_web":
            out.append(float(str(row.get("channel", "")).lower() == "web"))
        elif feat.endswith("_te"):
            col = feat[:-3]           # "merchant_category", "country", "bin_country"
            val = str(row.get(col, ""))
            out.append(te_maps.get(col, {}).get(val, GLOBAL_FR))
        else:
            out.append(_safe_float(row.get(feat, 0.0)))
    return np.array([out], dtype=float)


def build_combined_matrix(row):
    """Feature matrix in the same column order as training (numeric scaled + categorical)."""
    num_vec = build_numeric_vector(row, numeric_features)
    scaled  = scaler.transform(num_vec)
    if cat_features:
        cat_vec = build_cat_vector(row, cat_features)
        return np.hstack([scaled, cat_vec])
    return scaled


def score_combined(combined):
    prob = float(model.predict_proba(combined)[0][1])
    return prob, prob >= THRESHOLD


def score_transaction(row):
    return score_combined(build_combined_matrix(row))


@st.cache_resource(show_spinner=False)
def _tree_shap_explainer(_artifact_mtime: float):
    import shap
    return shap.TreeExplainer(model)


def top_feature_explanation(combined_2d: np.ndarray) -> str:
    """Feature with largest |SHAP| (or XGB contribution); UI shows name only, no numeric value."""
    if combined_2d is None or combined_2d.size == 0:
        return "—"
    names = ALL_FEATURE_NAMES
    # (1) Tree SHAP — matches notebook analysis when the package is available
    try:
        explainer = _tree_shap_explainer(_MODEL_MTIME)
        sv = explainer.shap_values(combined_2d)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv).reshape(-1)
        n = min(len(sv), len(names))
        if n == 0:
            raise ValueError("empty shap")
        sv = sv[:n]
        use_names = names[:n]
        j = int(np.argmax(np.abs(sv)))
        return str(use_names[j])
    except Exception:
        pass
    # (2) XGBoost marginal contributions — no extra deps, same |ranking| idea as local explanation
    try:
        if not hasattr(model, "get_booster"):
            return "—"
        booster = model.get_booster()
        arr = np.asarray(combined_2d, dtype=np.float32)
        dm = xgb.DMatrix(arr, feature_names=list(names))
        c = booster.predict(dm, pred_contribs=True)
        vec = np.asarray(c, dtype=float).reshape(-1)
        if vec.size < 2:
            return "—"
        feats = vec[:-1]
        n = min(len(feats), len(names))
        feats, use_names = feats[:n], names[:n]
        j = int(np.argmax(np.abs(feats)))
        return str(use_names[j])
    except Exception:
        return "—"


def _top_feature_display(val) -> str:
    """Strip SHAP/impact suffixes so the table shows only the feature name."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    s = str(val).strip()
    for sep in (" (SHAP ", " (impact "):
        if sep in s:
            return s.split(sep, 1)[0].strip()
    return s or "—"


# ── Leaderboard helper functions ──────────────────────────────────────────────
def get_flag_reason(profile):
    """Plain-English explanation of why a seller was flagged."""
    reasons = []
    if profile["avg_risk_score"] >= HIGH_RISK_AVG:
        reasons.append(f"High avg risk ({profile['avg_risk_score']:.2f})")
    if profile["high_risk_fraction"] >= HIGH_RISK_FRACTION:
        reasons.append(f"{profile['high_risk_fraction'] * 100:.0f}% txns suspicious")
    if profile["risk_trend_slope"] >= HIGH_RISK_SLOPE:
        reasons.append(f"Risk escalating (+{profile['risk_trend_slope']:.3f}/txn)")
    return " · ".join(reasons) if reasons else "—"


def get_top_signal(profile):
    """Single most important feature signal inferred from seller metrics
    (aligned with SHAP ranking: shipping_distance > avs_match > account_age)."""
    if profile["avg_risk_score"] >= 0.80:
        return "shipping_distance_km"
    elif profile["high_risk_fraction"] >= 0.60:
        return "amount / log_amount"
    elif profile["risk_trend_slope"] >= HIGH_RISK_SLOPE:
        return "Behavioral escalation"
    else:
        return "Multi-feature"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · Sidebar Controls (rendered before layout)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎮 Simulation Controls")

    n_max = len(sim_df)
    n_to_run = st.slider(
        "Transactions to simulate",
        min_value=50,
        max_value=n_max,
        value=min(500, n_max),
        step=50,
        help="How many transactions to feed into this simulation run. "
             "Hit Reset then Start to apply a new value.",
    )

    c_start, c_reset = st.columns(2)
    with c_start:
        btn_label = "⏸ Pause" if st.session_state.running else "▶ Start"
        if st.button(btn_label, use_container_width=True, type="primary"):
            st.session_state.running = not st.session_state.running
            st.rerun()
    with c_reset:
        if st.button("🔄 Reset", use_container_width=True):
            for k in ["transaction_log", "seller_profiles", "current_index",
                      "running", "total_processed", "total_fraud_caught",
                      "total_high_risk_sellers"]:
                if k in st.session_state:
                    del st.session_state[k]
            init_session_state()
            st.rerun()

    # Progress bar — based on the user-chosen transaction count
    n_total      = n_to_run
    progress_val = min(st.session_state.total_processed / max(n_total, 1), 1.0)
    st.progress(progress_val,
                text=f"{st.session_state.total_processed:,} / {n_total:,} transactions")

    st.divider()
    st.markdown(f"**Decision Threshold:** `{THRESHOLD:.2f}`")
    st.markdown(f"**Seller Risk Threshold:** `{RISK_THRESHOLD:.2f}`")

    st.divider()
    st.markdown("**🔍 Filters**")
    _lb_filter = st.radio(
        "Leaderboard sellers",
        options=["All", "🔴 High-risk only", "🟢 Low-risk only"],
        index=0,
        key="leaderboard_seller_filter",
        help="Choose one view for the seller risk leaderboard.",
    )
    show_hr_only = _lb_filter == "🔴 High-risk only"
    show_lr_only = _lb_filter == "🟢 Low-risk only"
    show_fraud_only = st.checkbox("🚨 Fraud transactions only (feed)", value=False)

    st.divider()
    n_sellers   = len(st.session_state.seller_profiles)
    n_hr        = st.session_state.total_high_risk_sellers
    pct_hr      = (n_hr / n_sellers * 100) if n_sellers > 0 else 0.0
    all_probs   = [t["Fraud Prob"] for t in st.session_state.transaction_log]
    avg_prob    = float(np.mean(all_probs)) if all_probs else 0.0

    st.markdown("**📊 Live Stats**")
    st.markdown(f"""
- Sellers seen: **{n_sellers:,}**
- High-risk: **{n_hr}** ({pct_hr:.1f}%)
- Avg fraud prob: **{avg_prob:.4f}**
- Model: **XGB + Categoricals**
""")



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · Dashboard Layout  (dashboard1.py-style UI)
# ══════════════════════════════════════════════════════════════════════════════

# ── TOP ROW · 4 Metric Cards ──────────────────────────────────────────────────
processed  = st.session_state.total_processed
caught     = st.session_state.total_fraud_caught
hr_sellers = st.session_state.total_high_risk_sellers
flag_rate  = (caught / processed * 100) if processed else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions processed", f"{processed:,}")
m2.metric("Fraud-flagged (predicted)", f"{caught:,}")
m3.metric("High-risk sellers", f"{hr_sellers:,}")
m4.metric("Flag rate %", f"{flag_rate:.2f}%")

st.divider()

# ── Seller search ─────────────────────────────────────────────────────────────
st.subheader("Search seller — all transactions & risk")
_log_all = list(st.session_state.transaction_log)
_sellers_in_log = sorted({int(e["Seller ID"]) for e in _log_all})
_sc1, _sc2 = st.columns([1, 2])
with _sc1:
    _pick = st.selectbox("Seller in log", options=["—"] + [str(s) for s in _sellers_in_log],
                         index=0, key="seller_search_pick")
with _sc2:
    _typed = st.text_input("Or enter seller ID", value="", key="seller_search_type",
                           placeholder="numeric user_id (overrides dropdown if filled)")

_lookup_sid = None
if _typed.strip():
    try:
        _lookup_sid = int(_typed.strip())
    except ValueError:
        st.warning("Seller ID must be an integer.")
elif _pick != "—":
    _lookup_sid = int(_pick)

if _lookup_sid is not None:
    _rows_s = [e for e in _log_all if int(e["Seller ID"]) == _lookup_sid]
    if _rows_s:
        _prof = st.session_state.seller_profiles[_lookup_sid]
        _u1, _u2, _u3, _u4, _u5 = st.columns(5)
        _u1.metric("Seller ID", f"{_lookup_sid}")
        _u2.metric("Transactions", f"{len(_rows_s):,}")
        _u3.metric("Avg fraud prob", f"{_prof['avg_risk_score']:.4f}")
        _u4.metric("High-risk frac", f"{100.0 * float(_prof['high_risk_fraction']):.1f}%")
        _u5.metric("Seller status", "HIGH RISK" if _prof["is_high_risk"] else "OK")
        _s_df = pd.DataFrame(_rows_s).copy()
        if "Top feature impacting" not in _s_df.columns:
            _s_df["Top feature impacting"] = "—"
        _s_df["Top feature impacting"] = _s_df["Top feature impacting"].map(_top_feature_display)
        def _plabel_s(p):
            if p >= 0.7: return f"🔴 {p:.4f}"
            elif p >= 0.3: return f"🟠 {p:.4f}"
            return f"🟢 {p:.4f}"
        _s_df["Fraud Prob"] = _s_df["Fraud Prob"].apply(_plabel_s)
        _s_df["Predicted Fraud"] = _s_df["Predicted Fraud"].apply(lambda x: "🔴 YES" if x else "✅ NO")
        _s_df["Actual Fraud"] = _s_df["Actual Fraud"].apply(lambda x: "⚠️ YES" if x else "NO")
        _cols_s = [c for c in [
            "Transaction ID", "Amount", "Channel", "Fraud Prob", "Predicted Fraud",
            "Actual Fraud", "Top feature impacting",
        ] if c in _s_df.columns]
        st.dataframe(_s_df[_cols_s], use_container_width=True, hide_index=True)
    else:
        st.info(f"No transactions for seller **{_lookup_sid}** in this session.")

# ── MIDDLE ROW: feed + leaderboard side by side ───────────────────────────────
FEED_HEIGHT = 448
left_col, right_col = st.columns([0.46, 0.54])

with left_col:
    st.subheader("Live transaction feed")
    txn_log = list(st.session_state.transaction_log)
    display_log = [t for t in txn_log if t["Predicted Fraud"]] if show_fraud_only else txn_log
    if display_log:
        feed_df = pd.DataFrame(display_log).copy()
        if "Top feature impacting" not in feed_df.columns:
            feed_df["Top feature impacting"] = "—"
        feed_df["Top feature impacting"] = feed_df["Top feature impacting"].map(_top_feature_display)
        def _prob_label(p):
            if p >= 0.7:   return f"🔴 {p:.4f}"
            elif p >= 0.3: return f"🟠 {p:.4f}"
            return             f"🟢 {p:.4f}"
        feed_df["Fraud Prob"]      = feed_df["Fraud Prob"].apply(_prob_label)
        feed_df["Predicted Fraud"] = feed_df["Predicted Fraud"].apply(lambda x: "🔴 YES" if x else "✅ NO")
        feed_df["Actual Fraud"]    = feed_df["Actual Fraud"].apply(lambda x: "⚠️ YES" if x else "NO")
        _cols_feed = [c for c in [
            "Transaction ID", "Seller ID", "Amount", "Channel", "Fraud Prob",
            "Predicted Fraud", "Actual Fraud", "Top feature impacting",
        ] if c in feed_df.columns]
        st.caption(f"Fraud prob: 🟢 low · 🟠 medium · 🔴 high — **{len(feed_df)}** rows · scroll inside table")
        st.dataframe(feed_df[_cols_feed], use_container_width=True, hide_index=True, height=FEED_HEIGHT)
    else:
        st.info("▶ Press **Start** to begin the simulation.")

with right_col:
    st.subheader("Seller risk leaderboard")
    _all_profiles = dict(st.session_state.seller_profiles)
    lb_rows = []
    for sid, p in _all_profiles.items():
        if p["tx_count"] == 0: continue
        if show_hr_only and not p["is_high_risk"]: continue
        if show_lr_only and p["is_high_risk"]: continue
        lb_rows.append({
            "Seller": sid,
            "Avg Risk": p["avg_risk_score"],
            "HR Frac": round(100.0 * p["high_risk_fraction"], 2),
            "Trans": int(p["tx_count"]),
            "Fraud": int(p["actual_fraud_count"]),
            "Status": "🔴 HIGH RISK" if p["is_high_risk"] else "🟢 OK",
        })
    if lb_rows:
        lb_df = pd.DataFrame(lb_rows).sort_values("Avg Risk", ascending=False).reset_index(drop=True)
        st.caption(f"Seller status: 🟢 OK · 🔴 HIGH RISK — **{len(lb_df)}** sellers · scroll inside table")
        st.dataframe(lb_df, use_container_width=True, hide_index=True, height=FEED_HEIGHT)
    else:
        st.caption("No sellers profiled yet.")

# ── BOTTOM: seller risk histogram ─────────────────────────────────────────────
st.subheader("Seller risk score distribution")
avgs = [p["avg_risk_score"] for p in st.session_state.seller_profiles.values() if p["tx_count"] > 0]
if avgs:
    counts, edges = np.histogram(np.clip(avgs, 0.0, 1.0), bins=22, range=(0, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    def _bar_color(v):
        if v >= 0.7: return "#e74c3c"
        if v >= 0.3: return "#f39c12"
        return "#2ecc71"
    bar_colors = [_bar_color(float(c)) for c in centers]
    fig_hist = go.Figure(data=[go.Bar(x=centers, y=counts, marker_color=bar_colors, marker_line_width=0)])
    fig_hist.update_layout(
        height=340,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_title="Avg risk score", yaxis_title="# Sellers",
        xaxis=dict(range=[0, 1], tickformat=".1f"),
        margin=dict(l=10, r=10, t=10, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.caption("No seller averages yet.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 · Main Simulation Loop
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.running:
    df  = st.session_state.df
    idx = st.session_state.current_index
    cap = n_to_run

    if idx >= cap:
        st.session_state.running = False
        st.success(
            f"✅ Simulation complete — {cap:,} transactions processed. "
            f"Fraud caught: {st.session_state.total_fraud_caught:,} "
            f"({st.session_state.total_fraud_caught / max(cap, 1):.1%}) | "
            f"High-risk sellers flagged: {st.session_state.total_high_risk_sellers:,}"
        )
    else:
        end_idx = min(idx + SIM_BATCH_SIZE, cap)

        for i in range(idx, end_idx):
            row = df.iloc[i]
            try:
                combined = build_combined_matrix(row)
                fraud_prob, is_predicted_fraud = score_combined(combined)
                top_feat = top_feature_explanation(combined)
            except Exception:
                fraud_prob, is_predicted_fraud = 0.0, False
                top_feat = "—"

            actual    = int(row.get("is_fraud",  0))
            seller_id = int(_safe_float(row.get("user_id", 0)))
            tx_id_raw = row.get("transaction_id", i)
            try:
                tx_id = int(tx_id_raw)
            except (TypeError, ValueError):
                tx_id = i

            update_seller_profile(seller_id, fraud_prob, actual)

            st.session_state.transaction_log.append({
                "Transaction ID": tx_id,
                "Seller ID":      seller_id,
                "Amount":         round(_safe_float(row.get("amount", 0)), 2),
                "Channel":        str(row.get("channel", "N/A")),
                "Fraud Prob":     round(fraud_prob, 4),
                "Predicted Fraud": is_predicted_fraud,
                "Actual Fraud":   bool(actual),
                "Correct":        is_predicted_fraud == bool(actual),
                "Top feature impacting": top_feat,
            })

            if is_predicted_fraud:
                st.session_state.total_fraud_caught += 1
            st.session_state.total_processed += 1

        st.session_state.current_index = end_idx
        st.session_state.total_high_risk_sellers = sum(
            1 for p in st.session_state.seller_profiles.values()
            if p["is_high_risk"]
        )

        time.sleep(1.0 / SIM_TRANSACTIONS_PER_SEC)
        st.rerun()
