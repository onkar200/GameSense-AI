
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from chatbot.engine import GameSenseChatbot

# ============================================================
# GameSense AI | Player Performance Intelligence Dashboard
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features"

@st.cache_resource
def load_gamesense_chatbot():
    return GameSenseChatbot()

chatbot = load_gamesense_chatbot()

st.set_page_config(
    page_title="GameSense AI | Player Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
:root {
    --bg: #090b10; --panel: #11151e; --panel2: #171d28;
    --line: #293241; --muted: #9aa6b7; --text: #edf2f8;
    --red: #ff4655; --cyan: #43d9e6; --amber: #ffbd4a;
}
.stApp { background: radial-gradient(circle at 80% 0%, #1a202d 0%, #090b10 42%); color: var(--text); }
.block-container { max-width: 1600px; padding-top: 1.35rem; padding-bottom: 3rem; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: .4px; }
h1 { font-size: 3.1rem !important; font-weight: 800 !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.45rem !important; }
[data-testid="stSidebar"] { background: #0d1119; border-right: 1px solid var(--line); }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #171d28, #10141c);
    border: 1px solid #2b3545; border-radius: 13px;
    padding: 17px 18px; box-shadow: 0 8px 24px #0003;
}
[data-testid="stMetricLabel"] { color: #aab5c5 !important; font-size: .82rem; }
[data-testid="stMetricValue"] { color: #f4f7fb !important; font-weight: 750; }
[data-testid="stMetricDelta"] { font-size: .78rem; }
div[data-testid="stPlotlyChart"] {
    background: #10151e; border: 1px solid #273142;
    border-radius: 13px; padding: 7px;
}
div[data-testid="stTabs"] button { font-weight: 650; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff6470; border-bottom-color: #ff4655;
}
div[data-testid="stDataFrame"] { border: 1px solid #293241; border-radius: 10px; }
.gs-eyebrow { color: #ff6571; font-size: .73rem; font-weight: 800; letter-spacing: 2px; }
.gs-subtitle { color: #aab5c5; margin-top: -12px; font-size: .95rem; }
.gs-card {
    background: linear-gradient(145deg, #171d28, #10141c);
    border: 1px solid #2b3545; border-radius: 13px;
    padding: 17px 19px; min-height: 100px;
}
.gs-card-label { color: #9aa6b7; font-size: .78rem; text-transform: uppercase; letter-spacing: 1px; }
.gs-card-value { color: #f4f7fb; font-size: 1.8rem; font-weight: 800; margin-top: 5px; }
.gs-card-note { color: #8f9bad; font-size: .78rem; margin-top: 3px; }
.gs-warning { border-left: 3px solid #ffbd4a; padding: 9px 13px; background: #211d16; border-radius: 5px; color: #e7d6b4; }
.gs-good { border-left: 3px solid #43d9a2; padding: 9px 13px; background: #12221e; border-radius: 5px; color: #bdebd9; }
hr { border-color: #293241 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #8f9bad !important; }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "#10151e"
GRID = "#273142"
RED = "#ff4655"
CYAN = "#43d9e6"
AMBER = "#ffbd4a"
GREEN = "#43d9a2"

CATEGORY_COLORS = {
    "Stable": CYAN,
    "Mild Signs": AMBER,
    "Mild": AMBER,
    "Elevated": "#ff8a4c",
    "Strong Potential Tilt": RED,
    "Strong": RED,
    "Insufficient Data": "#788596",
}

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_csv(filename):
    path = FEATURE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


gti = load_csv("gamesense_gti.csv")
validation = load_csv("gamesense_validation.csv")
paber = load_csv("gamesense_paber.csv")
validation_paber = load_csv("gamesense_validation_paber.csv")
predictions = load_csv("gamesense_paber_predictions.csv")
comparison = load_csv("gamesense_paber_comparison.csv")
bootstrap_summary = load_csv("gamesense_paber_bootstrap_summary.csv")


def find_column(df, candidates):
    if df.empty:
        return None
    return next((c for c in candidates if c in df.columns), None)


def numeric(df, candidates):
    col = find_column(df, candidates)
    return pd.to_numeric(df[col], errors="coerce") if col else None


def metric_value(series, fmt=".2f"):
    if series is None:
        return "—"
    values = pd.to_numeric(series, errors="coerce").dropna()
    return format(values.mean(), fmt) if not values.empty else "—"


def show_table(df, columns=None, height=380):
    if df is None or df.empty:
        st.info("No records are available for this view yet.")
        return
    cols = [c for c in (columns or df.columns.tolist()) if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True, height=height)


def style_fig(fig, height=350):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=18, r=18, t=55, b=25),
        font=dict(family="Inter", color="#dce4ef", size=12),
        title_font=dict(size=16, color="#f4f7fb"),
        legend_title_text="",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    return fig


def sort_records(df):
    out = df.copy()
    time_col = find_column(
        out, ["utc_timestamp", "timestamp", "date", "game_date"]
    )
    if time_col:
        out["_sort_time"] = pd.to_datetime(
            out[time_col], errors="coerce", utc=True
        )
        out = out.sort_values("_sort_time", kind="stable")
        out = out.drop(columns="_sort_time")
    elif "game_id" in out.columns:
        out["_sort_game"] = pd.to_numeric(
            out["game_id"], errors="coerce"
        )
        out = out.sort_values("_sort_game", kind="stable")
        out = out.drop(columns="_sort_game")
    return out


def score_band(score):
    if pd.isna(score):
        return "Insufficient Data"
    if score <= 25:
        return "Stable"
    if score <= 50:
        return "Mild Signs"
    if score <= 75:
        return "Elevated"
    return "Strong Potential Tilt"

# ============================================================
# GAMESENSE PLAYER ADVISOR CHAT ENGINE
# ============================================================

def answer_player_question(question, player_name, report, current_gti, current_paber):
    """Generate grounded answers from the selected player's report."""

    q = question.lower().strip()
    metrics = report.get("metrics", [])

    if not report.get("ready"):
        return (
            "I don't have enough historical data to analyse this player yet. "
            + report.get("message", "")
        )

    def metric_line(item):
        pct = item["change_pct"]

        if np.isfinite(pct):
            change = f"{pct:+.1f}%"
        else:
            change = f"{item['deterioration']:+.2f} absolute change"

        direction = (
            "deteriorated"
            if item["deterioration"] > 0
            else "improved"
        )

        return (
            f"**{item['metric']}**: baseline "
            f"{item['baseline']:.2f}, recent "
            f"{item['recent']:.2f}. This metric {direction} "
            f"({change} in the deterioration direction)."
        )

    if any(word in q for word in ["weakness", "weak", "worst", "decline", "deteriorat"]):
        weak = sorted(
            [m for m in metrics if m["deterioration"] > 0],
            key=lambda x: x["rank_score"],
            reverse=True,
        )

        if not weak:
            return (
                f"I couldn't identify a deteriorating tracked metric "
                f"for {player_name} across the recent maps compared "
                "with their earlier baseline."
            )

        return (
            f"Here are the main areas to review for **{player_name}**:\n\n"
            + "\n\n".join(metric_line(m) for m in weak[:3])
        )

    if any(word in q for word in ["recommend", "improve", "suggestion", "should", "advice"]):
        if not report["recommendations"]:
            return "No specific improvement area was flagged by the available metrics."

        return (
            f"My performance-review suggestions for **{player_name}**:\n\n"
            + "\n".join(
                f"{i}. {advice}"
                for i, advice in enumerate(report["recommendations"], 1)
            )
        )

    if "gti" in q:
        if pd.isna(current_gti):
            return "A recent GTI score is not available for this player."

        band = score_band(current_gti)

        return (
            f"**{player_name}'s recent average GTI is {current_gti:.1f}/100 "
            f"({band}).**\n\n"
            "GTI is a rule-based performance-deterioration indicator. "
            "It uses changes in recorded performance metrics and does "
            "not measure a player's actual mental state."
        )

    if "paber" in q:
        if pd.isna(current_paber):
            return (
                "PABER is not available for the selected player's "
                "recent maps. This can happen when historical baseline "
                "requirements are not met."
            )

        return (
            f"**{player_name}'s recent average PABER is "
            f"{current_paber:.1f}/100.**\n\n"
            "PABER compares performance with a player-adaptive "
            "historical baseline. It is a statistical signal, not "
            "a measurement of psychological tilt."
        )

    if any(word in q for word in ["acs", "adr", "kast", "death", "kill", "multikill", "first"]):
        matching = []

        keywords = {
            "acs": ["ACS"],
            "adr": ["ADR"],
            "kast": ["KAST"],
            "death": ["Deaths", "First Death Rate"],
            "kill": ["K/D Ratio", "Multikills"],
            "multikill": ["Multikills"],
            "first": ["First Death Rate"],
        }

        for word, names in keywords.items():
            if word in q:
                matching.extend(
                    m for m in metrics if m["metric"] in names
                )

        matching = list({
            m["metric"]: m for m in matching
        }.values())

        if matching:
            return "\n\n".join(metric_line(m) for m in matching)

        return (
            "That metric is not available or does not have enough "
            "valid observations in this player's dataset."
        )

    if any(word in q for word in ["summary", "overall", "analyse", "analyze", "performance", "how is"]):
        deteriorated = [
            m for m in metrics if m["deterioration"] > 0
        ]

        improved = [
            m for m in metrics if m["deterioration"] < 0
        ]

        summary = (
            f"**Performance summary for {player_name}**\n\n"
            f"- Recent GTI: {current_gti:.1f}/100\n"
            if pd.notna(current_gti)
            else f"**Performance summary for {player_name}**\n\n"
        )

        summary += (
            f"- Recent PABER: {current_paber:.1f}/100\n"
            if pd.notna(current_paber)
            else "- Recent PABER: unavailable\n"
        )

        summary += (
            f"- Tracked metrics deteriorating: {len(deteriorated)}\n"
            f"- Tracked metrics improving: {len(improved)}\n\n"
            "These comparisons use the recent three maps against "
            "earlier available map history."
        )

        return summary

    return (
        f"I can help analyse **{player_name}**. Try asking:\n\n"
        "- What are the biggest weaknesses?\n"
        "- How has ACS changed?\n"
        "- Explain the GTI score.\n"
        "- What does PABER indicate?\n"
        "- What should this player improve?\n"
        "- Give me an overall performance summary."
    )

# ============================================================
# GAMESENSE PLAYER ADVISOR ENGINE
# ============================================================

PLAYER_METRICS = {
    "ACS": {
        "columns": ["acs_all", "acs", "average_combat_score", "avg_combat_score"],
        "direction": "higher",
        "advice": "Review damage contribution, engagement quality and mid-round impact.",
    },
    "ADR": {
        "columns": ["adr_all", "adr", "average_damage_per_round"],
        "direction": "higher",
        "advice": "Review damage consistency and whether engagements are producing enough impact.",
    },
    "KAST": {
        "columns": ["kast_all", "kast", "kast_percentage"],
        "direction": "higher",
        "advice": "Review survival, trading, assists and overall round involvement.",
    },
    "K/D Ratio": {
        "columns": ["kd_ratio", "kd", "kill_death_ratio"],
        "direction": "higher",
        "advice": "Review duel selection, positioning and opportunities to trade teammates.",
    },
    "Deaths": {
        "columns": ["deaths_all", "deaths", "total_deaths"],
        "direction": "lower",
        "advice": "Review positioning, avoidable deaths and whether engagements have team support.",
    },
    "First Death Rate": {
        "columns": ["first_death_rate", "first_death_pct", "first_deaths_rate"],
        "direction": "lower",
        "advice": "Review opening-duel timing, information gathering and coordination.",
    },
    "Multikills": {
        "columns": ["multikills", "multi_kills", "multikills_all"],
        "direction": "higher",
        "advice": "Review conversion of multi-player opportunities and follow-up positioning.",
    },
}


def build_player_advisor_report(player_df):
    """
    Compare the latest three maps against the player's earlier
    available map history.
    """

    ordered = sort_records(player_df.copy())

    if len(ordered) < 5:
        return {
            "ready": False,
            "message": (
                f"Only {len(ordered)} map records are available. "
                "At least 5 records are required: 3 recent maps "
                "and at least 2 baseline maps."
            ),
            "metrics": [],
            "recommendations": [],
        }

    recent = ordered.tail(3)
    baseline = ordered.iloc[:-3]

    if len(baseline) < 2:
        return {
            "ready": False,
            "message": "Not enough earlier maps to establish a player baseline.",
            "metrics": [],
            "recommendations": [],
        }

    results = []

    for metric_name, config in PLAYER_METRICS.items():

        col = find_column(ordered, config["columns"])

        if not col:
            continue

        old_values = pd.to_numeric(
            baseline[col], errors="coerce"
        ).dropna()

        recent_values = pd.to_numeric(
            recent[col], errors="coerce"
        ).dropna()

        if len(old_values) < 2 or len(recent_values) < 2:
            continue

        old_avg = float(old_values.median())
        recent_avg = float(recent_values.mean())

        if not np.isfinite(old_avg) or not np.isfinite(recent_avg):
            continue

        # Positive deterioration means performance moved
        # in the unfavourable direction.
        if config["direction"] == "higher":
            deterioration = old_avg - recent_avg
        else:
            deterioration = recent_avg - old_avg

        if abs(old_avg) > 1e-9:
            change_pct = deterioration / abs(old_avg) * 100
        else:
            change_pct = np.nan

        rank_score = (
            change_pct if np.isfinite(change_pct)
            else deterioration
        )

        results.append({
            "metric": metric_name,
            "baseline": old_avg,
            "recent": recent_avg,
            "deterioration": deterioration,
            "change_pct": change_pct,
            "rank_score": rank_score,
            "advice": config["advice"],
        })

    if not results:
        return {
            "ready": False,
            "message": (
                "No usable performance metrics were found. "
                "Check whether the player dataset contains ACS, ADR, "
                "KAST, K/D or other supported columns."
            ),
            "metrics": [],
            "recommendations": [],
        }

    deteriorated = [
        item for item in results
        if item["deterioration"] > 0
    ]

    deteriorated.sort(
        key=lambda x: x["rank_score"],
        reverse=True,
    )

    if deteriorated:
        recommendations = [
            item["advice"] for item in deteriorated[:3]
        ]
    else:
        recommendations = [
            "No deterioration was observed across the tracked metrics.",
            "Continue monitoring future maps to confirm whether this pattern persists.",
        ]

    return {
        "ready": True,
        "message": (
            f"Compared {len(recent)} recent maps against "
            f"{len(baseline)} earlier maps."
        ),
        "metrics": results,
        "recommendations": recommendations,
    }


# ---------- Player Advisor Popup ----------

@st.dialog("GAME SENSE PLAYER ADVISOR", width="large")
def show_player_advisor(
    player_name,
    player_df,
    current_gti,
    current_paber,
):

    report = build_player_advisor_report(player_df)

    st.caption(
        f"Player: {player_name} · Recent performance intelligence"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "RECENT GTI",
        f"{current_gti:.1f}" if pd.notna(current_gti) else "—",
    )

    c2.metric(
        "RECENT PABER",
        f"{current_paber:.1f}" if pd.notna(current_paber) else "—",
    )

    if not report["ready"]:
        st.warning(report["message"])

        st.info(
            "The advisor will not generate unsupported recommendations "
            "when historical data is insufficient."
        )
        return

    st.success(report["message"])

    metrics_df = pd.DataFrame(report["metrics"])

    display_df = metrics_df[
        ["metric", "baseline", "recent", "change_pct"]
    ].copy()

    display_df.columns = [
        "Metric",
        "Earlier Baseline",
        "Latest 3-Map Average",
        "Deterioration (%)",
    ]

    for col in [
        "Earlier Baseline",
        "Latest 3-Map Average",
        "Deterioration (%)",
    ]:
        display_df[col] = display_df[col].round(2)

    st.markdown("### Recent Performance vs Historical Baseline")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Key Performance Observations")

    deteriorated = [
        item for item in report["metrics"]
        if item["deterioration"] > 0
    ]

    deteriorated.sort(
        key=lambda x: x["rank_score"],
        reverse=True,
    )

    if deteriorated:
        for item in deteriorated[:3]:

            percentage = (
                f"{item['change_pct']:.1f}%"
                if np.isfinite(item["change_pct"])
                else "absolute change"
            )

            st.markdown(
                f"- **{item['metric']}**: "
                f"Recent performance moved in an unfavourable "
                f"direction ({percentage} relative to baseline)."
            )
    else:
        st.markdown(
            "No tracked metric shows deterioration against "
            "the earlier player baseline."
        )

    st.markdown("### Custom GameSense Recommendations")

    for index, recommendation in enumerate(
        report["recommendations"], start=1
    ):
        st.markdown(f"**{index}.** {recommendation}")

    st.caption(
        "This report uses recorded match statistics and "
        "rule-based comparisons. It does not measure emotions, "
        "diagnose psychological tilt or guarantee future performance."
    )


# ---------- Header ----------
st.markdown(
    '<div class="gs-eyebrow">VALORANT CHAMPIONS TOUR · PLAYER INTELLIGENCE</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<h1>GAMESENSE <span style='color:#ff4655'>AI</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gs-subtitle">Performance deterioration monitoring · '
    'Player-adaptive risk signals · Next-map research</div>',
    unsafe_allow_html=True,
)

if gti.empty:
    st.error(
        "Missing data/features/gamesense_gti.csv. Keep the existing "
        "project folder structure and place the GTI CSV in data/features."
    )
    st.stop()

score_col = find_column(gti, ["gti_score", "GTI", "gti"])
level_col = find_column(gti, ["gti_level", "gti_category", "tilt_level"])
player_col = find_column(gti, ["player_name", "name", "player"])
player_id_col = find_column(gti, ["player_id"])
map_col = find_column(gti, ["map_name", "map"])
result_col = find_column(gti, ["map_result", "result"])
role_col = find_column(gti, ["player_role", "role"])
event_col = find_column(gti, ["event", "event_name"])
match_col = find_column(gti, ["match_id", "series_id"])
timestamp_col = find_column(gti, ["utc_timestamp", "timestamp", "date"])

if score_col is None:
    st.error("No GTI score column found in gamesense_gti.csv.")
    st.stop()

gti[score_col] = pd.to_numeric(gti[score_col], errors="coerce")

# PABER data can be joined safely on available player-map identifiers.
paber_score_col = find_column(
    paber, ["paber_score", "PABER", "paber", "paber_risk_score"]
)
paber_level_col = find_column(
    paber, ["paber_level", "paber_category", "paber_band"]
)
paber_persistence_col = find_column(
    paber, ["paber_persistence", "persistence_score"]
)

join_keys = [
    c for c in ["match_id", "game_id", "player_id"]
    if c in gti.columns and c in paber.columns
]

if not join_keys and player_id_col and player_id_col in paber.columns:
    join_keys = [player_id_col]

if not paber.empty and paber_score_col and join_keys:
    paber_small = paber[
        join_keys + [
            c for c in [
                paber_score_col,
                paber_level_col,
                paber_persistence_col,
            ]
            if c and c not in join_keys
        ]
    ].copy()

    paber_small = paber_small.drop_duplicates(join_keys, keep="last")

    data = gti.merge(
        paber_small,
        on=join_keys,
        how="left",
        suffixes=("", "_paber"),
    )

    paber_score_col = (
        paber_score_col
        if paber_score_col in data.columns
        else f"{paber_score_col}_paber"
    )
    paber_level_col = (
        paber_level_col
        if paber_level_col in data.columns
        else (
            f"{paber_level_col}_paber"
            if paber_level_col else None
        )
    )
    paber_persistence_col = (
        paber_persistence_col
        if paber_persistence_col in data.columns
        else (
            f"{paber_persistence_col}_paber"
            if paber_persistence_col else None
        )
    )
else:
    data = gti.copy()

if paber_score_col and paber_score_col in data.columns:
    data[paber_score_col] = pd.to_numeric(
        data[paber_score_col], errors="coerce"
    )


# ---------- Sidebar filters ----------
with st.sidebar:
    st.markdown("## 🎯 GAME CONTROL")
    st.caption("Filter the dataset to update dashboard views.")

    filtered = data.copy()

    filter_specs = [
        ("Tournament / Event", find_column(filtered, ["event", "event_name"])),
        ("Map", find_column(filtered, ["map_name", "map"])),
        ("Role", find_column(filtered, ["player_role", "role"])),
    ]

    identity_col = player_col or player_id_col
    filter_specs.append(("Player", identity_col))

    for label, col in filter_specs:
        if col and col in filtered.columns:
            vals = sorted(
                filtered[col].dropna().astype(str).unique().tolist()
            )
            selected = st.multiselect(
                label,
                vals,
                default=[],
                placeholder=f"All {label.lower()}s",
            )
            if selected:
                filtered = filtered[
                    filtered[col].astype(str).isin(selected)
                ]

    low, high = st.slider("GTI range", 0, 100, (0, 100))
    filtered = filtered[
        filtered[score_col].between(low, high, inclusive="both")
    ]

    if paber_score_col and paber_score_col in filtered.columns:
        st.caption(
            f"PABER available: "
            f"{filtered[paber_score_col].notna().sum():,} records"
        )

    st.divider()
    st.caption(
        f"Showing **{len(filtered):,}** of **{len(data):,}** "
        "player-map records."
    )

    if st.button("↻ Refresh dashboard data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ---------- Top KPIs ----------
scores = filtered[score_col].dropna()

players_count = (
    filtered[player_id_col].nunique()
    if player_id_col
    else (
        filtered[player_col].nunique()
        if player_col else 0
    )
)

avg_gti = scores.mean() if not scores.empty else np.nan
high_count = int((scores > 75).sum())

paber_values = (
    pd.to_numeric(filtered[paber_score_col], errors="coerce").dropna()
    if paber_score_col and paber_score_col in filtered.columns
    else pd.Series(dtype=float)
)

avg_paber = (
    paber_values.mean()
    if not paber_values.empty else np.nan
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("PLAYER-MAP RECORDS", f"{len(filtered):,}")
k2.metric("PLAYERS", f"{players_count:,}")
k3.metric(
    "AVG GTI",
    f"{avg_gti:.1f}" if pd.notna(avg_gti) else "—",
    help="Rule-based deterioration indicator, 0–100.",
)
k4.metric(
    "HIGH GTI SIGNALS",
    f"{high_count:,}",
    help="Records with GTI above 75.",
)
k5.metric(
    "AVG PABER",
    f"{avg_paber:.1f}" if pd.notna(avg_paber) else "—",
    help="Player-adaptive risk score where historical data is available.",
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Navigation ----------
tabs = st.tabs([
    "◈ COMMAND CENTER",
    "♙ PLAYER INTELLIGENCE",
    "▤ MAP & ROLE ANALYSIS",
    "⌁ GTI / PABER EXPLORER",
    "◉ VALIDATION",
    "⚙ MODEL LAB",
])
# 1. Overview
with tabs[0]:
    st.markdown("## Command Center")
    st.caption(
        "A filtered snapshot of historical player-map performance signals."
    )

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### GTI signal distribution")

        if level_col and not filtered.empty:
            dist = (
                filtered[level_col]
                .fillna("Insufficient Data")
                .value_counts()
                .rename_axis("Signal")
                .reset_index(name="Records")
            )

            fig = px.pie(
                dist,
                names="Signal",
                values="Records",
                hole=.62,
                color="Signal",
                color_discrete_map=CATEGORY_COLORS,
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
            )

            style_fig(fig, 370)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GTI category column unavailable.")

    with right:
        st.markdown("### Average GTI by map result")

        if result_col and not filtered.empty:
            grp = (
                filtered.groupby(result_col, dropna=False)[score_col]
                .mean()
                .reset_index()
            )

            grp.columns = ["Result", "Average GTI"]

            fig = px.bar(
                grp,
                x="Result",
                y="Average GTI",
                color="Result",
                color_discrete_map={
                    "Win": CYAN,
                    "Loss": RED,
                },
                text_auto=".1f",
            )

            fig.update_traces(marker_line_width=0)
            style_fig(fig, 370)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map result column unavailable.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Highest GTI observations")

        cols = [
            player_col,
            player_id_col,
            map_col,
            result_col,
            role_col,
            score_col,
            level_col,
        ]

        top = filtered.nlargest(12, score_col)
        show_table(top, cols, 340)

    with c2:
        st.markdown("### GTI vs PABER distribution")

        if paber_score_col and paber_score_col in filtered.columns:
            chart = (
                filtered[[score_col, paber_score_col]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )

            if not chart.empty:
                fig = px.scatter(
                    chart,
                    x=score_col,
                    y=paber_score_col,
                    opacity=.55,
                    labels={
                        score_col: "GTI score",
                        paber_score_col: "PABER score",
                    },
                )

                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=100,
                    y1=100,
                    line=dict(
                        color="#758195",
                        dash="dash",
                        width=1,
                    ),
                )

                style_fig(fig, 340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "No overlapping GTI and PABER scores "
                    "in the current filter."
                )
        else:
            st.info(
                "PABER file or compatible score column not found. "
                "Add gamesense_paber.csv to data/features."
            )

    st.markdown(
        '<div class="gs-warning">Interpretation: GTI and PABER '
        'represent observable statistical performance deterioration. '
        'They do not measure a player’s actual mental state or '
        'diagnose psychological tilt.</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# 2. PLAYER INTELLIGENCE + CHATBOT
# ============================================================

with tabs[1]:

    st.markdown("## Player Intelligence")

    id_col = player_col or player_id_col

    if not id_col or filtered.empty:

        st.info("Player identity data is unavailable under the current filters.")

    else:

        options = sorted(
            filtered[id_col].dropna().astype(str).unique().tolist()
        )

        selected = st.selectbox(
            "Select a player",
            options,
            index=None,
            placeholder="Search player...",
            key="player_intelligence_select",
        )

        if selected:

            pdf = sort_records(
                filtered[
                    filtered[id_col].astype(str) == selected
                ].copy()
            )

            ps = pd.to_numeric(pdf[score_col], errors="coerce")

            pavg = ps.mean()
            pmax = ps.max()
            p_high = int((ps > 75).sum())

            recent_gti = ps.tail(3).dropna()

            advisor_gti = (
                float(recent_gti.mean())
                if not recent_gti.empty else np.nan
            )

            advisor_paber = np.nan

            if paber_score_col and paber_score_col in pdf.columns:

                paber_series = pd.to_numeric(
                    pdf[paber_score_col],
                    errors="coerce",
                )

                recent_paber = paber_series.tail(3).dropna()

                if not recent_paber.empty:
                    advisor_paber = float(recent_paber.mean())

            # Build player analysis automatically
            advisor_report = build_player_advisor_report(pdf)

            # ---------- Player KPIs ----------

            pc1, pc2, pc3, pc4 = st.columns(4)

            pc1.metric("MAPS / RECORDS", f"{len(pdf):,}")

            pc2.metric(
                "AVERAGE GTI",
                f"{pavg:.1f}" if pd.notna(pavg) else "—",
            )

            pc3.metric(
                "PEAK GTI",
                f"{pmax:.1f}" if pd.notna(pmax) else "—",
            )

            pc4.metric("GTI > 75", f"{p_high:,}")

            # ---------- Main two-column layout ----------

            st.divider()

            left, right = st.columns(
                [1.75, 1],
                gap="large",
            )

            # ================================================
            # LEFT: PLAYER ANALYTICS
            # ================================================

            with left:

                st.markdown("### Performance Timeline")

                pdf = pdf.copy()
                pdf["Map Record"] = np.arange(1, len(pdf) + 1)

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=pdf["Map Record"],
                        y=pdf[score_col],
                        mode="lines+markers",
                        name="GTI",
                        line=dict(color=RED, width=2.5),
                    )
                )

                if paber_score_col and paber_score_col in pdf.columns:

                    fig.add_trace(
                        go.Scatter(
                            x=pdf["Map Record"],
                            y=pdf[paber_score_col],
                            mode="lines+markers",
                            name="PABER",
                            line=dict(color=CYAN, width=2.2),
                        )
                    )

                fig.add_hrect(
                    y0=0, y1=25,
                    fillcolor=CYAN,
                    opacity=.05,
                    line_width=0,
                )

                fig.add_hrect(
                    y0=50, y1=75,
                    fillcolor=AMBER,
                    opacity=.05,
                    line_width=0,
                )

                fig.add_hrect(
                    y0=75, y1=100,
                    fillcolor=RED,
                    opacity=.06,
                    line_width=0,
                )

                fig.update_yaxes(
                    range=[0, 100],
                    title="Risk score",
                )

                fig.update_xaxes(
                    title="Chronological player-map record"
                )

                style_fig(fig, 390)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                st.markdown("### Player Map History")

                cols = [
                    timestamp_col,
                    match_col,
                    map_col,
                    result_col,
                    role_col,
                    score_col,
                    level_col,
                    paber_score_col,
                    paber_level_col,
                ]

                show_table(pdf, cols, 330)

            # ================================================
            # RIGHT: GAME SENSE CHATBOT
            # ================================================

            with right:

                st.markdown("### ✦ GameSense Advisor")

                st.caption(
                    f"Player analysis · {selected}"
                )

                st.markdown(
                    """
                    <div class="gs-card">
                        <div class="gs-card-label">
                            PLAYER ADVISOR
                        </div>
                        <div class="gs-card-value">
                            Performance Assistant
                        </div>
                        <div class="gs-card-note">
                            Ask questions about recent performance,
                            GTI, PABER and areas to review.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("")

                # Reset conversation when player changes
                if st.session_state.get(
                    "advisor_chat_player"
                ) != selected:

                    st.session_state.advisor_chat_player = selected

                    st.session_state.advisor_chat_history = []

                    if advisor_report["ready"]:

                        weak = [
                            m for m in advisor_report["metrics"]
                            if m["deterioration"] > 0
                        ]

                        if weak:
                            opening = (
                                f"I've analysed **{selected}**. "
                                f"I found {len(weak)} tracked metrics "
                                "moving in an unfavourable direction "
                                "compared with their earlier baseline. "
                                "Ask me which areas need attention."
                            )
                        else:
                            opening = (
                                f"I've analysed **{selected}**. "
                                "No deterioration was identified across "
                                "the available tracked metrics compared "
                                "with the earlier baseline."
                            )

                    else:

                        opening = (
                            f"I'm ready to analyse **{selected}**, "
                            "but I need more historical map data. "
                            + advisor_report["message"]
                        )

                    st.session_state.advisor_chat_history.append({
                        "role": "assistant",
                        "content": opening,
                    })

                # Suggested questions
                st.markdown("**Suggested questions**")

                suggestions = [
                    "Give me an overall performance summary",
                    "What are the biggest weaknesses?",
                    "How has ACS changed?",
                    "Explain the GTI and PABER scores",
                    "What should this player improve?",
                ]

                chosen_question = st.selectbox(
                    "Ask the advisor",
                    suggestions,
                    key=f"advisor_suggestion_{selected}",
                    label_visibility="collapsed",
                )

                if st.button(
                    "Ask suggested question",
                    use_container_width=True,
                    key=f"advisor_ask_{selected}",
                ):

                    answer = chatbot.answer(
    question=chosen_question,
    player=selected,
    player_df=pdf,
    report=advisor_report,
    current_gti=advisor_gti,
    current_paber=advisor_paber,
    full_data=data,
    player_column=player_col,
    score_column=score_col,
    map_column=map_col,
    role_column=role_col,
    paber_column=paber_score_col,
    history=st.session_state.advisor_chat_history,
)

                    st.session_state.advisor_chat_history.extend([
                        {
                            "role": "user",
                            "content": chosen_question,
                        },
                        {
                            "role": "assistant",
                            "content": answer,
                        },
                    ])

                # Conversation history
                st.markdown("**Conversation**")

                with st.container(height=350, border=True):

                    for message in st.session_state.advisor_chat_history:

                        with st.chat_message(message["role"]):

                            st.markdown(message["content"])

                # Custom question input
                with st.form(
                    key=f"advisor_chat_form_{selected}",
                    clear_on_submit=True,
                ):

                    user_question = st.text_input(
                        "Your question",
                        placeholder="Ask about this player's performance...",
                    )

                    send = st.form_submit_button(
                        "Send message ➤",
                        use_container_width=True,
                    )

                    if send and user_question.strip():

                        answer = chatbot.answer(
    question=user_question,
    player=selected,
    player_df=pdf,
    report=advisor_report,
    current_gti=advisor_gti,
    current_paber=advisor_paber,
    full_data=data,
    player_column=player_col,
    score_column=score_col,
    map_column=map_col,
    role_column=role_col,
    paber_column=paber_score_col,
    history=st.session_state.advisor_chat_history,
)

                        st.session_state.advisor_chat_history.extend([
                            {
                                "role": "user",
                                "content": user_question.strip(),
                            },
                            {
                                "role": "assistant",
                                "content": answer,
                            },
                        ])

                        st.rerun()

                st.caption(
                    "GameSense Advisor uses recorded statistics "
                    "and rule-based analysis. It does not measure "
                    "a player's actual mental state."
                )

        else:

            st.info(
                "Select a player to open their performance dashboard "
                "and personalized GameSense Advisor."
            )
# 4. Map & role
with tabs[2]:
    st.markdown("## Map & Role Analysis")

    a, b = st.columns(2)

    with a:
        st.markdown("### GTI by map")

        if map_col and not filtered.empty:
            by_map = (
                filtered.groupby(map_col)[score_col]
                .agg(["mean", "count"])
                .reset_index()
            )

            by_map = by_map.sort_values(
                "mean",
                ascending=True,
            )

            fig = px.bar(
                by_map,
                x="mean",
                y=map_col,
                orientation="h",
                text="mean",
                labels={
                    "mean": "Average GTI",
                    map_col: "Map",
                    "count": "Records",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside",
                marker_color=RED,
            )

            style_fig(
                fig,
                max(350, 30 * len(by_map) + 100),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Map column unavailable.")

    with b:
        st.markdown("### GTI by role")

        if role_col and not filtered.empty:
            by_role = (
                filtered.groupby(role_col)[score_col]
                .agg(["mean", "count"])
                .reset_index()
            )

            fig = px.bar(
                by_role,
                x=role_col,
                y="mean",
                color=role_col,
                text="mean",
                labels={
                    "mean": "Average GTI",
                    role_col: "Role",
                },
            )

            fig.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside",
            )

            style_fig(fig, 370)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Role column unavailable.")

    if result_col and map_col:
        st.markdown("### Win / loss comparison by map")

        cross = (
            filtered.groupby([map_col, result_col])[score_col]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            cross,
            x=map_col,
            y=score_col,
            color=result_col,
            barmode="group",
            color_discrete_map={
                "Win": CYAN,
                "Loss": RED,
            },
            labels={
                score_col: "Average GTI",
                map_col: "Map",
            },
        )

        style_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)
        # 5. Explorer
with tabs[3]:
    st.markdown("## GTI / PABER Explorer")

    st.caption(
        "Inspect and export player-map observations. "
        "Missing PABER values indicate insufficient historical baseline."
    )

    sort_choice = st.radio(
        "Sort by",
        ["Highest GTI", "Lowest GTI", "Highest PABER"],
        horizontal=True,
    )

    sort_col = (
        paber_score_col
        if (
            sort_choice == "Highest PABER"
            and paber_score_col
            and paber_score_col in filtered.columns
        )
        else score_col
    )

    explorer = filtered.sort_values(
        sort_col,
        ascending=sort_choice == "Lowest GTI",
        na_position="last",
    )

    visible = [
        player_col,
        player_id_col,
        match_col,
        map_col,
        result_col,
        role_col,
        score_col,
        level_col,
        paber_score_col,
        paber_level_col,
        paber_persistence_col,
    ]

    show_table(explorer, visible, 500)

    st.download_button(
        "⬇ Download filtered records (CSV)",
        explorer.to_csv(index=False).encode("utf-8"),
        "gamesense_filtered_records.csv",
        "text/csv",
    )


# 6. Validation
with tabs[4]:
    st.markdown("## Research Validation")

    st.caption(
        "Next-map performance is an operational outcome label, "
        "not a direct measurement of psychological tilt."
    )

    vdf = (
        validation_paber
        if not validation_paber.empty
        else validation
    )

    if vdf.empty:
        st.warning(
            "Validation data not found. Expected "
            "gamesense_validation_paber.csv or gamesense_validation.csv."
        )

    else:
        target_col = find_column(
            vdf,
            [
                "next_map_poor_performance",
                "poor_next_map_performance",
                "poor_next_performance",
                "target",
                "label",
            ],
        )

        v_gti_col = find_column(
            vdf,
            ["gti_score", "GTI", "gti"],
        )

        v_paber_col = find_column(
            vdf,
            ["paber_score", "PABER", "paber"],
        )

        v_level_col = find_column(
            vdf,
            ["gti_group", "gti_level", "gti_category"],
        )

        v_rate_col = find_column(
            vdf,
            ["next_rating_change_pct", "next_rating_change"],
        )

        v_acs_col = find_column(
            vdf,
            ["next_acs_change_pct", "next_acs_change"],
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "VALIDATION PAIRS",
            f"{len(vdf):,}",
        )

        m2.metric(
            "POOR NEXT-MAP RATE",
            (
                f"{pd.to_numeric(vdf[target_col], errors='coerce').mean()*100:.1f}%"
                if target_col else "—"
            ),
        )

        m3.metric(
            "PABER AVAILABLE",
            (
                f"{pd.to_numeric(vdf[v_paber_col], errors='coerce').notna().sum():,}"
                if v_paber_col else "—"
            ),
        )

        left, right = st.columns(2)

        with left:
            st.markdown("### Poor-performance rate by GTI band")

            if target_col and v_gti_col:
                tmp = vdf[[target_col, v_gti_col]].copy()

                tmp[target_col] = pd.to_numeric(
                    tmp[target_col],
                    errors="coerce",
                )

                tmp[v_gti_col] = pd.to_numeric(
                    tmp[v_gti_col],
                    errors="coerce",
                )

                tmp = tmp.dropna()

                tmp["GTI band"] = pd.cut(
                    tmp[v_gti_col],
                    [-0.01, 25, 50, 75, 100],
                    labels=[
                        "Stable",
                        "Mild",
                        "Elevated",
                        "Strong",
                    ],
                )

                rates = (
                    tmp.groupby("GTI band", observed=False)[target_col]
                    .mean()
                    .mul(100)
                    .reset_index(name="Poor next-map rate (%)")
                )

                fig = px.bar(
                    rates,
                    x="GTI band",
                    y="Poor next-map rate (%)",
                    text="Poor next-map rate (%)",
                    color="GTI band",
                    color_discrete_map={
                        "Stable": CYAN,
                        "Mild": AMBER,
                        "Elevated": "#ff8a4c",
                        "Strong": RED,
                    },
                )

                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )

                style_fig(fig, 360)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(
                    "GTI and target columns not both found."
                )

        with right:
            st.markdown("### Next-map outcome changes")

            if v_rate_col or v_acs_col:
                cols_to_plot = [
                    c for c in [v_rate_col, v_acs_col]
                    if c
                ]

                plot = (
                    vdf[cols_to_plot]
                    .apply(pd.to_numeric, errors="coerce")
                )

                plot = plot.melt(
                    var_name="Metric",
                    value_name="Change (%)",
                ).dropna()

                fig = px.histogram(
                    plot,
                    x="Change (%)",
                    color="Metric",
                    barmode="overlay",
                    opacity=.65,
                    nbins=30,
                )

                style_fig(fig, 360)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(
                    "Next-map rating / ACS change columns unavailable."
                )

        if v_paber_col and target_col:
            st.markdown("### PABER score vs next-map outcome")

            tmp = (
                vdf[[v_paber_col, target_col]]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )

            if not tmp.empty:
                tmp["Outcome"] = (
                    tmp[target_col]
                    .map({
                        0: "No poor performance",
                        1: "Poor performance",
                    })
                    .fillna(tmp[target_col].astype(str))
                )

                fig = px.box(
                    tmp,
                    x="Outcome",
                    y=v_paber_col,
                    color="Outcome",
                    color_discrete_map={
                        "No poor performance": CYAN,
                        "Poor performance": RED,
                    },
                )

                style_fig(fig, 350)
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Inspect validation dataset"):
            show_table(vdf.head(100), height=400)


# 7. Model lab
with tabs[5]:
    st.markdown("## Model Lab")

    st.caption(
        "Chronological holdout comparison. Scores below are "
        "the saved experiment outputs, not live retraining."
    )

    if not comparison.empty:
        model_name_col = find_column(
            comparison,
            ["model"],
        )

        metric_map = {
            "Accuracy": ["accuracy"],
            "Balanced accuracy": ["balanced_accuracy"],
            "Precision": ["precision"],
            "Recall": ["recall"],
            "F1": ["f1"],
            "ROC-AUC": ["roc_auc"],
            "PR-AUC": ["pr_auc"],
        }

        if model_name_col:
            display = comparison.copy()

            metric_cols = [
                find_column(display, names)
                for names in metric_map.values()
            ]

            metric_cols = [
                c for c in metric_cols if c
            ]

            for c in metric_cols:
                display[c] = pd.to_numeric(
                    display[c],
                    errors="coerce",
                )

            st.markdown("### Saved model comparison")

            show_table(display, height=300)

            available_metrics = [
                x for x in metric_map
                if find_column(display, metric_map[x])
            ]

            if available_metrics:
                metric_choice = st.selectbox(
                    "Visualize metric",
                    available_metrics,
                    index=0,
                )

                chosen_metric = find_column(
                    display,
                    metric_map[metric_choice],
                )

                fig = px.bar(
                    display,
                    x=model_name_col,
                    y=chosen_metric,
                    color=model_name_col,
                    text=chosen_metric,
                    labels={
                        chosen_metric: metric_choice,
                        model_name_col: "Model",
                    },
                )

                fig.update_traces(
                    texttemplate="%{text:.3f}",
                    textposition="outside",
                )

                style_fig(fig, 390)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "Model comparison CSV not found. "
            "Expected gamesense_paber_comparison.csv."
        )

    st.markdown(
        "### Bootstrap uncertainty: GTI + PABER vs GTI only"
    )

    if not bootstrap_summary.empty:
        metric_col = find_column(
            bootstrap_summary,
            ["metric"],
        )

        mean_col = find_column(
            bootstrap_summary,
            ["mean_difference"],
        )

        lower_col = find_column(
            bootstrap_summary,
            ["ci_lower"],
        )

        upper_col = find_column(
            bootstrap_summary,
            ["ci_upper"],
        )

        fraction_col = find_column(
            bootstrap_summary,
            ["fraction_positive"],
        )

        if all([
            metric_col,
            mean_col,
            lower_col,
            upper_col,
        ]):
            bs = bootstrap_summary.copy()

            for c in [
                mean_col,
                lower_col,
                upper_col,
                fraction_col,
            ]:
                if c:
                    bs[c] = pd.to_numeric(
                        bs[c],
                        errors="coerce",
                    )

            for _, row in bs.iterrows():
                label = (
                    str(row[metric_col])
                    .replace("_difference", "")
                    .replace("_", " ")
                    .title()
                )

                lo = row[lower_col]
                hi = row[upper_col]

                st.markdown(
                    f"**{label}** · Difference "
                    f"{row[mean_col]:+.4f} · "
                    f"95% CI [{lo:+.4f}, {hi:+.4f}]"
                )

                if pd.notna(lo) and pd.notna(hi):
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=[lo, hi],
                            y=[0, 0],
                            mode="lines",
                            line=dict(
                                color=CYAN if lo > 0 else AMBER,
                                width=7,
                            ),
                            showlegend=False,
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[row[mean_col]],
                            y=[0],
                            mode="markers",
                            marker=dict(
                                color=RED,
                                size=12,
                            ),
                            showlegend=False,
                        )
                    )

                    fig.add_vline(
                        x=0,
                        line_dash="dash",
                        line_color="#d0d7e2",
                    )

                    fig.update_yaxes(
                        visible=False,
                        range=[-.5, .5],
                    )

                    fig.update_xaxes(
                        title="Combined model minus GTI-only"
                    )

                    style_fig(fig, 125)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                if fraction_col and pd.notna(row[fraction_col]):
                    st.caption(
                        f"Bootstrap fraction above zero: "
                        f"{row[fraction_col]:.1%}. "
                        "This is not a p-value."
                    )

        else:
            show_table(bootstrap_summary)

    else:
        st.info(
            "Bootstrap summary not found. Run "
            "scripts/13_evaluate_paber_significance.py "
            "to generate it."
        )

    st.markdown("### Research interpretation")

    st.markdown("""
    <div class="gs-warning">
    Current experiment: GTI + PABER improves F1 and balanced
    accuracy on the reported chronological holdout, while ROC-AUC
    is almost unchanged. The bootstrap confidence intervals include
    zero for all three metrics. Treat PABER as a promising
    complementary signal; the evidence does not establish a
    statistically reliable improvement.
    </div>
    """, unsafe_allow_html=True)


# ---------- Footer ----------
st.divider()

st.caption(
    "GAMESENSE AI · Experimental player-performance analytics · "
    "Historical association is not causation · "
    "Human review required for decisions."
)