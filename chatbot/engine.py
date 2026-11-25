
import re
from difflib import get_close_matches

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. NLP INTENT RECOGNITION
# ==========================================================

INTENTS = {
    "overview": [
        "analyse this player",
        "give me an overall performance summary",
        "how is this player performing",
        "tell me about this player's performance",
        "show player performance",
        "give me a player report",
    ],
    "weaknesses": [
        "what are the biggest weaknesses",
        "which stats have declined",
        "what is the weakest performance metric",
        "what is going wrong",
        "where is the player struggling",
        "what has deteriorated",
    ],
    "metric": [
        "analyse ACS",
        "how has ADR changed",
        "explain the KAST statistics",
        "what happened to his deaths",
        "show the kill death ratio",
        "analyse first death rate",
        "how are multikills changing",
    ],
    "gti": [
        "explain the GTI score",
        "why is GTI high",
        "what contributes to the tilt index",
        "how is the GameSense Tilt Index calculated",
        "explain this deterioration score",
    ],
    "paber": [
        "explain PABER",
        "what does PABER mean",
        "why is PABER different from GTI",
        "explain the player adaptive baseline",
    ],
    "trend": [
        "is the player improving",
        "is performance getting worse",
        "show performance trends",
        "how has performance changed over time",
        "is the player becoming consistent",
    ],
    "map": [
        "which map is the player weakest on",
        "analyse map performance",
        "which map has the highest GTI",
        "compare performance across maps",
    ],
    "role": [
        "analyse role performance",
        "how does this player perform in their role",
        "compare player role statistics",
    ],
    "recommendations": [
        "what should this player improve",
        "give performance recommendations",
        "how can this player improve",
        "suggest areas to work on",
        "what should the player focus on",
    ],
    "comparison": [
        "compare two players",
        "compare this player with another player",
        "who has better statistics",
        "which player performs better",
    ],
    "next_risk": [
        "what is the next map risk",
        "what does the prediction model say",
        "predict poor performance on the next map",
        "what is the next map prediction",
    ],
    "methodology": [
        "how does GameSense AI work",
        "explain the project methodology",
        "explain the dataset",
        "how was GTI calculated",
        "what are the limitations",
        "how was the model evaluated",
    ],
}


METRIC_ALIASES = {
    "ACS": ["acs", "combat score", "combat scoring"],
    "ADR": ["adr", "damage per round", "damage"],
    "KAST": ["kast"],
    "K/D Ratio": ["k/d", "kd ratio", "kill death ratio"],
    "Deaths": ["deaths", "death count"],
    "First Death Rate": [
        "first death",
        "first deaths",
        "opening deaths",
    ],
    "Multikills": ["multikill", "multikills", "multi kills"],
}


class GameSenseNLP:

    def __init__(self):

        self.labels = []
        self.examples = []

        for intent, questions in INTENTS.items():
            for question in questions:
                self.labels.append(intent)
                self.examples.append(question)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            stop_words="english",
            sublinear_tf=True,
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

        self.word_matrix = self.vectorizer.fit_transform(
            self.examples
        )

        self.char_matrix = self.char_vectorizer.fit_transform(
            self.examples
        )

    def detect(self, question):

        q = question.lower().strip()

        if not q:
            return "unknown", 0.0

        # Strong intent indicators
        if "compare" in q or "versus" in q or " vs " in f" {q} ":
            return "comparison", 1.0

        word_vec = self.vectorizer.transform([q])
        char_vec = self.char_vectorizer.transform([q])

        word_scores = cosine_similarity(
            word_vec, self.word_matrix
        )[0]

        char_scores = cosine_similarity(
            char_vec, self.char_matrix
        )[0]

        # Combine word and character similarity
        scores = (0.65 * word_scores) + (0.35 * char_scores)

        # Use the best matching example for each intent
        intent_scores = {}

        for index, intent in enumerate(self.labels):
            intent_scores[intent] = max(
                intent_scores.get(intent, 0),
                scores[index],
            )

        best_intent = max(
            intent_scores,
            key=intent_scores.get,
        )

        confidence = float(intent_scores[best_intent])

        if confidence < 0.12:
            return "unknown", confidence

        return best_intent, confidence

    def detect_metric(self, question):

        q = question.lower()

        for metric, aliases in METRIC_ALIASES.items():
            if any(alias in q for alias in aliases):
                return metric

        return None


# ==========================================================
# 2. ANALYTICS ENGINE
# ==========================================================

def analyse_metric(report, metric_name):

    if not report.get("ready"):
        return report.get("message", "Insufficient history.")

    matches = [
        item for item in report["metrics"]
        if item["metric"] == metric_name
    ]

    if not matches:
        return f"{metric_name} is unavailable for this player."

    item = matches[0]

    pct = item["change_pct"]

    change = (
        f"{pct:+.2f}%"
        if np.isfinite(pct)
        else f"{item['deterioration']:+.2f} absolute change"
    )

    if item["deterioration"] > 0:
        status = "moved in an unfavourable direction"
    elif item["deterioration"] < 0:
        status = "improved relative to the baseline"
    else:
        status = "remained unchanged"

    return (
        f"**{metric_name} analysis**\n\n"
        f"- Historical baseline: {item['baseline']:.2f}\n"
        f"- Recent average: {item['recent']:.2f}\n"
        f"- Change: {change}\n\n"
        f"This metric {status} compared with the earlier "
        "player baseline."
    )


def analyse_weaknesses(player, report):

    if not report.get("ready"):
        return report.get("message", "Insufficient history.")

    weak = sorted(
        [
            m for m in report["metrics"]
            if m["deterioration"] > 0
        ],
        key=lambda x: x["rank_score"],
        reverse=True,
    )

    if not weak:
        return (
            f"No deterioration was identified across the "
            f"tracked metrics for **{player}**."
        )

    lines = []

    for item in weak[:4]:
        pct = item["change_pct"]

        change = (
            f"{pct:.2f}%"
            if np.isfinite(pct)
            else f"{item['deterioration']:.2f}"
        )

        lines.append(
            f"- **{item['metric']}**: {change} deterioration "
            f"relative to baseline."
        )

    return (
        f"### Main areas to review for {player}\n\n"
        + "\n".join(lines)
    )


def analyse_map_role(
    player_df,
    group_column,
    score_column,
    label,
):

    if (
        not group_column
        or group_column not in player_df.columns
        or not score_column
    ):
        return f"{label} information is unavailable."

    result = player_df.copy()

    result[score_column] = pd.to_numeric(
        result[score_column],
        errors="coerce",
    )

    grouped = (
        result.groupby(group_column)[score_column]
        .agg(["mean", "count"])
        .dropna()
        .sort_values("mean")
    )

    if grouped.empty:
        return f"No usable {label.lower()} records found."

    lines = []

    for name, row in grouped.iterrows():
        lines.append(
            f"- **{name}**: average GTI "
            f"{row['mean']:.2f} across {int(row['count'])} records."
        )

    return (
        f"### {label} analysis\n\n"
        + "\n".join(lines)
    )

def analyse_map_stats(
    question,
    full_data,
    map_column,
    score_column,
    paber_column=None,
):
    if (
        full_data.empty
        or not map_column
        or map_column not in full_data.columns
        or score_column not in full_data.columns
    ):
        return "Map analysis data is unavailable."

    df = full_data.copy()

    df[score_column] = pd.to_numeric(
        df[score_column], errors="coerce"
    )

    if paber_column and paber_column in df.columns:
        df[paber_column] = pd.to_numeric(
            df[paber_column], errors="coerce"
        )

    # Find the map mentioned in the question, if any.
    maps = df[map_column].dropna().astype(str).unique()

    selected_map = next(
        (m for m in maps if m.lower() in question.lower()),
        None,
    )

    if selected_map:
        df = df[
            df[map_column].astype(str).str.lower()
            == selected_map.lower()
        ]

    grouped = df.groupby(map_column)

    summary = grouped[score_column].agg(
        ["mean", "count"]
    ).rename(columns={
        "mean": "avg_gti",
        "count": "records",
    })

    if paber_column and paber_column in df.columns:
        summary["avg_paber"] = grouped[paber_column].mean()

    if "map_result" in df.columns:
        results = df.pivot_table(
            index=map_column,
            columns="map_result",
            values=score_column,
            aggfunc="count",
            fill_value=0,
        )

        summary["wins"] = results.get("Win", 0)
        summary["losses"] = results.get("Loss", 0)

        total = summary["wins"] + summary["losses"]

        summary["win_rate"] = (
            summary["wins"] / total.replace(0, np.nan) * 100
        )

    if summary.empty:
        return "No map records found."

    lines = []

    for name, row in summary.iterrows():
        line = (
            f"- **{name}**: Average GTI "
            f"{row['avg_gti']:.2f}, "
            f"{int(row['records'])} records"
        )

        if "avg_paber" in summary.columns:
            if pd.notna(row["avg_paber"]):
                line += f", Average PABER {row['avg_paber']:.2f}"

        if "win_rate" in summary.columns:
            line += (
                f", Wins {int(row['wins'])}, "
                f"Losses {int(row['losses'])}, "
                f"Win rate {row['win_rate']:.1f}%"
            )

        lines.append(line)

    heading = (
        f"### Analysis for {selected_map}"
        if selected_map
        else "### Overall Map Analysis"
    )

    return heading + "\n\n" + "\n".join(lines)

def compare_players(
    question,
    full_data,
    player_column,
    score_column,
    paber_column=None,
):

    if (
        full_data.empty
        or not player_column
        or player_column not in full_data.columns
    ):
        return "Player comparison data is unavailable."

    names = (
        full_data[player_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    q = question.lower()

    found = [
        name for name in names
        if name.lower() in q
    ]

    if len(found) < 2:
        return (
            "Please mention both player names in your question. "
            "For example: Compare TenZ and aspas."
        )

    selected = found[:2]

    results = []

    for name in selected:

        subset = full_data[
            full_data[player_column].astype(str) == name
        ]

        gti_avg = pd.to_numeric(
            subset[score_column], errors="coerce"
        ).mean()

        paber_avg = (
            pd.to_numeric(
                subset[paber_column], errors="coerce"
            ).mean()
            if paber_column and paber_column in subset.columns
            else np.nan
        )

        results.append({
            "player": name,
            "gti": gti_avg,
            "paber": paber_avg,
            "records": len(subset),
        })

    lines = []

    for item in results:

        paber_text = (
            f"{item['paber']:.2f}"
            if pd.notna(item["paber"])
            else "Unavailable"
        )

        lines.append(
            f"**{item['player']}**\n"
            f"- Average GTI: {item['gti']:.2f}\n"
            f"- Average PABER: {paber_text}\n"
            f"- Records: {item['records']}"
        )

    return (
        "### Player comparison\n\n"
        + "\n\n".join(lines)
        + "\n\nLower average GTI/PABER indicates a lower "
        "recorded deterioration score. This is not a direct "
        "measure of overall player skill."
    )


# ==========================================================
# 3. RECOMMENDATION ENGINE
# ==========================================================

def generate_recommendations(report):

    if not report.get("ready"):
        return report.get("message", "Insufficient history.")

    weak = sorted(
        [
            item for item in report["metrics"]
            if item["deterioration"] > 0
        ],
        key=lambda x: x["rank_score"],
        reverse=True,
    )

    if not weak:
        return (
            "No specific deterioration-based recommendation "
            "was triggered by the available metrics. "
            "Continue monitoring future maps."
        )

    output = []

    for item in weak[:3]:
        output.append(
            f"**{item['metric']}**\n"
            f"{item['advice']}"
        )

    return (
        "### Suggested performance-review priorities\n\n"
        + "\n\n".join(output)
        + "\n\nThese are statistical review suggestions, "
        "not proof of the cause of a performance change."
    )


# ==========================================================
# 4. MAIN CONVERSATION ENGINE
# ==========================================================

class GameSenseChatbot:

    def __init__(self):
        self.nlp = GameSenseNLP()

    def answer(
        self,
        question,
        player,
        player_df,
        report,
        current_gti,
        current_paber,
        full_data,
        player_column,
        score_column,
        map_column=None,
        role_column=None,
        paber_column=None,
        history=None,
    ):

        q = question.lower().strip()

        previous_intent = None

        if history:
            for message in reversed(history):
                if message.get("role") == "user":
                    previous_intent, _ = self.nlp.detect(
                        message.get("content", "")
                    )
                    break

        intent, confidence = self.nlp.detect(question)


 # Recognise a map name typed on its own
        if (
            full_data is not None
            and not full_data.empty
            and map_column in full_data.columns
        ):
            available_maps = (
                full_data[map_column]
                .dropna()
                .astype(str)
                .unique()
            )

            if q in [m.lower().strip() for m in available_maps]:
                intent = "map"


        # Follow-up context
        follow_up = any(
            phrase in q
            for phrase in [
                "what about",
                "and his",
                "and her",
                "explain that",
                "tell me more",
                "why is that",
                "what else",
            ]
        )

        if follow_up and previous_intent not in [None, "unknown"]:
            if intent == "unknown":
                intent = previous_intent

        # Metric-specific question
        metric = self.nlp.detect_metric(question)

        if metric:
            return analyse_metric(report, metric)

        if intent == "overview":

            if not report.get("ready"):
                return report.get("message", "Insufficient data.")

            weak_count = sum(
                m["deterioration"] > 0
                for m in report["metrics"]
            )

            improving_count = sum(
                m["deterioration"] < 0
                for m in report["metrics"]
            )

            return (
                f"### Performance overview: {player}\n\n"
                f"- Recent GTI: "
                f"{current_gti:.2f}/100\n"
                if pd.notna(current_gti)
                else f"### Performance overview: {player}\n\n"
            ) + (
                f"- Recent PABER: "
                f"{current_paber:.2f}/100\n"
                if pd.notna(current_paber)
                else "- Recent PABER: unavailable\n"
            ) + (
                f"- Metrics moving unfavourably: {weak_count}\n"
                f"- Metrics improving: {improving_count}\n\n"
                f"{report['message']}"
            )

        if intent == "weaknesses":
            return analyse_weaknesses(player, report)

        if intent == "gti":

            if pd.isna(current_gti):
                return "A recent GTI score is unavailable."

            band = (
                "Stable" if current_gti <= 25 else
                "Mild Signs" if current_gti <= 50 else
                "Elevated" if current_gti <= 75 else
                "Strong Potential Tilt"
            )

            return (
                f"**{player}'s recent average GTI: "
                f"{current_gti:.2f}/100 ({band}).**\n\n"
                "GTI is a rule-based indicator using changes "
                "in recorded performance metrics. It is not "
                "a measurement of actual psychological tilt."
            )

        if intent == "paber":

            if pd.isna(current_paber):
                return "PABER is unavailable for the recent maps."

            return (
                f"**{player}'s recent average PABER: "
                f"{current_paber:.2f}/100.**\n\n"
                "PABER compares performance with a "
                "player-adaptive historical baseline. "
                "It is a statistical deterioration indicator."
            )

        if intent == "trend":

            if not report.get("ready"):
                return report.get("message", "Insufficient history.")

            weak = sum(
                m["deterioration"] > 0
                for m in report["metrics"]
            )

            improving = sum(
                m["deterioration"] < 0
                for m in report["metrics"]
            )

            if weak > improving:
                trend = "More tracked metrics are deteriorating than improving."
            elif improving > weak:
                trend = "More tracked metrics are improving than deteriorating."
            else:
                trend = "The tracked metrics show a mixed pattern."

            return (
                f"### Recent trend for {player}\n\n"
                f"{trend}\n\n"
                f"{report['message']}\n\n"
                "This compares the latest three maps against "
                "earlier available history."
            )

        if intent == "map":
         return analyse_map_stats(
        question,
        full_data,
        map_column,
        score_column,
        paber_column,
           )

        if intent == "role":
            return analyse_map_role(
                player_df,
                role_column,
                score_column,
                "Role",
            )

        if intent == "recommendations":
            return generate_recommendations(report)

        if intent == "comparison":
            return compare_players(
                question,
                full_data,
                player_column,
                score_column,
                paber_column,
            )

        if intent == "next_risk":
            return (
                "The next-map risk model is a separate prediction "
                "task. Its saved prediction output is not connected "
                "to this chatbot response yet. I won't invent a "
                "risk probability."
            )

        if intent == "methodology":
            return (
                "### GameSense AI methodology\n\n"
                "1. Historical VCT player-map data is prepared.\n"
                "2. Map-level performance features are engineered.\n"
                "3. GTI calculates rule-based deterioration.\n"
                "4. PABER compares performance against a "
                "player-adaptive baseline.\n"
                "5. Validation links current-map signals to "
                "the next map's performance outcome.\n"
                "6. Machine-learning models are evaluated using "
                "a chronological holdout.\n\n"
                "The project analyses observable performance "
                "patterns, not a player's actual mental state."
            )

        return (
            f"I'm analysing **{player}**. I can help with:\n\n"
            "- Overall performance\n"
            "- Weaknesses and metric changes\n"
            "- GTI and PABER explanations\n"
            "- Performance trends\n"
            "- Map and role analysis\n"
            "- Player comparisons\n"
            "- Improvement recommendations\n"
            "- Project methodology\n\n"
            "Try asking your question in another way."
        )
