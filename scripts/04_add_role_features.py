import pandas as pd
import ast

# GameSense AI - Role and Agent Feature Engineering

INPUT_FILE = "data/features/gamesense_features.csv"
OUTPUT_FILE = "data/features/gamesense_role_features.csv"

# Load feature dataset

df = pd.read_csv(INPUT_FILE)

print("Feature dataset loaded")
print("Rows:", len(df))

# Parse agents column

def parse_agents(value):

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(agent).strip().lower()
                for agent in parsed
            ]

    except (ValueError, SyntaxError):

        pass

    return []


df["agent_list"] = df["agents"].apply(parse_agents)

# Agent to role mapping

role_map = {

    # Duelists
    "jett": "Duelist",
    "raze": "Duelist",
    "reyna": "Duelist",
    "phoenix": "Duelist",
    "neon": "Duelist",
    "yoru": "Duelist",
    "iso": "Duelist",
    "waylay": "Duelist",

    # Initiators
    "sova": "Initiator",
    "breach": "Initiator",
    "skye": "Initiator",
    "kayo": "Initiator",
    "kay/o": "Initiator",
    "fade": "Initiator",
    "gekko": "Initiator",
    "tejo": "Initiator",

    # Controllers
    "brimstone": "Controller",
    "viper": "Controller",
    "omen": "Controller",
    "astra": "Controller",
    "harbor": "Controller",
    "clove": "Controller",

    # Sentinels
    "killjoy": "Sentinel",
    "cypher": "Sentinel",
    "sage": "Sentinel",
    "chamber": "Sentinel",
    "deadlock": "Sentinel",
    "vyse": "Sentinel"
}

# Determine role

def determine_role(agent_list):

    if len(agent_list) == 0:
        return "Unknown"

    roles = []

    for agent in agent_list:

        role = role_map.get(
            agent,
            "Unknown"
        )

        if role != "Unknown":
            roles.append(role)

    if len(roles) == 0:
        return "Unknown"

    if len(set(roles)) == 1:
        return roles[0]

    return "Flex"


# Primary agent

df["primary_agent"] = df["agent_list"].apply(
    lambda agents:
        agents[0] if len(agents) > 0 else "Unknown"
)

# Player role for this map

df["player_role"] = df["agent_list"].apply(
    determine_role
)

# Number of agents recorded

df["agent_count"] = df["agent_list"].apply(len)

# Role availability

df["role_available"] = (
    df["player_role"] != "Unknown"
).astype(int)

# Find unmapped agents

unknown_agents = set()

for agents in df["agent_list"]:

    for agent in agents:

        if agent not in role_map:
            unknown_agents.add(agent)

if unknown_agents:

    print("\nAgents without role mapping:")

    for agent in sorted(unknown_agents):
        print("-", agent)

else:

    print("\nAll agents successfully mapped to roles.")


# Role distribution

print("\nROLE DISTRIBUTION")

print(
    df["player_role"]
    .value_counts()
)


# Agent distribution

print("\nTOP AGENTS")

print(
    df["primary_agent"]
    .value_counts()
    .head(20)
)


# Role performance summary

role_summary = (
    df.groupby("player_role")
    .agg(
        players=("player_id", "nunique"),
        maps=("game_id", "nunique"),
        avg_rating=("rating_all", "mean"),
        avg_acs=("acs_all", "mean"),
        avg_adr=("adr_all", "mean"),
        avg_kast=("kast_all", "mean"),
        avg_assists=("assists_all", "mean"),
        avg_kills=("kills_all", "mean"),
        avg_deaths=("deaths_all", "mean")
    )
    .reset_index()
)

print("\nROLE PERFORMANCE SUMMARY")

print(
    role_summary
    .round(2)
    .to_string(index=False)
)


# Save role-enhanced dataset

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nROLE FEATURE ENGINEERING COMPLETE")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nSaved:")
print(OUTPUT_FILE)