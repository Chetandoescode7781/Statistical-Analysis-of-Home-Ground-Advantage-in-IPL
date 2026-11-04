import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Locate the input CSVs robustly, whether this script is run from the project
# root or from inside the ipl_data folder. Generated outputs go to ipl_data
# when that folder exists; otherwise they go beside this script.
SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name.lower() == "ipl_data":
    BASE_DIR = SCRIPT_DIR
    PROJECT_DIR = SCRIPT_DIR.parent
elif (SCRIPT_DIR / "ipl_data").is_dir():
    BASE_DIR = SCRIPT_DIR / "ipl_data"
    PROJECT_DIR = SCRIPT_DIR
else:
    BASE_DIR = SCRIPT_DIR
    PROJECT_DIR = SCRIPT_DIR.parent

BALLS_FILENAME = "ipl_ball_by_ball_2008_2026.csv"
RESULTS_FILENAME = "IPL_Matches_Data_2008_2026.csv"

# Try common locations: the output folder, script folder, project folder,
# and the parent folder. This avoids assuming where test.py was launched from.
def find_input_file(filename):
    candidates = [
        BASE_DIR / filename,
        SCRIPT_DIR / filename,
        PROJECT_DIR / filename,
        SCRIPT_DIR / "ipl_data" / filename,
        SCRIPT_DIR.parent / filename,
        SCRIPT_DIR.parent / "ipl_data" / filename,
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.is_file():
            return candidate
        seen.add(candidate)
    searched = "\n".join(f"  - {p}" for p in dict.fromkeys(p.resolve() for p in candidates))
    raise FileNotFoundError(
        f"Could not find required input file '{filename}'. Searched these locations:\n{searched}\n\n"
        f"Put the CSV in the project folder or its ipl_data folder, then rerun the script."
    )

CSV_FILE = find_input_file(BALLS_FILENAME)
RESULTS_FILE = find_input_file(RESULTS_FILENAME)
print("Ball-by-ball input:", CSV_FILE)
print("Match-results input:", RESULTS_FILE)
print("Output folder:", BASE_DIR)
SELECTED_TEAMS = [
    "Royal Challengers Bengaluru",
    "Mumbai Indians",
    "Chennai Super Kings",
    "Kolkata Knight Riders"
]
# Analyze all IPL seasons covered by the supplied datasets.
FIRST_SEASON = 2008
LAST_SEASON = 2026
TARGET_SEASONS = range(FIRST_SEASON, LAST_SEASON + 1)

# Chronological holdout for the match-winner comparison.
# 2008 has no prior-season data in this dataset, so its strength features are unavailable.
TRAIN_SEASONS = range(2009, 2023)  # 2009-2022
TEST_SEASONS = range(2023, LAST_SEASON + 1)  # 2023-2026

BATTING_WEIGHTS = {
    "Runs": 0.30,
    "StrikeRate": 0.25,
    "BoundaryPct": 0.15,
    "RunsPerWicket": 0.20,
    "DotBallPct": 0.10
}
BOWLING_WEIGHTS = {
    "Economy": 0.30,
    "WicketRate": 0.25,
    "DotBallPct": 0.20,
    "BoundaryConcededPct": 0.15,
    "RunsConceded": 0.10
}
HIGHER_IS_BETTER = {
    "Runs": True, "StrikeRate": True, "BoundaryPct": True,
    "RunsPerWicket": True, "DotBallPct": False,
    "Economy": False, "WicketRate": True,
    "BoundaryConcededPct": False, "RunsConceded": False
}

HOME_VENUES = {
    "Royal Challengers Bengaluru": [
        "M Chinnaswamy Stadium", "M. Chinnaswamy Stadium",
        "M Chinnaswamy Stadium, Bengaluru", "M Chinnaswamy Stadium, Bangalore"
    ],
    "Mumbai Indians": ["Wankhede Stadium", "Wankhede Stadium, Mumbai"],
    "Chennai Super Kings": [
        "MA Chidambaram Stadium", "M. A. Chidambaram Stadium",
        "MA Chidambaram Stadium, Chepauk",
        "MA Chidambaram Stadium, Chepauk, Chennai",
        "M. A. Chidambaram Stadium, Chepauk", "Chepauk"
    ],
    "Kolkata Knight Riders": ["Eden Gardens", "Eden Gardens, Kolkata"],
    "Delhi Capitals": ["Arun Jaitley Stadium", "Feroz Shah Kotla"],
    "Punjab Kings": [
        "Punjab Cricket Association IS Bindra Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Punjab Cricket Association Stadium",
        "Punjab Cricket Association Stadium, Mohali",
        "Maharaja Yadavindra Singh International Cricket Stadium",
        "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"
    ],
    "Rajasthan Royals": ["Sawai Mansingh Stadium", "Sawai Mansingh Stadium, Jaipur"],
    "Sunrisers Hyderabad": [
        "Rajiv Gandhi International Stadium",
        "Rajiv Gandhi International Stadium, Uppal",
        "Rajiv Gandhi Intl. Cricket Stadium",
        "Rajiv Gandhi International Cricket Stadium, Hyderabad"
    ],
    "Gujarat Titans": [
        "Narendra Modi Stadium", "Narendra Modi Stadium (Motera)",
        "Sardar Patel Stadium, Motera"
    ],
    "Lucknow Super Giants": [
        "BRSABV Ekana Cricket Stadium", "BRSABV Ekana Cricket Stadium, Lucknow",
        "Ekana Cricket Stadium"
    ]
}

def normalize_name(value):
    if pd.isna(value):
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).split())

TEAM_ALIASES = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "chennai super kings": "Chennai Super Kings",
    "mumbai indians": "Mumbai Indians",
    "kolkata knight riders": "Kolkata Knight Riders",
    "rajasthan royals": "Rajasthan Royals",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "gujarat titans": "Gujarat Titans",
    "lucknow super giants": "Lucknow Super Giants",
    "deccan chargers": "Deccan Chargers"
}

def normalize_team(value):
    if pd.isna(value):
        return ""
    name = str(value).strip()
    return TEAM_ALIASES.get(normalize_name(name), name)

def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

def minmax_normalize(series, higher_is_better):
    low, high = series.min(), series.max()
    if pd.isna(low) or pd.isna(high):
        return pd.Series(np.nan, index=series.index)
    if high == low:
        return pd.Series(0.5, index=series.index)
    return (series - low) / (high - low) if higher_is_better else (high - series) / (high - low)

VENUE_TO_TEAMS = {}
for team, venues in HOME_VENUES.items():
    for venue in venues:
        VENUE_TO_TEAMS.setdefault(normalize_name(venue), set()).add(team)

def classify_home(team, venue):
    venue_key = normalize_name(venue)
    if not venue_key:
        return np.nan
    matches = set()
    for known_venue, teams in VENUE_TO_TEAMS.items():
        if venue_key == known_venue or venue_key.startswith(known_venue + " "):
            matches.update(teams)
    if not matches:
        return np.nan
    return int(team in matches)

df = pd.read_csv(CSV_FILE, low_memory=False)
required = [
    "match_id", "season", "venue", "innings", "batting_team", "bowling_team",
    "total_runs", "batter_runs", "extra_runs", "is_legal_ball", "is_four",
    "is_six", "is_dot", "is_wicket", "extra_type"
]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["season"] = pd.to_numeric(df["season"], errors="coerce")
df = df.dropna(subset=["season"]).copy()
df["season"] = df["season"].astype(int)
df["batting_team"] = df["batting_team"].apply(normalize_team)
df["bowling_team"] = df["bowling_team"].apply(normalize_team)

if "is_super_over" in df.columns:
    df = df[~to_bool(df["is_super_over"])].copy()

for column in ["total_runs", "batter_runs", "extra_runs"]:
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
for column in ["is_legal_ball", "is_four", "is_six", "is_dot", "is_wicket"]:
    df[column] = to_bool(df[column])

extra_type = df["extra_type"].fillna("").astype(str).str.lower()
uncharged = extra_type.str.contains(r"bye|leg.?bye", regex=True)
df["BowlerRuns"] = (df["total_runs"] - df["extra_runs"].where(uncharged, 0)).clip(lower=0)

if "bowler_wicket" in df.columns:
    df["BowlerWicket"] = to_bool(df["bowler_wicket"])
elif "dismissal_kind" in df.columns:
    non_bowler_dismissals = {
        "run out", "retired hurt", "retired out",
        "obstructing the field", "retired not out"
    }
    dismissal = df["dismissal_kind"].fillna("").astype(str).str.strip().str.lower()
    df["BowlerWicket"] = df["is_wicket"] & ~dismissal.isin(non_bowler_dismissals)
else:
    df["BowlerWicket"] = df["is_wicket"]

for new, old in {
    "LegalBall": "is_legal_ball", "Four": "is_four", "Six": "is_six",
    "DotBall": "is_dot", "Wicket": "is_wicket", "BowlerWicket": "BowlerWicket"
}.items():
    df[new] = df[old].astype(int)

def calculate_strengths(history):
    keys = ["match_id", "season", "innings"]
    bat = history.groupby(keys + ["batting_team", "bowling_team"], as_index=False).agg(
        Score=("total_runs", "sum"), BatterRuns=("batter_runs", "sum"),
        LegalBalls=("LegalBall", "sum"), Fours=("Four", "sum"),
        Sixes=("Six", "sum"), DotBalls=("DotBall", "sum"),
        WicketsLost=("Wicket", "sum")
    )
    bt = bat.groupby("batting_team").agg(
        TotalRuns=("Score", "sum"), Innings=("Score", "size"),
        BatterRuns=("BatterRuns", "sum"), LegalBalls=("LegalBalls", "sum"),
        Fours=("Fours", "sum"), Sixes=("Sixes", "sum"),
        DotBalls=("DotBalls", "sum"), WicketsLost=("WicketsLost", "sum")
    )
    bt["Runs"] = bt["TotalRuns"] / bt["Innings"]
    bt["StrikeRate"] = np.where(bt["LegalBalls"] > 0, bt["BatterRuns"] / bt["LegalBalls"] * 100, np.nan)
    bt["BoundaryPct"] = np.where(bt["LegalBalls"] > 0, (bt["Fours"] + bt["Sixes"]) / bt["LegalBalls"] * 100, np.nan)
    bt["RunsPerWicket"] = np.where(bt["WicketsLost"] > 0, bt["TotalRuns"] / bt["WicketsLost"], np.nan)
    bt["DotBallPct"] = np.where(bt["LegalBalls"] > 0, bt["DotBalls"] / bt["LegalBalls"] * 100, np.nan)

    bowl = history.groupby(keys + ["bowling_team", "batting_team"], as_index=False).agg(
        RunsConceded=("BowlerRuns", "sum"), LegalBalls=("LegalBall", "sum"),
        Wickets=("BowlerWicket", "sum"), FoursConceded=("Four", "sum"),
        SixesConceded=("Six", "sum"), DotBalls=("DotBall", "sum")
    )
    bw = bowl.groupby("bowling_team").agg(
        TotalRunsConceded=("RunsConceded", "sum"), Innings=("RunsConceded", "size"),
        LegalBalls=("LegalBalls", "sum"), Wickets=("Wickets", "sum"),
        FoursConceded=("FoursConceded", "sum"), SixesConceded=("SixesConceded", "sum"),
        DotBalls=("DotBalls", "sum")
    )
    bw["Economy"] = np.where(bw["LegalBalls"] > 0, bw["TotalRunsConceded"] / bw["LegalBalls"] * 6, np.nan)
    bw["WicketRate"] = bw["Wickets"] / bw["Innings"]
    bw["DotBallPct"] = np.where(bw["LegalBalls"] > 0, bw["DotBalls"] / bw["LegalBalls"] * 100, np.nan)
    bw["BoundaryConcededPct"] = np.where(
        bw["LegalBalls"] > 0,
        (bw["FoursConceded"] + bw["SixesConceded"]) / bw["LegalBalls"] * 100,
        np.nan
    )
    bw["RunsConceded"] = bw["TotalRunsConceded"] / bw["Innings"]

    for metric in BATTING_WEIGHTS:
        bt[metric + "_N"] = minmax_normalize(bt[metric], HIGHER_IS_BETTER[metric])
    for metric in BOWLING_WEIGHTS:
        bw[metric + "_N"] = minmax_normalize(bw[metric], HIGHER_IS_BETTER[metric])
    bt["BattingStrength"] = sum(w * bt[m + "_N"] for m, w in BATTING_WEIGHTS.items())
    bw["BowlingStrength"] = sum(w * bw[m + "_N"] for m, w in BOWLING_WEIGHTS.items())
    return bt["BattingStrength"].to_dict(), bw["BowlingStrength"].to_dict()

innings_data = df.groupby(
    ["match_id", "season", "venue", "innings", "batting_team", "bowling_team"],
    as_index=False
).agg(Score=("total_runs", "sum"))

season_average = innings_data.groupby("season")["Score"].mean()
innings_data["SeasonAvgScore"] = innings_data["season"].map(season_average)
innings_data["RelativeScore"] = innings_data["Score"] / innings_data["SeasonAvgScore"].replace(0, np.nan)

strength_rows = []
for season in TARGET_SEASONS:
    history = df[(df["season"] >= season - 3) & (df["season"] < season)]
    if history.empty:
        print(f"WARNING: No historical data for season {season}.")
        continue
    batting_strength, bowling_strength = calculate_strengths(history)
    for team in set(batting_strength) | set(bowling_strength):
        strength_rows.append({
            "season": season,
            "team": team,
            "BattingStrength": batting_strength.get(team, np.nan),
            "BowlingStrength": bowling_strength.get(team, np.nan)
        })

strength_df = pd.DataFrame(
    strength_rows,
    columns=["season", "team", "BattingStrength", "BowlingStrength"]
).drop_duplicates(["season", "team"])

print("\nStrength history note: strengths use only the preceding three seasons.")
print("For 2008, prior-season data is unavailable in this dataset, so 2008 strength features are missing.")
print("The regression datasets retain those rows, but regression fitting drops rows with missing predictors.")

# Restrict the batting regression to head-to-head innings between the four selected teams.
analysis = innings_data[
    innings_data["batting_team"].isin(SELECTED_TEAMS)
    & innings_data["bowling_team"].isin(SELECTED_TEAMS)
    & innings_data["season"].isin(TARGET_SEASONS)
].copy()
analysis["Home"] = analysis.apply(lambda r: classify_home(r["batting_team"], r["venue"]), axis=1)
analysis = analysis.merge(
    strength_df[["season", "team", "BattingStrength"]].rename(columns={
        "team": "batting_team", "BattingStrength": "OwnBattingStrength"
    }),
    on=["season", "batting_team"], how="left"
)
analysis = analysis.merge(
    strength_df[["season", "team", "BowlingStrength"]].rename(columns={
        "team": "bowling_team", "BowlingStrength": "OppBowlingStrength"
    }),
    on=["season", "bowling_team"], how="left"
)
analysis.to_csv(BASE_DIR / "ipl_regression_dataset.csv", index=False)

print("\n========== DATASET CHECK ==========")
print("Selected-team innings:", len(analysis))
print("\nHome classification:")
print(analysis["Home"].value_counts(dropna=False))
print("\nRows per season:")
print(analysis.groupby("season").size())
print("\nUnknown or neutral venues:")
print(analysis.loc[analysis["Home"].isna(), "venue"].value_counts().head(20))

def fit_model(data, outcome, predictors, title, output_file):
    usable = data.dropna(subset=[outcome, "Home", *predictors, "match_id"]).copy()
    if len(usable) < 10:
        raise ValueError(f"Too few usable observations for {title}. Check venue mappings and historical strengths.")
    if usable["Home"].nunique() < 2:
        raise ValueError(f"{title} needs both Home=0 and Home=1 observations.")
    X = sm.add_constant(usable[["Home", *predictors]])
    model = sm.OLS(usable[outcome], X).fit(
        cov_type="cluster", cov_kwds={"groups": usable["match_id"]}
    )
    print(f"\n========== {title} DATA ==========")
    print("Usable observations:", len(usable))
    print("Unique matches:", usable["match_id"].nunique())
    print("Home observations:", int((usable["Home"] == 1).sum()))
    print("Away observations:", int((usable["Home"] == 0).sum()))
    print("\n========== COEFFICIENTS ==========")
    print(model.params)
    print("\n========== P-VALUES ==========")
    print(model.pvalues)
    print("\n========== 95% CONFIDENCE INTERVALS ==========")
    print(model.conf_int())
    print("\n========== FULL REGRESSION SUMMARY ==========")
    print(model.summary())
    print(f"\nHome coefficient: {model.params['Home']:.5f}")
    print(f"Estimated home effect: {model.params['Home'] * 100:.2f}% of the season-average innings score, holding the other predictors constant.")
    print(f"\nSaved dataset to {output_file}")
    return model, usable

batting_model, batting_model_data = fit_model(
    analysis, "RelativeScore",
    ["OwnBattingStrength", "OppBowlingStrength"],
    "BATTING REGRESSION", "ipl_regression_dataset.csv"
)

# Restrict the bowling regression to the same four-team head-to-head scope.
bowling_analysis = innings_data[
    innings_data["season"].isin(TARGET_SEASONS)
    & innings_data["bowling_team"].isin(SELECTED_TEAMS)
    & innings_data["batting_team"].isin(SELECTED_TEAMS)
].copy()
bowling_analysis["RelativeRunsConceded"] = (
    bowling_analysis["Score"] / bowling_analysis["SeasonAvgScore"].replace(0, np.nan)
)
bowling_analysis["Home"] = bowling_analysis.apply(
    lambda r: classify_home(r["bowling_team"], r["venue"]), axis=1
)
bowling_analysis = bowling_analysis.merge(
    strength_df[["season", "team", "BattingStrength"]].rename(columns={
        "team": "batting_team", "BattingStrength": "OppBattingStrength"
    }),
    on=["season", "batting_team"], how="left"
)
bowling_analysis = bowling_analysis.merge(
    strength_df[["season", "team", "BowlingStrength"]].rename(columns={
        "team": "bowling_team", "BowlingStrength": "OwnBowlingStrength"
    }),
    on=["season", "bowling_team"], how="left"
)
bowling_analysis.to_csv(BASE_DIR / "ipl_bowling_regression_dataset.csv", index=False)

bowling_model, bowling_model_data = fit_model(
    bowling_analysis, "RelativeRunsConceded",
    ["OwnBowlingStrength", "OppBattingStrength"],
    "BOWLING REGRESSION", "ipl_bowling_regression_dataset.csv"
)
print("\nFor the bowling model, a negative Home coefficient means the bowling team conceded fewer runs at home, relative to the seasonal average.")


# ============================================================
# MATCH WINNER COMPARATOR
# Compare logistic regression with and without home advantage.
# Restrict to matches where both teams are in SELECTED_TEAMS.
# Dataset span: 2008-2026. Train: 2009-2022. Test: 2023-2026.
# ============================================================

print("\n========== MATCH WINNER COMPARATOR ==========")
print("Results file:", RESULTS_FILE)

if not RESULTS_FILE.exists():
    nearby = sorted(p.name for p in BASE_DIR.glob("*.csv"))
    raise FileNotFoundError(
        f"Could not find results CSV:\n{RESULTS_FILE}\n"
        f"CSV files in the script folder: {nearby}"
    )

results = pd.read_csv(RESULTS_FILE, low_memory=False)
# Normalize headers so Season, Team_1, and similar names are handled consistently.
results.columns = [
    re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
    for col in results.columns
]

# Support common header variants in the match-results file.
def find_column(candidates, required=True):
    for candidate in candidates:
        if candidate in results.columns:
            return candidate
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}. "
            f"Columns in results CSV: {results.columns.tolist()}"
        )
    return None

season_col = find_column(["season", "year"])
team1_col = find_column(["team_1", "team1", "team_1_name", "first_team"])
team2_col = find_column(["team_2", "team2", "team_2_name", "second_team"])
winner_col = find_column(["winner", "winning_team", "match_winner"])
venue_col = find_column(["venue", "stadium", "ground"], required=False)
date_col = find_column(["date", "match_date", "start_date"], required=False)

# Fallback parser for season labels. Prefer the actual match date below because
# this CSV uses labels such as '2007/08', '2009/10', and '2020/21', while the
# ball-by-ball file identifies seasons by the calendar year in which matches
# were played. The date correctly maps these to 2008, 2010, and 2020.
def parse_match_season(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"(19|20)\d{2}", str(value).strip())
    return int(match.group(0)) if match else np.nan

match_data = pd.DataFrame()
season_from_label = results[season_col].apply(parse_match_season)
if date_col:
    parsed_dates = pd.to_datetime(results[date_col], errors="coerce", dayfirst=True)
    season_from_date = parsed_dates.dt.year.astype("float")
    # Actual match date takes precedence; use season label only when date is missing/unparseable.
    match_data["season"] = season_from_date.where(season_from_date.notna(), season_from_label)
else:
    print("WARNING: No match date column found; falling back to parsing season labels.")
    match_data["season"] = season_from_label
match_data["season"] = pd.to_numeric(match_data["season"], errors="coerce")
match_data["team1"] = results[team1_col].apply(normalize_team)
match_data["team2"] = results[team2_col].apply(normalize_team)
match_data["winner"] = results[winner_col].apply(normalize_team)
match_data["venue"] = results[venue_col].fillna("").astype(str) if venue_col else ""

# Keep only completed matches where the winner is one of the two participating teams.
valid_winner = (
    match_data["winner"].eq(match_data["team1"])
    | match_data["winner"].eq(match_data["team2"])
)
# Keep only matches where BOTH teams belong to our four-team research scope.
match_data = match_data[
    match_data["season"].isin(TARGET_SEASONS)
    & match_data["team1"].isin(SELECTED_TEAMS)
    & match_data["team2"].isin(SELECTED_TEAMS)
    & valid_winner
].copy()
match_data["Team1Won"] = (match_data["winner"] == match_data["team1"]).astype(int)

# Add team-strength estimates calculated only from the previous three seasons.
match_data = match_data.merge(
    strength_df[["season", "team", "BattingStrength", "BowlingStrength"]].rename(
        columns={
            "team": "team1",
            "BattingStrength": "Team1BattingStrength",
            "BowlingStrength": "Team1BowlingStrength",
        }
    ),
    on=["season", "team1"], how="left"
)
match_data = match_data.merge(
    strength_df[["season", "team", "BattingStrength", "BowlingStrength"]].rename(
        columns={
            "team": "team2",
            "BattingStrength": "Team2BattingStrength",
            "BowlingStrength": "Team2BowlingStrength",
        }
    ),
    on=["season", "team2"], how="left"
)
match_data["BattingStrengthDiff"] = (
    match_data["Team1BattingStrength"] - match_data["Team2BattingStrength"]
)
match_data["BowlingStrengthDiff"] = (
    match_data["Team1BowlingStrength"] - match_data["Team2BowlingStrength"]
)

# Positive HomeAdvantage means team1 is at its home venue; negative means team2 is.
# Unknown/neutral venues are assigned 0 for this comparator.
def calculate_home_advantage(row):
    home1 = classify_home(row["team1"], row["venue"])
    home2 = classify_home(row["team2"], row["venue"])
    if pd.isna(home1) or pd.isna(home2):
        return 0
    if home1 == 1 and home2 == 0:
        return 1
    if home2 == 1 and home1 == 0:
        return -1
    return 0

match_data["HomeAdvantage"] = match_data.apply(calculate_home_advantage, axis=1)

comparator_file = BASE_DIR / "ipl_match_prediction_dataset.csv"
match_data.to_csv(comparator_file, index=False)
print(f"Completed head-to-head matches between selected teams ({FIRST_SEASON}-{LAST_SEASON}):", len(match_data))
print("Rows by season:")
print(match_data.groupby("season").size())
print("Home advantage values (+1 team1 home, -1 team2 home, 0 neutral/unknown):")
print(match_data["HomeAdvantage"].value_counts(dropna=False).sort_index())
print("Saved comparator dataset:", comparator_file)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix

base_features = ["BattingStrengthDiff", "BowlingStrengthDiff"]
features_with_home = base_features + ["HomeAdvantage"]
train = match_data[match_data["season"].isin(TRAIN_SEASONS)].copy()
test = match_data[match_data["season"].isin(TEST_SEASONS)].copy()

# Ensure both models are evaluated on exactly the same rows.
required_features = features_with_home + ["Team1Won"]
train_before = len(train)
test_before = len(test)
train = train.dropna(subset=required_features)
test = test.dropna(subset=required_features)
print(f"Training rows excluded for missing strength/outcome features: {train_before - len(train)}")
print(f"Testing rows excluded for missing strength/outcome features: {test_before - len(test)}")

if train.empty or test.empty:
    raise ValueError(
        "The match winner comparator has no usable train or test rows after joining team strengths. "
        f"Train rows={len(train)}, test rows={len(test)}. Check season labels and team names."
    )
if train["Team1Won"].nunique() < 2:
    raise ValueError("Training data must contain wins by both team1 and team2.")
if test["Team1Won"].nunique() < 2:
    print("WARNING: Test set contains only one outcome class; ROC-AUC will be unavailable.")

print(f"\nTraining matches ({min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}):", len(train))
print(f"Testing matches ({min(TEST_SEASONS)}-{max(TEST_SEASONS)}):", len(test))

models = {
    "WITHOUT HOME ADVANTAGE": base_features,
    "WITH HOME ADVANTAGE": features_with_home,
}
summary_rows = []
prediction_output = test[["season", "team1", "team2", "winner", "Team1Won", "venue", "HomeAdvantage"]].copy()

for model_name, features in models.items():
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    )
    model.fit(train[features], train["Team1Won"])
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(test["Team1Won"], predictions)
    brier = brier_score_loss(test["Team1Won"], probabilities)
    auc = (
        roc_auc_score(test["Team1Won"], probabilities)
        if test["Team1Won"].nunique() == 2 else np.nan
    )
    print(f"\n========== {model_name} ==========")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"ROC-AUC:  {auc:.4f}" if not pd.isna(auc) else "ROC-AUC: unavailable (test set has one class)")
    print(f"Brier score (lower is better): {brier:.4f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(test["Team1Won"], predictions, labels=[0, 1]))

    key = "with_home" if "HOME ADVANTAGE" in model_name and "WITHOUT" not in model_name else "without_home"
    prediction_output[f"{key}_prob_team1_wins"] = probabilities
    prediction_output[f"{key}_prediction_team1_wins"] = predictions
    summary_rows.append({
        "Model": model_name,
        "TestMatches": len(test),
        "Accuracy": accuracy,
        "ROC_AUC": auc,
        "BrierScore": brier,
    })

summary = pd.DataFrame(summary_rows)
print("\n========== MODEL COMPARISON ==========")
print(summary.to_string(index=False))
summary.to_csv(BASE_DIR / "ipl_match_model_comparison.csv", index=False)
prediction_file = BASE_DIR / f"ipl_match_predictions_{min(TEST_SEASONS)}_{max(TEST_SEASONS)}.csv"
prediction_output.to_csv(prediction_file, index=False)
print("\nSaved model comparison:", BASE_DIR / "ipl_match_model_comparison.csv")
print("Saved test predictions:", prediction_file)
