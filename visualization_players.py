# viz_safe.py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------
# CONFIG - set your CSV path here
# -----------------------
file_path = r"C:\Users\samar\OneDrive\Documents\Infosys_internship\Player_Features_2022_23.csv"

# -----------------------
# Helpers
# -----------------------
def normalize_cols(cols):
    """Normalize column names to simple tokens"""
    norm = []
    for c in cols:
        c0 = str(c).strip().lower()
        # keep digits and letters and underscores
        c0 = re.sub(r"[^0-9a-z_]+", "_", c0)
        c0 = re.sub(r"_+", "_", c0).strip("_")
        norm.append(c0)
    return norm

def find_first(df_cols, candidates):
    """Return first matching column from candidates (candidates are substrings/aliases)"""
    for cand in candidates:
        for col in df_cols:
            if cand == col:
                return col
    return None

def find_by_contains(df_cols, substrings):
    for s in substrings:
        for col in df_cols:
            if s in col:
                return col
    return None

def compute_per90(df, raw_col, minutes_col, minutes_is_90s):
    """Compute per90 value robustly"""
    if raw_col not in df.columns:
        return None
    if minutes_col not in df.columns:
        return None
    if minutes_is_90s:
        denom = df[minutes_col].replace(0, np.nan)
        return df[raw_col] / denom
    else:
        # minutes column (actual minutes) -> per90 = raw * 90 / minutes
        denom = df[minutes_col].replace(0, np.nan)
        return df[raw_col] * 90.0 / denom

# -----------------------
# Load and normalize columns
# -----------------------
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV not found at {file_path} — update file_path variable.")

df = pd.read_csv(file_path, low_memory=False)
orig_cols = df.columns.tolist()
norm_cols = normalize_cols(orig_cols)

# keep mapping from normalized -> original
col_map = dict(zip(norm_cols, orig_cols))
# rename dataframe columns to normalized names for easier processing
df.rename(columns=col_map, inplace=True)

print("Loaded file:", file_path)
print("Columns found (sample):", list(df.columns)[:30])

# -----------------------
# Identify minutes / 90s column
# -----------------------
# prefer exact names
minutes_candidates = ["90s", "90", "minutes", "mins", "minutes_played", "minutes_played"]
minutes_col = find_first(df.columns, minutes_candidates)
minutes_is_90s = False

if minutes_col is None:
    # fallback: look for column that contains '90' or 'min'
    minutes_col = find_by_contains(df.columns, ["90", "mins", "minutes"])
    if minutes_col is None:
        print("⚠️ Could not find a '90s' or 'minutes' column automatically. Per-90 features will be skipped.")
    else:
        minutes_is_90s = "90" in minutes_col  # crude
else:
    minutes_is_90s = ("90" in minutes_col)

print("Detected minutes column:", minutes_col, " (treated as n90s?)", minutes_is_90s)

# -----------------------
# Column alias lists (normalized names)
# -----------------------
aliases = {
    "goals": ["gls", "goals", "g"],
    "assists": ["ast", "assists"],
    "shots": ["sh", "shots", "shots_total"],
    "shots_on": ["sot", "shots_on", "shots_on_target", "shots_on_target_total"],
    "passes": ["passes", "passes_attempted", "passes_total"],
    "passes_completed": ["passes_completed", "passes_completed_total"],
    "tackles": ["tkl", "tackles"],
    "interceptions": ["int", "interceptions"],
    "blocks": ["blocks"],
    "touches": ["touches"],
    "pressures": ["press", "pressures"],
}

# -----------------------
# Create per90 and composite features if missing
# -----------------------
created = []
missing = []

def get_alias_col(key):
    return find_by_contains(df.columns, aliases.get(key, []))

# compute per90 for each alias if not present
per90_mappings = {
    "goals_per90": ("goals",),
    "assists_per90": ("assists",),
    "shots_per90": ("shots",),
    "shots_on_target_per90": ("shots_on",),
    "passes_attempted_per90": ("passes",),
    "passes_completed_per90": ("passes_completed",),
    "tackles_per90": ("tackles",),
    "interceptions_per90": ("interceptions",),
    "blocks_per90": ("blocks",),
    "pressures_per90": ("pressures",),
    "touches_per90": ("touches",),
}

for new_col, (raw_key,) in per90_mappings.items():
    if new_col in df.columns:
        continue
    raw_col = get_alias_col(raw_key)
    if raw_col and minutes_col:
        df[new_col] = compute_per90(df, raw_col, minutes_col, minutes_is_90s)
        created.append(new_col)
    else:
        missing.append(new_col)

# composite features
# goal_contrib_per90
if "goal_contrib_per90" not in df.columns:
    gcol = get_alias_col("goals")
    acol = get_alias_col("assists")
    if gcol and acol and minutes_col:
        df["goal_contrib_per90"] = compute_per90(df, gcol, minutes_col, minutes_is_90s) + compute_per90(df, acol, minutes_col, minutes_is_90s)
        created.append("goal_contrib_per90")
    else:
        missing.append("goal_contrib_per90")

# shot_accuracy = shots_on / shots
if "shot_accuracy" not in df.columns:
    sc = get_alias_col("shots_on")
    s = get_alias_col("shots")
    if sc and s:
        df["shot_accuracy"] = df[sc].replace(0, np.nan) / df[s].replace(0, np.nan)
        created.append("shot_accuracy")
    else:
        missing.append("shot_accuracy")

# pass_accuracy = passes_completed / passes
if "pass_accuracy" not in df.columns:
    pc = get_alias_col("passes_completed")
    p = get_alias_col("passes")
    if pc and p:
        df["pass_accuracy"] = df[pc].replace(0, np.nan) / df[p].replace(0, np.nan)
        created.append("pass_accuracy")
    else:
        missing.append("pass_accuracy")

# defensive_index = (tkl + int + blocks) per90
if "defensive_index" not in df.columns:
    tcol = get_alias_col("tackles")
    icol = get_alias_col("interceptions")
    bcol = get_alias_col("blocks")
    if minutes_col and (tcol or icol or bcol):
        # sum raw then convert to per90
        sum_raw_name = "_def_raw_sum"
        df[sum_raw_name] = 0
        if tcol: df[sum_raw_name] = df[sum_raw_name] + df[tcol].fillna(0)
        if icol: df[sum_raw_name] = df[sum_raw_name] + df[icol].fillna(0)
        if bcol: df[sum_raw_name] = df[sum_raw_name] + df[bcol].fillna(0)
        df["defensive_index"] = compute_per90(df, sum_raw_name, minutes_col, minutes_is_90s)
        # drop helper
        df.drop(columns=[sum_raw_name], inplace=True)
        created.append("defensive_index")
    else:
        missing.append("defensive_index")

print("\nFeature creation summary:")
print(" Created:", created)
print(" Missing (couldn't create):", missing)

# -----------------------
# Ensure 'position' label exists (bulletproof version)
# -----------------------
if "position" not in df.columns:
    pos_dummy_cols = [c for c in df.columns if c.startswith("pos_")]

    if pos_dummy_cols:
        clean_cols = []
        for col in pos_dummy_cols:
            try:
                # Force everything into string, then into numeric (0/1 if possible)
                df[col] = pd.to_numeric(df[col].astype(str), errors="coerce").fillna(0).astype(int)
                clean_cols.append(col)
            except Exception as e:
                print(f"⚠️ Skipping {col} during numeric conversion: {e}")

        if clean_cols:
            df["position"] = df[clean_cols].idxmax(axis=1).str.replace("pos_", "")
            print("Derived 'position' from pos_* dummy columns.")
        else:
            print("⚠️ No usable pos_* dummy columns after cleaning.")

    elif "pos" in df.columns:
        def map_pos_raw(p):
            if not isinstance(p, str):
                return "Other"
            p = p.upper()
            if p.startswith("F") or p in ["RW","LW","CF","ST","FW"]: return "Attacker"
            if p.startswith("M"): return "Midfielder"
            if p.startswith("D"): return "Defender"
            if p in ["GK","G"]: return "Goalkeeper"
            return "Other"
        df["position"] = df["pos"].apply(map_pos_raw)
        print("Derived 'position' from 'pos' column.")

    else:
        print("⚠️ Could not derive 'position' (no pos or pos_* columns). Some plots will be skipped.")


# -----------------------
# Now plotting — guard each plot with existence checks
# -----------------------
sns.set(style="whitegrid")
plots_done = []

def safe_plot_hist(col, title, bins=30, color=None):
    if col in df.columns and df[col].dropna().shape[0] > 0:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col].dropna(), bins=bins, kde=True, color=color)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plots_done.append(title)
    else:
        print(f"Skipping plot (missing): {title}")

def safe_boxplot(xcol, ycol, title):
    if xcol in df.columns and ycol in df.columns:
        plt.figure(figsize=(8,5))
        sns.boxplot(x=xcol, y=ycol, data=df)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plots_done.append(title)
    else:
        print(f"Skipping plot (missing): {title}")

def safe_scatter(xcol, ycol, hue, title):
    if xcol in df.columns and ycol in df.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=xcol, y=ycol, hue=(hue if hue in df.columns else None), data=df, alpha=0.6)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plots_done.append(title)
    else:
        print(f"Skipping plot (missing): {title}")

# 1: Goals per90 distribution
safe_plot_hist("goals_per90", "Distribution of Goals per 90")

# 2: Assists per90
safe_plot_hist("assists_per90", "Distribution of Assists per 90", color="green")

# 3: Defensive index
safe_plot_hist("defensive_index", "Distribution of Defensive Index", color="red")

# 4: Goals per90 by position (boxplot)
safe_boxplot("position", "goals_per90", "Goals per 90 by Position")

# 5: Pass accuracy by position
safe_boxplot("position", "pass_accuracy", "Pass Accuracy by Position")

# 6: Age vs Goal Contributions scatter
if "age" in df.columns:
    safe_scatter("age", "goal_contrib_per90", "position", "Age vs Goal Contributions per 90")
else:
    print("Skipping Age vs Goal Contributions — 'age' missing")

# 7: Offensive correlation heatmap
off_keys = [k for k in ["goals_per90","assists_per90","goal_contrib_per90","shots_per90","shots_on_target_per90"] if k in df.columns]
if off_keys:
    plt.figure(figsize=(10,6))
    sns.heatmap(df[off_keys].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation (offensive key features)")
    plt.tight_layout()
    plt.show()
    plots_done.append("Offensive Correlation Heatmap")
else:
    print("Skipping offensive correlation heatmap — not enough offensive features")

# 8: Defensive correlation heatmap
def_keys = [k for k in ["tackles_per90","interceptions_per90","blocks_per90","defensive_index"] if k in df.columns]
if len(def_keys) >= 2:
    plt.figure(figsize=(6,5))
    sns.heatmap(df[def_keys].corr(), annot=True, cmap="Blues")
    plt.title("Correlation of Defensive Features")
    plt.tight_layout()
    plt.show()
    plots_done.append("Defensive Correlation Heatmap")
else:
    print("Skipping defensive heatmap — not enough defensive features")

# 9: Shots vs Goals scatter
safe_scatter("shots_per90", "goals_per90", "position", "Shots per 90 vs Goals per 90 (by Position)")

# 10: Top N players by goal contribution
if "player" in df.columns and "goal_contrib_per90" in df.columns:
    top_n = 20
    top_players = df.nlargest(top_n, "goal_contrib_per90").dropna(subset=["goal_contrib_per90"])
    plt.figure(figsize=(10,8))
    sns.barplot(x="goal_contrib_per90", y="player", data=top_players, palette="viridis")
    plt.title(f"Top {top_n} Players by Goal Contributions per 90")
    plt.tight_layout()
    plt.show()
    plots_done.append("Top players by goal contrib")
else:
    print("Skipping top players plot — missing 'player' or 'goal_contrib_per90'")

# 11: Average performance by position
pos_metrics = [c for c in ["goals_per90","assists_per90","passes_completed_per90","defensive_index","shots_per90"] if c in df.columns]
if "position" in df.columns and pos_metrics:
    avg_stats = df.groupby("position")[pos_metrics].mean().reset_index()
    plt.figure(figsize=(10,6))
    avg_stats.set_index("position").T.plot(kind="bar")
    plt.title("Average Performance by Position")
    plt.ylabel("Per 90 Values")
    plt.tight_layout()
    plt.show()
    plots_done.append("Average Performance by Position")
else:
    print("Skipping average-by-position plot — missing position or metrics")

print("\nPlots generated:", len(plots_done))
for p in plots_done[:10]:
    print(" -", p)

print("\nDone. If a plot was skipped, check the 'Missing (couldn't create)' list above and ensure your raw columns (goals, assists, shots, passes, tackles, minutes/90s) are present.")
