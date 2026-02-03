import os
import sys
import math
import json
from datetime import datetime, date

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def parse_date_safe(value: str):
    if pd.isna(value) or value == "":
        return None
    # Try a few common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%b %d, %Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except Exception:
            continue
    # Last resort: pandas parser
    try:
        return pd.to_datetime(value, errors="coerce").date()
    except Exception:
        return None


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {name}")
    return pd.read_csv(path, **kwargs)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop perfect duplicate rows
    df = df.drop_duplicates()

    # Basic NA handling for key numeric columns: fill with medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Standardize some categorical text columns by stripping whitespace
    for c in ["p_id2", "nationality", "work_rate", "position"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    return df


def clean_competitions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    # Strip strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    # Coerce booleans
    if "is_major_national_league" in df.columns:
        df["is_major_national_league"] = df["is_major_national_league"].astype(str).str.lower().map({
            "true": True, "false": False
        })
    # Fill numeric medians
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def clean_players_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    # Trim whitespace
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    # Parse dates
    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    if "contract_expiration_date" in df.columns:
        df["contract_expiration_date"] = pd.to_datetime(df["contract_expiration_date"], errors="coerce")
    # Numeric coercions
    for c in ["height_in_cm", "market_value_in_eur", "highest_market_value_in_eur"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    return df


def clean_player_valuations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Numeric
    for c in ["market_value_in_eur", "current_club_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    # Categorical trim
    if "player_club_domestic_competition_id" in df.columns:
        df["player_club_domestic_competition_id"] = df["player_club_domestic_competition_id"].astype(str).str.strip().replace({"nan": np.nan})
    return df


def clean_player_injuries_impact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    # Parse date strings
    for col in ["Date of Injury", "Date of return"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Standardize rating numbers where possible
    for col in df.columns:
        if "Player_rating" in col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill numeric medians
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    # Trim text
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
    return df


def build_player_slug(first: str, last: str) -> str:
    if pd.isna(first) and pd.isna(last):
        return None
    first_norm = str(first).strip().lower().replace(" ", "") if not pd.isna(first) else ""
    last_norm = str(last).strip().lower().replace(" ", "") if not pd.isna(last) else ""
    slug = f"{first_norm}{last_norm}"
    return slug if slug else None


def engineer_features(main_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    df = main_df.copy()

    # Ensure types
    if "start_year" in df.columns:
        df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")

    # Performance trend features (by player over time)
    sort_cols = ["p_id2", "start_year"] if set(["p_id2", "start_year"]).issubset(df.columns) else None
    if sort_cols:
        df = df.sort_values(sort_cols)
        group = df.groupby("p_id2", dropna=False)

        # Rolling averages over previous 2 seasons where sensible (use transform to preserve index)
        for base_col in [
            "season_minutes_played",
            "season_games_played",
            "season_days_injured",
            "pace",
            "physic",
            "fifa_rating",
        ]:
            if base_col in df.columns:
                df[f"{base_col}_roll2_mean"] = group[base_col].transform(
                    lambda s: s.shift(1).rolling(2, min_periods=1).mean()
                )

        # Simple year-over-year deltas for minutes and games
        for base_col in ["season_minutes_played", "season_games_played"]:
            if base_col in df.columns:
                df[f"{base_col}_yoy_delta"] = group[base_col].diff()

        # Minutes trend: linear trend proxy using expanding mean difference (transform for alignment)
        if "season_minutes_played" in df.columns:
            df["minutes_trend"] = group["season_minutes_played"].transform(
                lambda s: s - s.expanding(min_periods=2).mean()
            )

    # Injury risk metric: combine previous season days injured and rolling mean
    if "season_days_injured_prev_season" in df.columns:
        prev = df["season_days_injured_prev_season"].fillna(0)
    else:
        prev = 0
    roll = df["season_days_injured_roll2_mean"] if "season_days_injured_roll2_mean" in df.columns else 0
    # Normalize components and combine
    def safe_norm(x):
        x = pd.to_numeric(x, errors="coerce").fillna(0)
        denom = (x.max() - x.min()) if (x.max() - x.min()) else 1
        return (x - x.min()) / denom

    prev_n = safe_norm(prev)
    roll_n = safe_norm(roll)
    df["injury_risk_score"] = 0.6 * roll_n + 0.4 * prev_n

    # Contract-related features via fuzzy slug match on first+last name
    # Create a slug on players_df
    players = players_df.copy()
    if {"first_name", "last_name"}.issubset(players.columns):
        players["slug"] = players.apply(lambda r: build_player_slug(r.get("first_name"), r.get("last_name")), axis=1)
    else:
        players["slug"] = None

    players_contract = players[["slug", "contract_expiration_date"]].dropna().copy()
    # Parse dates
    players_contract["contract_expiration_date"] = players_contract["contract_expiration_date"].apply(parse_date_safe)

    # Build slug for dataset names if available
    if "p_id2" in df.columns:
        df["slug"] = df["p_id2"].astype(str).str.strip().str.lower()
    else:
        df["slug"] = None

    merged = df.merge(players_contract, on="slug", how="left")

    # Estimate contract years remaining at start of season (assume July 1 of start_year)
    def years_remaining(row):
        sy = row.get("start_year")
        exp = row.get("contract_expiration_date")
        if pd.isna(sy) or exp is None:
            return np.nan
        try:
            start_dt = date(int(sy), 7, 1)
            delta = (exp - start_dt).days
            return round(delta / 365.25, 3)
        except Exception:
            return np.nan

    merged["contract_years_remaining"] = merged.apply(years_remaining, axis=1)

    # Drop helper columns
    merged = merged.drop(columns=[c for c in ["slug", "contract_expiration_date"] if c in merged.columns])
    return merged


def scale_and_encode(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    df = df.copy()

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in id_cols]

    categorical_cols = [
        c for c in df.columns
        if c not in id_cols and c not in numeric_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", "passthrough", categorical_cols),  # We'll one-hot manually to keep column names
        ],
        remainder="drop",
    )

    # Scale numeric
    scaled_numeric = pd.DataFrame(
        preprocessor.fit_transform(df),
        columns=[*numeric_cols, *categorical_cols],
        index=df.index,
    )

    # One-hot encode categoricals
    if categorical_cols:
        ohe = pd.get_dummies(df[categorical_cols].astype("category"), dummy_na=True)
        scaled_numeric = pd.concat([scaled_numeric.drop(columns=categorical_cols), ohe], axis=1)

    # Restore identifiers at front
    final_df = pd.concat([df[id_cols], scaled_numeric], axis=1)

    # Persist transformers for reproducibility
    joblib.dump({
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "id_columns": id_cols,
    }, os.path.join(MODELS_DIR, "preprocessing_columns.json"))

    return final_df


def process_main_dataset():
    print("[Week2] Loading main dataset.csv ...")
    df = load_csv("dataset.csv")
    players = load_csv("players.csv")

    print("[Week2] Cleaning dataset ...")
    df_clean = clean_dataset(df)

    print("[Week2] Engineering features ...")
    df_feat = engineer_features(df_clean, players)

    # Choose identifiers to keep
    id_cols = []
    for candidate in ["p_id2", "start_year"]:
        if candidate in df_feat.columns:
            id_cols.append(candidate)

    print("[Week2] Scaling numeric and one-hot encoding categoricals ...")
    processed = scale_and_encode(df_feat, id_cols=id_cols)

    print("[Week2] Saving outputs ...")
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "dataset_clean.csv"), index=False)
    df_feat.to_csv(os.path.join(OUTPUT_DIR, "dataset_features.csv"), index=False)
    processed.to_csv(os.path.join(OUTPUT_DIR, "dataset_processed.csv"), index=False)


def process_auxiliary_tables():
    # competitions
    try:
        comp = load_csv("competitions.csv")
        comp_clean = clean_competitions(comp)
        comp_clean.to_csv(os.path.join(OUTPUT_DIR, "competitions_clean.csv"), index=False)
    except Exception as e:
        print(f"[Week2] competitions.csv skipped: {e}")

    # players master
    try:
        players = load_csv("players.csv")
        players_clean = clean_players_table(players)
        players_clean.to_csv(os.path.join(OUTPUT_DIR, "players_clean.csv"), index=False)
    except Exception as e:
        print(f"[Week2] players.csv skipped: {e}")

    # player valuations
    try:
        vals = load_csv("player_valuations.csv")
        vals_clean = clean_player_valuations(vals)
        vals_clean.to_csv(os.path.join(OUTPUT_DIR, "player_valuations_clean.csv"), index=False)
    except Exception as e:
        print(f"[Week2] player_valuations.csv skipped: {e}")

    # injuries impact
    try:
        inj = load_csv("player_injuries_impact.csv")
        inj_clean = clean_player_injuries_impact(inj)
        inj_clean.to_csv(os.path.join(OUTPUT_DIR, "player_injuries_impact_clean.csv"), index=False)
    except Exception as e:
        print(f"[Week2] player_injuries_impact.csv skipped: {e}")


def sentiment_pipeline():
    print("[Week2] Loading Twitter training/validation ...")
    train_path = os.path.join(DATA_DIR, "twitter_training.csv")
    valid_path = os.path.join(DATA_DIR, "twitter_validation.csv")

    def load_twitter(path):
        # Many variations exist; try without header, then with
        try:
            df = pd.read_csv(path, header=None, names=["tweet_id", "entity", "label", "text"], encoding_errors="ignore")
        except Exception:
            df = pd.read_csv(path, encoding_errors="ignore")
        # Ensure required columns
        cols = df.columns.str.lower().tolist()
        mapping = {}
        for c in df.columns:
            lc = c.lower()
            if "id" in lc:
                mapping[c] = "tweet_id"
            elif lc in ("entity", "topic", "category"):
                mapping[c] = "entity"
            elif lc in ("label", "sentiment"):
                mapping[c] = "label"
            elif lc in ("text", "tweet"):
                mapping[c] = "text"
        df = df.rename(columns=mapping)
        # Reduce to necessary
        keep = [c for c in ["tweet_id", "entity", "label", "text"] if c in df.columns]
        df = df[keep]
        return df

    train_df = load_twitter(train_path)
    valid_df = load_twitter(valid_path)

    # Clean rows
    def basic_clean_text(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        # Remove surrounding quotes and stray artifacts
        s = s.str.replace("\u0000", " ", regex=False)
        s = s.str.replace("\r", " ", regex=False)
        s = s.str.replace("\n", " ", regex=False)
        s = s.str.strip()
        return s

    for df in (train_df, valid_df):
        if "text" in df.columns:
            df["text"] = basic_clean_text(df["text"]).str.slice(0, 1000)
        if "label" in df.columns:
            df["label"] = df["label"].astype(str).str.strip()

    # Filter to known labels
    valid_labels = {"positive", "negative", "neutral"}
    for df in (train_df, valid_df):
        if "label" in df.columns:
            df["label_norm"] = df["label"].str.lower()
            df = df[df["label_norm"].isin(valid_labels)]
            df["label"] = df["label_norm"].str.capitalize()
            df.drop(columns=["label_norm"], inplace=True, errors="ignore")

    # Persist cleaned twitter csvs
    cleaned_train = train_df.copy()
    cleaned_valid = valid_df.copy()
    cleaned_train.to_csv(os.path.join(OUTPUT_DIR, "twitter_training_clean.csv"), index=False)
    cleaned_valid.to_csv(os.path.join(OUTPUT_DIR, "twitter_validation_clean.csv"), index=False)

    # Train/Validation split if validation missing labels
    if valid_df.empty or valid_df["label"].isna().all():
        tmp = train_df.dropna(subset=["text", "label"]).copy()
        train_df, valid_df = train_test_split(tmp, test_size=0.2, random_state=42, stratify=tmp["label"])

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"].fillna("Neutral")

    X_valid = valid_df["text"].fillna("")
    y_valid = valid_df["label"].fillna("Neutral")

    print("[Week2] Vectorizing text (TF-IDF) and training Logistic Regression ...")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=50000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=200, n_jobs=None)

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_valid)

    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xva)
    report = classification_report(y_valid, y_pred, digits=3)

    print("[Week2] Writing sentiment report ...")
    with open(os.path.join(OUTPUT_DIR, "sentiment_report.txt"), "w", encoding="utf-8") as f:
        f.write("Preliminary Sentiment Analysis Report\n")
        f.write("Model: TF-IDF + LogisticRegression\n\n")
        f.write(report)

    # Persist artifacts
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "sentiment_logreg.joblib"))


def main():
    ensure_dirs()
    process_main_dataset()
    process_auxiliary_tables()
    sentiment_pipeline()
    # EDA Summary
    try:
        df_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_clean.csv"))
        df_feat = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_features.csv"))
        df_proc = pd.read_csv(os.path.join(OUTPUT_DIR, "dataset_processed.csv"))
        comp_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "competitions_clean.csv"))
        players_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "players_clean.csv"))
        vals_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "player_valuations_clean.csv"))
        injuries_clean = pd.read_csv(os.path.join(OUTPUT_DIR, "player_injuries_impact_clean.csv"))
        twitter_train = pd.read_csv(os.path.join(OUTPUT_DIR, "twitter_training_clean.csv"))
        twitter_valid = pd.read_csv(os.path.join(OUTPUT_DIR, "twitter_validation_clean.csv"))

        lines = []
        def summarize(df: pd.DataFrame, name: str):
            lines.append(f"=== {name} ===")
            lines.append(f"shape: {df.shape[0]} rows, {df.shape[1]} columns")
            # Missingness
            miss = df.isna().mean().sort_values(ascending=False)
            top_miss = miss.head(10)
            lines.append("top_missing_cols (%):")
            for k, v in top_miss.items():
                lines.append(f"  {k}: {round(v*100, 2)}%")
            # Numeric summary
            num = df.select_dtypes(include=[np.number])
            if not num.empty:
                desc = num.describe().T
                lines.append("numeric_summary (first 10 cols):")
                for col in desc.index[:10]:
                    s = desc.loc[col]
                    lines.append(f"  {col}: mean={round(s['mean'],2)}, std={round(s['std'],2)}, min={round(s['min'],2)}, max={round(s['max'],2)}")
            # Categorical cardinality
            cat_cols = [c for c in df.columns if c not in num.columns]
            lines.append("top_categorical_cardinality:")
            for c in cat_cols[:10]:
                lines.append(f"  {c}: {df[c].nunique()} unique")
            lines.append("")

        summarize(df_clean, "dataset_clean")
        summarize(df_feat, "dataset_features")
        summarize(df_proc, "dataset_processed")
        summarize(comp_clean, "competitions_clean")
        summarize(players_clean, "players_clean")
        summarize(vals_clean, "player_valuations_clean")
        summarize(injuries_clean, "player_injuries_impact_clean")
        summarize(twitter_train, "twitter_training_clean")
        summarize(twitter_valid, "twitter_validation_clean")

        with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"EDA generation failed: {e}")
    print("[Week2] Done. Outputs in 'processed/' and models in 'models/'.")


if __name__ == "__main__":
    main()


