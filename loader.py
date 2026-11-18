"""Small CSV ingestion helpers; raw data remains the user's responsibility."""
from pathlib import Path
import pandas as pd


def load_matches_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    if (df[["FTHG", "FTAG"]] < 0).any().any():
        raise ValueError("Goal counts cannot be negative")
    return df.sort_values("Date").reset_index(drop=True)
