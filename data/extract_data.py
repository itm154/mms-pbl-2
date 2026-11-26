import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse(dir: Path, bin_hours: float = 1.0) -> pd.DataFrame:
    if not dir.exists():
        raise FileNotFoundError(f"Directory not found: {dir}")

    records = []
    for thread in dir.iterdir():
        if not thread.is_dir():
            continue
        for subfolder in ["source-tweets", "reactions"]:
            folder = thread / subfolder
            for f in folder.glob("*.json"):
                try:
                    with open(f, "r") as fp:
                        data = json.load(fp)
                    timestamp = datetime.strptime(
                        data["created_at"], "%a %b %d %H:%M:%S %z %Y"
                    )
                    count = int(data.get("retweet_count", 0)) + 1
                    records.append(
                        {"timestamp": timestamp, "count": count, "thread": thread.name}
                    )
                except Exception:
                    continue

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    t0 = df["timestamp"].iloc[0]
    df["hours_from_start"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0

    total_hours = df["hours_from_start"].max()
    n_bins = int(np.ceil(total_hours / bin_hours)) + 1

    df["time_bin"] = (df["hours_from_start"] / bin_hours).astype(int)
    aggregated = (
        df.groupby("time_bin")
        .agg(
            active_spreaders=("count", "sum"),
            raw_tweets=("count", "count"),
        )
        .reset_index()
    )

    full_index = pd.DataFrame({"time_bin": np.arange(0, n_bins)})
    result = pd.merge(full_index, aggregated, on="time_bin", how="left").fillna(0)
    result["active_spreaders"] = result["active_spreaders"].astype(int)
    result["raw_tweets"] = result["raw_tweets"].astype(int)
    result.rename(columns={"time_bin": "time_step"}, inplace=True)

    return result


def main():
    dirin = Path("data/charliehebdo")
    dirout = Path("data")

    df_charlie = parse(dirin, bin_hours=4.0)
    df_charlie = df_charlie[df_charlie["time_step"] <= 60]
    charlie_csv = dirout / "charliehebdo.csv"
    df_charlie.to_csv(charlie_csv, index=False)
    print("Saved successfully")


if __name__ == "__main__":
    main()
