"""Validate daily data with pandera, fill short NO2 gaps, drop long ones."""
import pandas as pd
import pandera.pandas as pa

from aq import config

MAX_FILL_DAYS = 3
WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "surface_pressure", "wind_speed_10m", "wind_u", "wind_v"]


def build_schema(allow_missing_no2: bool) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, unique=True),            # no duplicate dates
            "no2": pa.Column(float, pa.Check.in_range(0, 500), nullable=allow_missing_no2),
            "temperature_2m": pa.Column(float, pa.Check.in_range(-30, 50), nullable=True),
            "wind_speed_10m": pa.Column(float, pa.Check.ge(0), nullable=True),
            "precipitation": pa.Column(float, pa.Check.ge(0), nullable=True),
        },
        coerce=True,
    )


def gap_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df["no2"].isna()
    groups = (missing != missing.shift()).cumsum()
    return (df[missing].groupby(groups[missing])
            .agg(start=("date", "min"), end=("date", "max"), days=("date", "size"))
            .reset_index(drop=True))


def main():
    df = pd.read_parquet(config.DATA_PROCESSED / "daily_raw.parquet")
    build_schema(allow_missing_no2=True).validate(df)               # range + duplicates

    report = gap_report(df)
    print(f"Missing NO2 days: {int(df['no2'].isna().sum())} in {len(report)} gaps")
    if len(report):
        print(report.sort_values("days", ascending=False).head(10).to_string(index=False))

    # fill only gaps of <= MAX_FILL_DAYS; longer gaps stay missing
    missing = df["no2"].isna()
    run_len = missing.groupby((~missing).cumsum()).transform("sum")
    filled = df["no2"].interpolate(limit_area="inside")
    fillable = missing & (run_len <= MAX_FILL_DAYS)
    df.loc[fillable, "no2"] = filled[fillable]
    df[WEATHER_COLS] = df[WEATHER_COLS].interpolate(limit=MAX_FILL_DAYS, limit_area="inside")

    before = len(df)
    df = df.dropna(subset=["no2"] + WEATHER_COLS).reset_index(drop=True)
    print(f"Filled {int(fillable.sum())} short-gap days, dropped {before - len(df)} days")

    build_schema(allow_missing_no2=False).validate(df)              # strict pass
    df.to_parquet(config.DATA_PROCESSED / "daily.parquet", index=False)
    print(f"Saved {len(df)} clean days: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()