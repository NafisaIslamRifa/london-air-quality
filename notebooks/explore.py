# %%
import matplotlib.pyplot as plt
import pandas as pd
from aq import config

df = pd.read_parquet(config.DATA_PROCESSED / "daily.parquet").set_index("date")

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
df["no2"].plot(ax=ax[0], alpha=0.35, label="daily")
df["no2"].rolling(30, min_periods=15).mean().plot(ax=ax[0], lw=2, label="30-day mean")
ax[0].set(title="NO₂ at MY1 over time", ylabel="µg/m³"); ax[0].legend()

by_day = df.groupby(df.index.dayofweek)["no2"].mean()
by_day.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_day.plot.bar(ax=ax[1], rot=0); ax[1].set(title="Weekly pattern", ylabel="mean µg/m³")

ax[2].scatter(df["wind_speed_10m"], df["no2"], s=6, alpha=0.4)
ax[2].set(title="NO₂ vs wind speed", xlabel="wind speed (km/h)", ylabel="µg/m³")

plt.tight_layout()
(config.ROOT / "docs").mkdir(exist_ok=True)
plt.savefig(config.ROOT / "docs" / "no2_story.png", dpi=150)
plt.show()