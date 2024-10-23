"""Plot data that was downloaded from wandb."""

import os
from eval.eval_util import pull_fields
import pandas as pd
import matplotlib.pyplot as plt

FIELDS = ["env_name", "agent_type", "learning_rate", "seed", "obs_mask"]
VAL_FIELD = "best_eval"
BY = ["agent_type", "obs_mask"]
SPLIT_PLOTS_BY = "env_name"

df = pd.read_csv("eval/data/wandb_runs.csv", index_col=0)
df = df.dropna(subset=VAL_FIELD)

SWEEPS = ["olx8u5gy", "bkngzbt9"]
# df = df[df["Sweep"].isin(SWEEPS)]
# COLUMNS PRESENT IN DF ARE OVERWRITTEN!
df = pull_fields(df, FIELDS)

df[BY] = df[BY].fillna("none")


# Filtering
def mask_fn(row):
    return (
        True
        # & row["agent_type"] == "rflo"
        & (row["seed"] in [1, 2, 3, 4, 5])
    )


df = df[df.apply(mask_fn, axis=1)]

all_by = BY + [SPLIT_PLOTS_BY, "seed"]
df = df.sort_values("created_at").groupby(all_by).tail(1)

print(df.groupby(BY)[VAL_FIELD].count())
fig = plt.figure(figsize=(15, 4))
# Make a box for mean reward grouped by plasticity and memory length
print(df[all_by].nunique())
axes = df.groupby(SPLIT_PLOTS_BY).boxplot(
    column=VAL_FIELD,
    by=BY,
    layout=(1, -1),
    ax=fig.gca(),
    sharey=False,
    rot=90,
)
fig.suptitle("", y=0.1)
fig.tight_layout(w_pad=2)
for ax in axes:
    ax.set_xlabel("")

os.makedirs("eval/plots", exist_ok=True)
plt.savefig("eval/plots/boxplot.pdf", transparent=True)
plt.show()
