"""Gather csv files and make latex table."""

import pandas as pd
from eval.eval_util import pull_fields

FIELDS = ["env_name", "agent_type", "learning_rate", "seed", "obs_mask"]
VAL_FIELD = "best eval"
BY = ["agent type"]  # , "obs mask"
INDEX = "env name"
MEASURE = "median"


df = pd.read_csv("eval/data/wandb_runs.csv", index_col=0)

SWEEPS = ["olx8u5gy", "bkngzbt9"]
# df = df[df["Sweep"].isin(SWEEPS)]

# COLUMNS PRESENT IN DF ARE OVERWRITTEN!
df = pull_fields(df, FIELDS)
df[FIELDS] = df[FIELDS].fillna("none")
df[FIELDS] = df[FIELDS].apply(lambda r: r.str.replace("_", " ") if r.dtype == "object" else r)
df.columns = df.columns.str.replace("_", " ")


df = df.dropna(subset=[VAL_FIELD])


# Filtering
def mask_fn(row):
    return (
        True
        # & row["agent_type"] == "rflo"
        & (row["obs mask"].lower() == "first half")
        # & (row["obs mask"].lower() == "none")
        & (row["seed"] in [1, 2, 3, 4, 5])
        # & (row["env name"] in ["inverted pendulum", "ant", "halfcheetah", "reacher"])
    )


df = df[df.apply(mask_fn, axis=1)]

# Get the most recent run for each seed
# df = df.sort_values("created at").groupby(BY + [INDEX, "seed"]).tail(1)
df = df.sort_values(VAL_FIELD).groupby(BY + [INDEX, "seed"]).tail(1)


def mean_pm_std(x):
    """Convert mean and std to latex string."""
    return x[MEASURE].map(lambda a: f"{a:.2f}") + "$\pm$" + x["std"].map(lambda a: f"{a:.2f}")


# Make table
df = df.pivot_table(
    index=INDEX, columns=BY, values=VAL_FIELD, aggfunc={VAL_FIELD: [MEASURE, "std", "count"]}, sort=False
)
print(df)
print("")
print("")
df = df.apply(mean_pm_std, axis=1)
df = df.transpose().sort_index(axis=1)
# df = df.swaplevel(0, 1, axis=1)

print(
    df.to_latex(
        escape=False,
        column_format="l" + "c" * (df.shape[-1]),
        multicolumn_format="c",
        float_format="{:.2f}".format,
        # header=headers,
    )
)
