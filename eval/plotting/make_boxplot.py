"""Plot data that was downloaded from wandb."""

from functools import reduce
import os
import pandas as pd
import matplotlib.pyplot as plt

FIELDS = ["env_name", "agent_type", "learning_rate", "seed", "obs_mask"]
VAL_FIELD = "best_eval"
BY = ["agent_type", "obs_mask"]
SPLIT_PLOTS_BY = "env_name"

df = pd.read_csv("eval/data/wandb_runs.csv", index_col=0)
# data = data[data["Sweep"].isin(sweeps)]
# df = df[df["env_name_full"].isin(table_env_names)]


def gen_dict_extract(var, key):
    if hasattr(var, "items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(v, key):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(d, key):
                        yield result


def deep_get(dictionary, keys, default=None):
    generator = gen_dict_extract(dictionary, keys)
    try:
        return next(generator)
    except StopIteration:
        return default


def make_pull_fields(names=[]):
    def _pull_fields(cfg):
        """Pull relevant fields from the config field."""
        cfg = eval(cfg)

        return pd.Series({n: deep_get(cfg, n) for n in names})

    return _pull_fields


# COLUMNS PRESENT IN DF ARE OVERWRITTEN!
df = df.assign(**df.config.apply(make_pull_fields(FIELDS)))

df[BY] = df[BY].fillna("none")


# Filtering
def mask_fn(row):
    return (
        True
        # & row["agent_type"] == "rflo"
        # & (row["seed"] in [1, 2, 3, 4, 5])
    )


df = df[df.apply(mask_fn, axis=1)]

all_by = BY + [SPLIT_PLOTS_BY, "seed"]
df = df.dropna(subset=VAL_FIELD)
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
