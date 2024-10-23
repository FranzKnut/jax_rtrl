"""Plot data that was downloaded from wandb."""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval.eval_util import pull_fields

# Load data
df = pd.read_csv("eval/data/wandb_runs_history64ctrnn.csv", index_col=[0, 1])

df = pull_fields(df)

excluded_envs = [
    # "MinesweeperEasy",
    "MultiArmedBanditEasy",
    "MultiArmedBanditHard",
    "StatelessCartPoleHard",
    "NoisyStatelessCartPoleHard",
    "MinesweeperHard",
    # "RepeatPreviousEasy",
    "RepeatPreviousHard",
    "RepeatFirstHard",
    "MemoryChain-bsuite",
    "DeepSea-bsuite",
]
df = df[~df["env_name"].isin(excluded_envs)]
# df = df[~df["env_name"].str.contains("Hard")]

excluded_models = [
    ""
]
df = df[~df["Model"].isin(excluded_models)]
df = df[~(df["trace"]=="dutch")]

# Look for Multiindex as string tuples
# \( and \): Match the literal parentheses surrounding the tuple.
# ('([^']*)'|\"([^\"]*)\"): Captures a string enclosed in either single or double quotes.

#     ([^']*): Captures any characters inside single quotes.
#     ([^\"]*): Captures any characters inside double quotes.

# ,\s*: Matches a comma followed by optional whitespace.
# (-?\d+): Captures an integer.
pattern = r"\(('([^']*)'|\"([^\"]*)\"),\s*(\d+)\)"

# Convert index to a consistent format (all tuples)
new_index = [eval(x) if re.match(pattern, x) else (x, "") for x in df.columns]
# Create MultiIndex
df.columns = pd.MultiIndex.from_tuples(new_index)

# data["Task Length"] = data.config.apply(lambda x: eval(x)["env_init_args"]["size"])
fig = plt.figure(figsize=(10, 4))

BY = ["meta", "fa"]
max_x = None
agg_method = "mean"
shade_agg_method = "var"

num_envs = len(df["env_name_full"].unique())
downsample = 10
rows = 2
n = 1
for key, group in df.groupby("env_name_full"):
    plt.subplot(rows, np.ceil(num_envs / rows).astype(int), n)
    plt.title(key)
    plt.xlabel("steps")
    plt.ylabel("reward")
    n += 1
    print(key)
    for line, subgroup in group.groupby(BY):
        x = subgroup["_step"].loc[subgroup.index[0]][::downsample]
        y = subgroup["mean_reward"].loc[:, ::downsample].agg([agg_method, shade_agg_method]).T
        label = ("CTRNN RFLO " + " ".join(line).strip()) if n == num_envs else "_nolegend_"
        plt.plot(x, y[agg_method], label=label)  # 
        if shade_agg_method:
            plt.fill_between(x, y[agg_method] - y[shade_agg_method], y[agg_method] + y[shade_agg_method], alpha=0.3, label="_nolegend_")

    if max_x:
        plt.xlim(0, max_x)
        
# Dummy plot to make legend
fig.legend(loc=(.78,.15), title="Architecture")




fig.suptitle("")
# fig.tight_layout(w_pad=1, pad=1)
fig.tight_layout()
os.makedirs("eval/plots/", exist_ok=True)
plt.savefig("eval/plots/lineplot.pdf")
plt.show()
