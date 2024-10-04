
import numpy as np

from functools import reduce
import aim
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

repo1 = aim.Repo('aim_repos/brax_new/')
repo2 = aim.Repo('aim_repos/baselines/')
 
# _extra_query = "run.hparams.env_params.env_name == 'brax-ant' and run.hparams.env_params.obs_mask == 'even' and metric.name == 'eval/rewards'"

envs = ['brax-ant', 'brax-halfcheetah', 'brax-hopper']


def make_plot(repo, fig, env_name, extra_query="", name="", color="blue"):
    """Plot learning curve of given environment."""
    query = f"(run.hparams.env_params.env_name == '{env_name}' or run.hparams.env_name == '{env_name.split('-')[1]}')" + \
            " and (run.hparams.env_params.obs_mask == 'even' or run.hparams.obs_mask == 'even')" + \
            " and (metric.name == 'eval/rewards' or metric.name =='eval/episode_reward')"
    if extra_query:
        query += f" and ({extra_query})"

    # columns = {
    #     "filter_type":  "Filter Type",
    #     # "filtered_size":  "Neurons",
    #     # "best_eval_reward": "Best eval reward",
    # }

    # last_column = "best_eval_reward"

    # keys = list(columns.keys())+[last_column]
    # labels = list(columns.values())+["Best eval reward"]

    runs = repo.query_metrics(query)

    def index_nested_dict(d, keys):
        """Index a nested dictionary using a dot-separated string."""
        return [reduce(dict.get, k.split("."), d) for k in keys]

    # Get the data from the runs
    # metadata = [index_nested_dict(r.run["hparams"], columns.keys()) + [r.run.get(last_column, -np.inf)]
    #             for r in runs.iter_runs()]
    best = [-np.inf]
    x = [0]
    for run in runs.iter_runs():
        for metric in run:
            # print(metric.run.name)
            # metric.values()
            values = metric.values.values_numpy()
            if max(values) > max(best):
                best = values
                x = metric.epochs.values_list()
    col = envs.index(env_name)
    fig.add_trace(
        go.Scatter(y=best, x=sorted(x), name=name if name else env_name, marker=dict(color=color), showlegend=col < 1),
        row=1, col=col+1
    )

    # df = pd.DataFrame(metadata, columns=keys)
    # filtered = df.dropna()
    # filtered.head()


fig = make_subplots(rows=1, cols=len(envs), subplot_titles=envs)
for i, env in enumerate(envs):
    make_plot(repo1, fig, env, name="RTRRL", color="red")
    make_plot(repo2, fig, env, name="PPO", color="blue")
    fig.update_xaxes(title_text="Epochs", row=1, col=i+1)
    fig.update_yaxes(title_text="Episode Return", row=1, col=i+1)

fig.show()
