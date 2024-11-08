"""Timing of RTRL and RFLO for different hidden sizes."""
import argparse
import os
import timeit

import jax.numpy as jnp
import jax.random as jrand
import pandas as pd
import seaborn as sns
from supervised_online import make_model
from training_utils import train_rnn_online

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
parser.add_argument("--outfile", default="artifacts/timing.csv")
args = parser.parse_args()

if not os.path.exists(args.outfile) or args.force:
    key = jrand.PRNGKey(0)
    key, key_train = jrand.split(key)

    x = jnp.linspace(0, 5 * jnp.pi, 100)[:, None]
    y = jnp.sin(x) + 2

    rows = ["rtrl", "rflo"]
    cols = [32, 64, 128, 256]
    runtimes = pd.DataFrame(index=rows, columns=cols)

    for plast in rows:
        for size in cols:
            model, params, h0 = make_model(x[0], key, kwargs={"hidden_size": size, "plasticity": plast})

            def loss(p, __x, __y, carry=None):
                # MSE loss
                carry, y_hat = model.apply(p, __x, carry)
                return jnp.sum((y_hat - __y) ** 2), carry

            t = timeit.timeit(lambda: train_rnn_online(loss, params, (x, y), key_train, h0, num_steps=100), number=3)
            runtimes.loc[plast, size] = t

    runtimes.to_csv(args.outfile)
else:
    runtimes = pd.read_csv(args.outfile, index_col=0, header=0)

print(runtimes)
runtimes = runtimes.unstack().reset_index()
runtimes.columns = ["size", "plasticity", "time"]

plot = sns.barplot(x=runtimes["size"], y=runtimes["time"], hue=runtimes["plasticity"])
plot.set_yscale('log')
plot.figure.savefig(os.path.join(os.path.dirname(__file__), "..", "plots", "timing.png"))
