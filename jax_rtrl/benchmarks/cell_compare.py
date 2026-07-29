"""Small runtime benchmark for all registered RNN cell types.

The benchmark compares a short forward pass and a forward-plus-gradient pass
for every registered cell key in :mod:`jax_rtrl.models.cells`.

Some model/config combinations are currently not compatible with the shared
benchmark shape contract. Those cases are reported as skipped instead of
failing the whole run.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jax_rtrl.models.cells import CELL_TYPES
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig, scan_rnn


@dataclass
class BenchmarkRow:
    model_name: str
    status: str
    forward_ms: float | None = None
    forward_grad_ms: float | None = None
    note: str = ""


def _tree_scalar(tree):
    leaves = [jnp.ravel(jnp.real(jnp.asarray(leaf))) for leaf in jax.tree.leaves(tree)]
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    flat = jnp.concatenate(leaves) if len(leaves) > 1 else leaves[0]
    return jnp.sum(flat**2)


def _block_until_ready(value):
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def _time_callable(fn, params, repeats: int):
    fn(params)
    jax.block_until_ready(fn(params))

    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(params)
        _block_until_ready(result)
        timings.append((time.perf_counter() - start) * 1000.0)
    return sum(timings) / len(timings), result


def _make_model(model_name: str, hidden_size: int, num_modules: int, num_blocks: int):
    config = RNNEnsembleConfig(
        model_name=model_name,
        hidden_size=hidden_size,
        num_modules=num_modules,
        num_blocks=num_blocks,
    )
    return RNNEnsemble(config, out_size=None)


def benchmark_model(
    model_name: str,
    *,
    hidden_size: int,
    input_size: int,
    seq_len: int,
    num_modules: int,
    num_blocks: int,
    repeats: int,
):
    rng = jrandom.PRNGKey(0)
    x_seq = jrandom.normal(jrandom.PRNGKey(1), (seq_len, input_size))

    model = _make_model(model_name, hidden_size, num_modules, num_blocks)
    params = model.init(rng, None, x_seq[0])
    carry = model.apply(params, rng, x_seq[0].shape, method=model.initialize_carry)

    def loss_fn(p):
        _, outputs = scan_rnn(model, p, x_seq, init_carry=carry)
        return _tree_scalar(outputs)

    forward_fn = jax.jit(loss_fn)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    forward_ms, _ = _time_callable(forward_fn, params, repeats)
    grad_ms, _ = _time_callable(grad_fn, params, repeats)
    return BenchmarkRow(
        model_name=model_name,
        status="ok",
        forward_ms=forward_ms,
        forward_grad_ms=grad_ms,
    )


def run_benchmark(args):
    rows: list[BenchmarkRow] = []
    model_names = list(CELL_TYPES.keys()) if args.models is None else args.models

    for model_name in model_names:
        try:
            row = benchmark_model(
                model_name,
                hidden_size=args.hidden_size,
                input_size=args.input_size,
                seq_len=args.seq_len,
                num_modules=args.num_modules,
                num_blocks=args.num_blocks,
                repeats=args.repeats,
            )
        except Exception as exc:  # pragma: no cover - benchmark should keep going
            row = BenchmarkRow(
                model_name=model_name,
                status="skipped",
                note=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)

    if args.outfile:
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        with open(args.outfile, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model_name", "status", "forward_ms", "forward_grad_ms", "note"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "model_name": row.model_name,
                        "status": row.status,
                        "forward_ms": row.forward_ms,
                        "forward_grad_ms": row.forward_grad_ms,
                        "note": row.note,
                    }
                )

    print("model_name,status,forward_ms,forward_grad_ms,note")
    for row in rows:
        print(
            f"{row.model_name},{row.status},"
            f"{'' if row.forward_ms is None else f'{row.forward_ms:.3f}'},"
            f"{'' if row.forward_grad_ms is None else f'{row.forward_grad_ms:.3f}'},"
            f"{row.note}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--num-modules", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outfile", default="artifacts/cell_compare.csv")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model keys to benchmark.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
