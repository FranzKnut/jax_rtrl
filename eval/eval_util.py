import pandas as pd


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


def pull_fields(df, names=[]):
    def _pull_fields(cfg):
        """Pull relevant fields from the config field."""
        cfg = eval(cfg)

        return pd.Series({n: deep_get(cfg, n) for n in names})

    return df.assign(**df.config.apply(_pull_fields))
