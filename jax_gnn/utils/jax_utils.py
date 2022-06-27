# coding=utf-8
from itertools import chain

import jax.numpy as jnp
import jax

def convert_to_jax(data, dtype=None):
    if data is None:
        return data
    else:
        return jnp.asarray(data, dtype=dtype)

#
# def find_params_by_keywords(params, keywords):
#     if

# https://github.com/google/flax/discussions/1654
def find_params_by_name(params, filter_func):
    from typing import Iterable

    def _is_leaf_fun(x):
        if isinstance(x, Iterable) and jax.tree_util.all_leaves(x.values()):
            return True
        return False

    results = []

    def _finder(x):
        # results = []
        for key, value in x.items():
            if filter_func(key):
                results.append(value)
                # results.append({key: value})
        return None

    jax.tree_map(_finder, params, is_leaf=_is_leaf_fun)

    return results

