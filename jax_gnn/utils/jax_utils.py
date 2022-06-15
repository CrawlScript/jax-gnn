# coding=utf-8
import jax.numpy as jnp


def convert_to_jax(data, dtype=None):
    if data is None:
        return data
    else:
        return jnp.asarray(data, dtype=dtype)
