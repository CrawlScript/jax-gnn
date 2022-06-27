# coding=utf-8

import warnings
warnings.simplefilter('always', DeprecationWarning)

from jax.config import config
config.update("jax_enable_x64", True)

from . import utils, data, datasets
from .data.graph import Graph

from . import nn