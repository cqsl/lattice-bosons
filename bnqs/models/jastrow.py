from typing import (Any, Tuple)
import flax.linen as nn

import jax.numpy as jnp
from jax.nn.initializers import normal
import numpy as np

from netket.utils.types import NNInitFunc
from netket.graph import AbstractGraph

PRNGKey = Any
Shape = Tuple[int, ...]
DType = Any
Array = Any


def _min_image_distance(x, extent):
    dis = -x[np.newaxis, :, :] + x[:, np.newaxis, :]
    dis = dis - extent * np.rint(dis / extent)
    return dis


class SQJastrow(nn.Module):
    """Two-body Jastrow in second quantization."""
    graph: AbstractGraph
    """Graph corresponding to the lattice."""
    kernel_init: NNInitFunc = normal(0.5)
    """Initializer for the Jastrow weights."""
    param_dtype: DType = jnp.float64
    """The dtype of all parameters."""
    @nn.compact
    def __call__(self, x):
        d_max = np.floor_divide(self.graph.extent, 2).sum()
        sd = _min_image_distance(self.graph.positions, self.graph.extent)
        d = np.linalg.norm(sd, ord=1, axis=-1).astype(int)
        kernel = self.param('Jastrow', self.kernel_init, (d_max+1,), self.param_dtype)
        return jnp.einsum('...i,ij,...j->...', x, kernel[(d,)], x)
