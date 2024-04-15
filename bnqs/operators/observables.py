from functools import partial
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class
import numpy as np
import netket as nk
from netket.operator import DiscreteJaxOperator
from netket.hilbert import DiscreteHilbert
from netket.utils.types import DType, Array, PyTree
from netket.graph import AbstractGraph
from flax import struct


@jax.jit
def get_conn_padded(edges, loops, x):
    n_sites = x.shape[-1]
    i = edges[:, 0]
    j = edges[:, 1]
    n_i = x[:, i]
    n_j = x[:, j]

    x_prime0 = x[:, None]
    mels0 = x[..., loops].sum(axis=-1, keepdims=True)

    mask = (n_j > 0)
    mels = mask * (jnp.sqrt(n_j) * jnp.sqrt(n_i + 1))
    x_prime = x[:, None] * mask[..., None]
    add_at = jax.vmap(lambda x, idx, addend, : x.at[:, idx].add(addend), (-2, 0, None), -2)
    x_prime = add_at(x_prime, j, -1)
    x_prime = add_at(x_prime, i, +1)

    mels = jnp.concatenate([mels0, mels], axis=-1)
    x_prime = jnp.concatenate([x_prime0, x_prime], axis=-2)

    return x_prime, mels / n_sites**2


@register_pytree_node_class
class CondensateFraction_(DiscreteJaxOperator):
    def __init__(self, hilbert: DiscreteHilbert, edges: Array, loops: Array, dtype: DType = float):
        super().__init__(hilbert)
        self.edges = edges
        self.loops = loops
        self._dtype = dtype

    def tree_flatten(self):
        array_data = (self.edges, self.loops)
        struct_data = {'hilbert': self.hilbert, 'dtype': self.dtype}
        return array_data, struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        edges, loops = array_data
        hilbert = struct_data['hilbert']
        dtype = struct_data['dtype']
        return cls(hilbert, edges, loops, dtype)

    @property
    def max_conn_size(self) -> int:
        return self.edges.size + self.loops.size > 0

    @property
    def dtype(self):
        # deduct it from the data
        # alternatively define it via struct.field(pytree_node=False)
        return self._dtype

    def get_conn_padded(self, x):
        # compute xp and mels
        return get_conn_padded(self.edges, self.loops, x)


def CondensateFraction(hilbert: DiscreteHilbert, edges: Optional[Array] = None, dtype: DType = float):
    if edges is None:
        idcs = np.arange(hilbert.size)
        i, j = np.meshgrid(idcs, idcs)
        edges = np.column_stack((i.ravel(), j.ravel()))
    loops = edges[edges[:, 0] == edges[:, 1], 0]
    edges = edges[edges[:, 0] != edges[:, 1]]
    return CondensateFraction_(hilbert, edges, loops, dtype)
