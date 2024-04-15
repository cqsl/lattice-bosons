from netket.graph import AbstractGraph
from netket.sampler.rules.base import MetropolisRule

import jax
import jax.numpy as jnp
from netket.utils import struct

import numpy as np
import math
from typing import Optional, Sequence, Union
from itertools import islice, repeat, chain

from jax._src import core
from jax._src.typing import Array, ArrayLike
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from jax._src.random import RealArray, Shape, KeyArray, randint, uniform


def batch_choice(
        key: KeyArray,
        a: Union[int, ArrayLike],
        p: Optional[RealArray] = None,
        shape: Shape = (1,),
        axis: int = -1,
) -> Array:
    """Generates a random sample from a given array.

    .. warning::
    If ``p`` has fewer non-zero elements than the requested number of samples,
    as specified in ``shape``, and ``replace=False``, the output of this
    function is ill-defined. Please make sure to use appropriate inputs.

    Args:
    key: a PRNG key used as the random key.
    a : array or int. If an ndarray, a random sample is generated from
      its elements. If an int, the random sample is generated as if a were
      arange(a).
    shape : tuple of ints, optional. Output shape.  If the given shape is,
      e.g., ``(m, n)``, then ``m * n`` samples are drawn.  Default is (),
      in which case a single value is returned.
    replace : boolean.  Whether the sample is with or without replacement.
      default is True.
    p : 1-D array-like, The probabilities associated with each entry in a.
      If not given the sample assumes a uniform distribution over all
      entries in a.
    axis: int, optional. The axis along which the selection is performed.
      The default, 0, selects by row.

    Returns:
    An array of shape `shape` containing samples from `a`.
    """
    # key, _ = _check_prng_key(key)
    if not isinstance(shape, Sequence):
        raise TypeError("shape argument of jax.random.choice must be a sequence, "
                        f"got {shape}")
    check_arraylike("choice", a)
    arr = jnp.asarray(a)
    if arr.ndim == 0:
        n_inputs = core.concrete_or_error(int, a, "The error occurred in jax.random.choice()")
    else:
        # axis = canonicalize_axis(axis, arr.ndim)
        n_inputs = arr.shape[axis]
    n_draws = math.prod(shape)
    if n_draws == 0:
        return jnp.zeros(shape, dtype=arr.dtype)
    if n_inputs <= 0:
        raise ValueError("a must be greater than 0 unless no samples are taken")

    if p is None:
        ind = randint(key, shape, 0, n_inputs)
        result = ind if arr.ndim == 0 else jnp.take(arr, ind, axis)
    else:
        assert p.shape[:-1] == shape
        p_arr, = promote_dtypes_inexact(p)
        if p_arr.shape[-1] != n_inputs:
            raise ValueError("p must be None or match the shape of a")
        p_cuml = jnp.cumsum(p_arr, axis=-1)
        r = p_cuml[:, -1] * (1 - uniform(key, shape, dtype=p_cuml.dtype))
        ind = jax.vmap(jnp.searchsorted)(p_cuml, r).astype(int)
        result = ind if arr.ndim == 0 else jnp.take(arr, ind, axis)
    return result.reshape(shape)


def pad(x):
    zeros = repeat(-1)
    n = max(map(len, x))
    return [list(islice(chain(row, zeros), n)) for row in x]


@struct.dataclass
class LocalRule(MetropolisRule):
    r"""
    A transition rule acting on a particle by moving it to a neighboring site.
    """
    adjacency_list: Array

    @classmethod
    def from_graph(cls, graph: AbstractGraph):
        adjacency_list = np.array(pad(graph.adjacency_list()))
        return cls(adjacency_list)

    def transition(rule, sampler, machine, parameters, state, key, x):
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise TypeError(
                "LocalRule does not work with complex " "basis elements."
            )
        n_chains = x.shape[0]
        hilb = sampler.hilbert
        k1, k2 = jax.random.split(key, num=2)
        i = batch_choice(k1, a=np.arange(hilb.size), p=x, shape=(n_chains,))
        n_i = jax.random.randint(k2, minval=0, maxval=rule.adjacency_list.shape[-1], shape=(n_chains,))

        def update(x, i, n_i, adjacency_list):
            x_i = x[i]
            j = adjacency_list[i, n_i]
            x = x.at[i].add(-1)
            x = x.at[j].add(1)
            xp_j = x[j]
            return x, x_i, xp_j

        xp, x_i, xp_j = jax.vmap(update, (0, 0, 0, None))(x, i, n_i, rule.adjacency_list)
        log_prob_corr = jnp.log(xp_j) - jnp.log(x_i)

        return xp, log_prob_corr

    def __repr__(self):
        return "LocalRule"
