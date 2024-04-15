from functools import partial
from typing import (Any, Sequence, Callable, Union, Optional, Tuple)
import flax.linen as nn

import jax.numpy as jnp
from jax.nn.initializers import zeros, normal
import numpy as np

from netket.utils.types import NNInitFunc
from netket.nn.masked_linear import default_kernel_init
from netket.graph import AbstractGraph

PRNGKey = Any
Shape = Tuple[int, ...]
DType = Any
Array = Any


def _min_image_distance(x, extent):
    dis = -x[np.newaxis, :, :] + x[:, np.newaxis, :]
    dis = dis - extent * np.rint(dis / extent)
    return dis


class ConvNetJastrow(nn.Module):
    """ConvNet-based neural backflow Jastrow."""
    graph: AbstractGraph
    """Graph corresponding to the lattice."""
    features: Sequence[int]
    """Number of channels at each layer."""
    kernel_size: Union[Sequence[int]]
    """Size of the convolutional filters."""
    padding: str = 'CIRCULAR'
    """Type of padding. Must be set to 'CIRCULAR' for OBC and 'SAME' for OBC."""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    param_dtype: DType = float
    """The dtype of all parameters."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""
    activation: Callable = nn.gelu
    """Hidden activation function."""
    output_activation: Optional[Callable] = None
    """Output activation function."""

    backflow_kernel_init: NNInitFunc = normal(1e-3)
    """Initializer for the backflow mixing weights."""
    jastrow_kernel_init: NNInitFunc = normal(5e-1)
    """Initializer for the Jastrow weights."""

    density: float = 1.0
    """Filling factor."""

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        depth = len(self.features)
        kernel_sizes = depth * (self.kernel_size,)  # if isinstance(self.kernel_size, int) else self.kernel_size
        assert len(kernel_sizes) == depth
        cnn = partial(nn.Conv, padding=self.padding, use_bias=self.use_bias, param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init, bias_init=self.bias_init)

        a = nn.Dense(1, use_bias=False, kernel_init=self.backflow_kernel_init, param_dtype=self.param_dtype)

        d_max = np.floor_divide(self.graph.extent, 2).sum()
        sd = _min_image_distance(self.graph.positions, self.graph.extent)
        d = np.linalg.norm(sd, ord=1, axis=-1).astype(int)
        jastrow = self.param('Jastrow', self.jastrow_kernel_init, (d_max + 1,), self.param_dtype)

        x0 = x.copy()
        x = x.reshape(*shape[:-1], *self.graph.extent, 1)
        for i, (feature, kernel_size) in enumerate(zip(self.features, kernel_sizes)):
            x = self.activation(x) if i else x / self.density - 1.0
            x = cnn(name=f"CNN_{i}", features=feature, kernel_size=kernel_size)(x)
        if self.output_activation:
            x = self.output_activation(x)
        x_tilde = x0 + a(x).reshape(shape)

        return jnp.einsum('...i,ij,...j->...', x_tilde, jastrow[(d,)], x_tilde)


class ResNetJastrow(nn.Module):
    """ResNet-based neural backflow Jastrow."""
    graph: AbstractGraph
    """Graph corresponding to the lattice."""
    features: Sequence[int]
    """Number of channels at each layer."""
    kernel_size: Union[Sequence[int]]
    """Size of the convolutional filters."""
    padding: str = 'CIRCULAR'
    """Type of padding. Must be set to 'CIRCULAR' for OBC and 'SAME' for OBC."""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    param_dtype: DType = jnp.complex128
    """The dtype of all parameters."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""
    activation: Callable = nn.gelu
    """Hidden activation function."""
    output_activation: Optional[Callable] = None
    """Output activation function."""

    backflow_kernel_init: NNInitFunc = normal(1e-3)
    """Initializer for the backflow mixing weights."""
    jastrow_kernel_init: NNInitFunc = normal(5e-1)
    """Initializer for the Jastrow weights."""

    density: float = 1.0
    """Filling factor."""

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        depth = len(self.features)
        kernel_sizes = depth * (self.kernel_size,)  # if isinstance(self.kernel_size, int) else self.kernel_size
        assert len(kernel_sizes) == depth
        cnn = partial(nn.Conv, padding=self.padding, use_bias=self.use_bias, param_dtype=self.param_dtype,
                      kernel_init=self.kernel_init, bias_init=self.bias_init)

        a = nn.Dense(1, use_bias=False, kernel_init=self.backflow_kernel_init, param_dtype=self.param_dtype)

        d_max = np.floor_divide(self.graph.extent, 2).sum()
        sd = _min_image_distance(self.graph.positions, self.graph.extent)
        d = np.linalg.norm(sd, ord=1, axis=-1).astype(int)
        jastrow = self.param('Jastrow', self.jastrow_kernel_init, (d_max + 1,), self.param_dtype)

        x0 = x.copy()
        x = x.reshape(*shape[:-1], *self.graph.extent, 1)
        residual = x.copy()
        for i, (feature, kernel_size) in enumerate(zip(self.features, kernel_sizes)):
            if i:
                x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
                x = self.activation(x)
            else:
                x = x / self.density - 1.0
            x = cnn(name=f"CNN_{i}", features=feature, kernel_size=kernel_size)(x)
            if i % 2:
                x += residual
                residual = x.copy()
        if self.output_activation:
            x = self.output_activation(x)
        x_tilde = x0 + a(x).reshape(shape)

        return jnp.einsum('...i,ij,...j->...', x_tilde, jastrow[(d,)], x_tilde)
