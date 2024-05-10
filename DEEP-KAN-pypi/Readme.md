# Deep-KAN PyPi

```
pip install deep-kan
```

This repository contains implementations of Kolmogorov-Arnold Networks (KAN) in PyTorch, including SplineLinearLayer, DeepKAN, and ChebyshevKANLayer.

## SplineLinearLayer

The SplineLinearLayer is a PyTorch module that combines a linear layer with a spline kernel activation function. It takes the following arguments:

- `input_dim` (int): Dimensionality of the input data.
- `output_dim` (int): Dimensionality of the output data.
- `num_knots` (int, optional): Number of knots for the spline. Default is 5.
- `spline_order` (int, optional): Order of the spline. Default is 3.
- `noise_scale` (float, optional): Scale of the noise. Default is 0.1.
- `base_scale` (float, optional): Scale of the base weights. Default is 1.0.
- `spline_scale` (float, optional): Scale of the spline weights. Default is 1.0.
- `activation` (torch.nn.Module, optional): Activation function to use. Default is torch.nn.SiLU.
- `grid_epsilon` (float, optional): Epsilon value for the grid. Default is 0.02.
- `grid_range` (list, optional): Range of the grid. Default is [-1, 1].
- `standalone_spline_scaling` (bool, optional): Whether to use standalone spline scaling. Default is True.

### Example usage:

```
import torch.nn as nn
from deepkan import SplineLinearLayer

input_dim = 10
output_dim = 5
layer = SplineLinearLayer(input_dim, output_dim)

```
## DeepKAN

The DeepKAN class is a PyTorch module that implements a neural network with Kolmogorov-Arnold Networks (KAN) layers. It takes the following arguments:

- `input_dim` (int): Dimensionality of the input data.
- `hidden_layers` (list): List of hidden layer dimensions (the last one should be the target layer).
- `num_knots` (int, optional): Number of knots for the spline. Default is 5.
- `spline_order` (int, optional): Order of the spline. Default is 3.
- `noise_scale` (float, optional): Scale of the noise. Default is 0.1.
- `base_scale` (float, optional): Scale of the base weights. Default is 1.0.
- `spline_scale` (float, optional): Scale of the spline weights. Default is 1.0.
- `activation` (torch.nn.Module, optional): Activation function to use. Default is torch.nn.SiLU.
- `grid_epsilon` (float, optional): Epsilon value for the grid. Default is 0.02.
- `grid_range` (list, optional): Range of the grid. Default is [-1, 1].


### Example usage:

```
import torch.nn as nn
from deepkan import DeepKAN

input_dim = 28 * 28  # Flattened MNIST image size
hidden_layers = [64, 10]  # Hidden layer dimensions
model = DeepKAN(input_dim, hidden_layers)
```

## ChebyshevKANLayer

The ChebyshevKANLayer is a PyTorch module that implements a Chebyshev Kernel Activation Network layer. It takes the following arguments:

- `input_dim` (int): Dimensionality of the input data.
- `output_dim` (int): Dimensionality of the output data.
- `degree` (int): Degree of the Chebyshev polynomial.

### Example usage:

```
import torch.nn as nn
from deepkan import ChebyshevKANLayer

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.chebykan1 = ChebyshevKANLayer(1, 32, 4)
        self.chebykan2 = ChebyshevKANLayer(32, 16, 4)
        self.chebykan3 = ChebyshevKANLayer(16, 1, 4)
    def forward(self, x):
        x = self.chebykan1(x)
        x = self.chebykan2(x)
        x = self.chebykan3(x)
        return x
```
# Guide for Using RBFKAN

The `RBFKAN` class is a PyTorch module that implements a Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN). This module combines the traditional KAN with a radial basis function (RBF) kernel to capture non-linear relationships in the input data. It is designed to be used as a layer in a larger neural network architecture.

## Class RBFLinear

The `RBFLinear` class is a sub-module that implements the RBF kernel transformation of the input data. It takes the following arguments:

- `in_features`: The number of input features.
- `out_features`: The number of output features.
- `grid_min` (default=-2.0): The minimum value of the grid for the RBF kernel.
- `grid_max` (default=2.0): The maximum value of the grid for the RBF kernel.
- `num_grids` (default=8): The number of grid points for the RBF kernel.
- `spline_weight_init_scale` (default=0.1): The scale factor for initializing the spline weights.

The `forward` method of this class applies the RBF kernel transformation to the input data.

## Class RBFKANLayer

The `RBFKANLayer` class is the main building block of the `RBFKAN` module. It combines the `RBFLinear` transformation with a traditional linear layer. It takes the following arguments:

- `input_dim`: The number of input features.
- `output_dim`: The number of output features.
- `grid_min`, `grid_max`, `num_grids`, and `spline_weight_init_scale`: Same as in `RBFLinear`.
- `use_base_update` (default=True): Whether to use the traditional linear layer in addition to the RBF kernel.
- `base_activation` (default=nn.SiLU()): The activation function for the traditional linear layer.

The `forward` method of this class applies the RBF kernel transformation and, optionally, the traditional linear layer to the input data.

## Class RBFKAN

The `RBFKAN` class is the main module that combines multiple `RBFKANLayer` instances into a larger neural network architecture. It takes the following arguments:

- `layers_hidden`: A list of integers representing the number of features in each hidden layer, including the input and output layers.
- `grid_min`, `grid_max`, `num_grids`, `use_base_update`, `base_activation`, and `spline_weight_init_scale`: Same as in `RBFKANLayer`.

The `forward` method of this class applies the sequence of `RBFKANLayer` instances to the input data, passing the output of one layer as the input to the next layer.

## Usage Example

```
# Define the input and output dimensions
input_dim = 10
output_dim = 5

# Define the hidden layer dimensions
hidden_dims = [16, 32, 16]

# Create an RBFKAN instance
model = RBFKAN([input_dim] + hidden_dims + [output_dim])

# Forward pass with some input data
x = torch.randn(batch_size, input_dim)
y = model(x)

```
