# Kolmogorov-Arnold Networks (KAN)

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

```

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
