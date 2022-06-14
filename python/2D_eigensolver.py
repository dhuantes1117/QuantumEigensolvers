#! /bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi

# Domain                (2D)
# set up bounds
N = 10
x_l, x_r = (0, 1)
y_l, y_r = (0, 1)
# discretize space
x_1D = np.linspace(x_l, x_r, N)
y_1D = np.linspace(y_l, y_r, N)
x_2D, y_2D = np.meshgrid(x_1D, y_1D)
# x_list -> x_grid, y_grid

# Potential Energy      (2D)
# V(x, y)
# V_list -> V_grid
# V(x_list) -> V(x_grid, y_grid)
# V(x, y)[0] -> V_grid[0]
V_2D = np.ones(np.shape(x_2D))
# V_3D = np.array([np.diag(V_2D[i]) for i in range(N)])
print(V_3D)
exit()

# Kinetic Energy        (2D)
# p^2 / 2m = -(hbar^2 / 2m) * Del^2 = -(hbar^2 / 2m) * (d2/dx2 + d2/dy2) = px^2 / 2m + py^2 / 2m


# Hamiltonian           (2D)
# H = K + V

# Calculate Eigenvalues (2D)
# ???

# Compare Eigenvalues to analytic solutions (2D)
