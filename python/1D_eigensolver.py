#! /bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar
from scipy.signal import convolve2d
from numpy.polynomial.hermite import Hermite as H

# Set up constants
# Number of points
N = 20
# bounds for particle locations
x_l, x_r = (0, 1e-9)
x_c = (x_r - x_l) / 2
L = x_r - x_l
# "spring" constant for potential
k = 1
h_bar = hbar
# electron mass
m = 9.11e-31

# spacial discretization
x_list = np.linspace(x_l, x_r, N)
dx = x_list[1] - x_list[0]

# Potential Energy - Set up potential and corresponding matrix
V_list = np.zeros(N)
V_grid = np.diag(V_list)

# Set up Kinetic Energy operator
# Done by convolving a kernel with an identity matrix in order to correctly
# construct the kinetic energy operator based on a 2nd finite difference
# derivative
K_grid = np.identity(N)
K_kernel = np.zeros((3, 3))
K_kernel[1, 0] =  1
K_kernel[1, 1] = -2
K_kernel[1, 2] =  1
K_grid = -h_bar**2 * convolve2d(K_grid, K_kernel, mode='same') / (2 * m * dx**2)

# construct the Hamiltonian!
H_grid = K_grid + V_grid

# Calculate Eigenvalues, they come out sorted via eigh
eigenval, eigenvec = np.linalg.eigh(H_grid)

# Compare Eigenvalues to analytic solutions
# Constructing Analytic Eigenvectors
N_list = np.array(np.arange(1, N+1))
N_grid, x_grid = np.meshgrid(N_list, x_list)
# E = np.sqrt(2 * dx / L) * np.sin(N_grid * pi * x_grid / L)
# Eigenvalues for finite square well are modeled considering the -1th and Nth (out of bounds) nodes are the first nodes with V->inf, thereby making the well L + 2*dx wide, with the origin at the left point
E = np.sqrt(2 * dx / (L + dx + dx)) * np.sin(N_grid * pi * (x_grid + dx) / (L + dx + dx))
w = np.sqrt(k / m)

# Attempts at constructing eigenvectors for harmonic oscillator potential
"""
Hermite = []
coeff = np.zeros(N)
for i in range(N):
    coeff[i] = 1
    coeff[i-1] = 0
    Hermite.append(H(coeff, domain=(x_l, x_r)))

Hermite_grid = np.array([H_(x_list) for H_ in Hermite])

E = np.exp(-m * w * x_grid**2 / 2 / hbar) * Hermite_grid

E = E.T
"""

# Calculation of errors
error           = E * E - eigenvec *  eigenvec
sq_err_grid     = error * error
avg_sq_err      = np.sum(sq_err_grid, 1) / N
std_dev_sq_err  = np.std(sq_err_grid, 1)
max_sq_err      = np.max(sq_err_grid, 1)
print("-------------------------------------")
print("n\tAvg.Err\t\tStd.Dev\t\tMax.Err")
n = 0
for avg_, std_, max_ in zip(avg_sq_err, std_dev_sq_err, max_sq_err):
    print("%d\t%e\t%e\t%e" %(n, avg_, std_, max_))
    n += 1
print("-------------------------------------")

# Plotting Eigenvalues
for i in range(5):
    plt.plot(x_list, error[:, i]**2, label="Error")
    plt.plot(x_list, eigenvec[:, i]**2, label="Numerical")
    plt.plot(x_list, E[:, i]**2, label="Analytical")
    plt.legend()
    plt.show()
plt.close()
