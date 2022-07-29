#! /bin/python3

# Set up constants
# Set up domain
# Set up potential
# Set up perturbing potential
# Set up kinetic energy (momentum squared)
# Calculate C_mn matrices
# Calculate E^1_n values
# Calculate unperturbed eigenvectors # Calculate the transfer probabilities
# Calculate new eigenvectors
# Hurray

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, pi, e
from scipy.signal import convolve2d

# Set up constants
N = 200
N_t = 10
x_l, x_r = (0, 1e-9)
t_min, t_max = (0, 1)
k = 1e58
EMF = 1
h_bar = hbar
m = 9.11e-31
omega = 2*pi

# Set up spatial domain
x_list = np.linspace(x_l, x_r, N)
dx = x_list[1] - x_list[0]
x_c = dx + (x_r - x_l) / 2

# Set up time domain
t_list = np.linspace(t_min, t_max, N_t)
dt = t_list[1] - t_list[0]

# Set up potential
# Zero potential (square well)
V0 = np.zeros(N)
V0_grid = np.diag(V0)

# Set up perturbing potential
# Centered harmonic oscillator
V_prime = 0.5 * k * (x_list - x_c)**2
V_prime_grid = np.diag(V_prime)

# Set up kinetic energy (momentum squared)
K_grid = np.identity(N)
K_kernel = np.zeros((3, 3))
K_kernel[1, 0] =  1
K_kernel[1, 1] = -2
K_kernel[1, 2] =  1
K_grid = -h_bar**2 * convolve2d(K_grid, K_kernel, mode='same') / (2 * m * dx**2)

# Construct Hamiltonian
H0_grid = K_grid + V0_grid
H_grid = K_grid + V0_grid + V_prime_grid

# Calculate unperturbed states and energy values
E0, eigenvect0rs = np.linalg.eigh(H0_grid)

# Construct C_n(t = 0)
C_n = np.zeros(N)
C_n[0] = 1


# Constructing time dependent perturbing Hamiltonian in unperturbed basis
H_prime_0 = np.matmul(eigenvect0rs.T, np.matmul(V_prime_grid, eigenvect0rs))
# introduce time dependence (assuming H'(t) = H'_0 * f(t))
H_prime_0_t = np.resize(H_prime_0, (N_t, *np.shape(H_prime_0)))
f_t = np.cos(omega * t_list.reshape((N_t, 1, 1)))
H_prime_0_t = H_prime_0_t * f_t

E_n, E_m = np.meshgrid(E0, E0)
Energy_Diff = E_n - E_m
# Make sure to check this should be n - m and not m - n
Net_Time_Dependence_exp = np.exp(-1j * Energy_Diff)
Net_Time_Dependence_exp = np.resize(Net_Time_Dependence_exp, (N_t, *np.shape(Net_Time_Dependence_exp)))
Net_Time_Dependence_exp = Net_Time_Dependence_exp ** t_list.reshape(N_t, 1, 1)
exit()

# Enter the Runge Kutta

# plot potentials
plt.plot(x_list, V0, label="orig")
plt.plot(x_list, V_prime, label="pert")
plt.legend()
plt.show()
plt.close()

# Calculate E^1_n values
E1 = np.diagonal(H_prime_0)
E = E0 + E1

# Calculate unperturbed eigenvectors
# Calculate full eigenvectors
eigenvalues_solutions, eigenvectors_solutions = np.linalg.eigh(H_grid)
# Calculate new eigenvectors
eigenvectors = np.matmul(eigenvect0rs, C_mn)
area = np.resize(np.sum(eigenvectors ** 2, 0), (N, N))
eigenvectors = eigenvectors / np.sqrt(area)
