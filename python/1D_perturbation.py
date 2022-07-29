#! /bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, pi, e
from scipy.signal import convolve2d

# Set up constants
N = 200
x_l, x_r = (0, 1e-9)
k = 1e58
EMF = 1
h_bar = hbar
m = 9.11e-31

# Set up domain
x_list = np.linspace(x_l, x_r, N)
dx = x_list[1] - x_list[0]
x_c = dx + (x_r - x_l) / 2

# Set up potential
# offset harmonic oscillator
# V0 = 0.5 * k * (x_list - x_c + 10 * dx)**2
# centered harmonic oscillator
# V0 = 0.5 * k * (x_list - x_c)**2
# Coulomb potential
# V0 = (-9e9) * e**2 / (x_list + dx)
# Zero potential (square well)
V0 = np.zeros(N)
V0_grid = np.diag(V0)

# Set up perturbing potential
Volt = 6e12
# No perturbation
# V_prime = np.zeros(N)
# Constant E field
# V_prime = -EMF * np.linspace(-Volt, Volt, N) * e
# Centered harmonic oscillator
V_prime = 0.5 * k * (x_list - x_c)**2
V_prime_grid = np.diag(V_prime)

plt.plot(x_list, V0, label="orig")
plt.plot(x_list, V_prime, label="pert")
plt.legend()
plt.show()
plt.close()

# Set up kinetic energy (momentum squared)
K_grid = np.identity(N)
K_kernel = np.zeros((3, 3))
K_kernel[1, 0] =  1
K_kernel[1, 1] = -2
K_kernel[1, 2] =  1
K_grid = -h_bar**2 * convolve2d(K_grid, K_kernel, mode='same') / (2 * m * dx**2)

H0_grid = K_grid + V0_grid
H_grid = K_grid + V0_grid + V_prime_grid

# Calculate unperturbed states and energy values
E0, eigenvect0rs = np.linalg.eigh(H0_grid)
E_n, E_m = np.meshgrid(E0, E0)

# Construct C_mn matrices
# C_mn = I +  <m|H_prime|n> / (E_n - E_m) + (<m|H_prime|n><n|H_prime|n>) / (E_n - E_m)**2 + [H'H' @ 1 / E_n - E_k][m, n] / (E_n)
# this should be inner product WITH eigenstates
H_prime_0 = np.matmul(eigenvect0rs.T, np.matmul(V_prime_grid, eigenvect0rs))

C0_mn = np.identity(N)

Energy_Diff = E_n - E_m
C1_mn = H_prime_0 / (Energy_Diff + np.identity(N))
diag_elim = np.ones((N, N)) - np.identity(N)
C1_mn = C1_mn * diag_elim

# Construct C2_mn
# First term in element-wise multiplication of C1 and H_prime
Term1 = -C1_mn * np.resize(np.diagonal(V_prime_grid), (N, N))
# Second term is matrix multiplication H_prime C1_mn but remove diagonal
Term2 = np.matmul(V_prime, C1_mn) * diag_elim # What are the diagonal terms
C2_mn = diag_elim * (Term1 + Term2) / (Energy_Diff + np.identity(N))

plt.imshow(C1_mn, cmap=plt.get_cmap("Greys"))
plt.title("C1_mn visualized")
plt.show()

plt.imshow(C2_mn, cmap=plt.get_cmap("Greys"))
plt.title("C2_mn visualized")
plt.show()

C_mn = C0_mn + C1_mn + C2_mn

plt.imshow(C_mn, cmap=plt.get_cmap("Greys"))
plt.title("C_mn visualized")
plt.show()


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


plt.plot(x_list, eigenvectors[:, 0])
plt.plot(x_list, eigenvectors[:, 1])
plt.plot(x_list, eigenvectors[:, 2])
plt.title("full perturbative numerical solutions")
plt.show()
plt.close()

for i in range(6):
	plt.title("full numerical solutions")
	plt.plot(x_list, eigenvectors_solutions[:, i]**2, label="full sol.")
	plt.plot(x_list, eigenvectors[:, i]**2, label="pert. sol.")
	plt.plot(x_list, eigenvect0rs[:, i]**2, label="unpert. sol.")
	plt.legend()
	plt.show()
	plt.close()

plt.imshow(np.matmul(eigenvectors.T, eigenvectors), cmap=plt.get_cmap("Greys"))
plt.title("orthoganality of perturbed eigenvectors")
plt.show()
plt.close()

exit()

print("normalizations")
print(np.sum(eigenvectors, 1))


# Plotting!
plt.plot(x_list, eigenvect0rs[:, 0] - eigenvectors[:, 0])
plt.plot(x_list, eigenvect0rs[:, 1] - eigenvectors[:, 1])
plt.plot(x_list, eigenvect0rs[:, 2] - eigenvectors[:, 2])
plt.plot(x_list, eigenvect0rs[:, 3] - eigenvectors[:, 3])
#plt.plot(x_list, eigenvect0rs[:, 4] - eigenvectors[:, 4])
#plt.plot(x_list, eigenvect0rs[:, 5] - eigenvectors[:, 5])
plt.title("error")
plt.show()
plt.close()

plt.plot(x_list, eigenvect0rs[:, 0])
plt.plot(x_list, eigenvect0rs[:, 1])
plt.plot(x_list, eigenvect0rs[:, 2])
plt.plot(x_list, eigenvect0rs[:, 3])
#plt.plot(x_list, eigenvect0rs[:, 4])
#plt.plot(x_list, eigenvect0rs[:, 5])
plt.title("unperturbed")
plt.show()
plt.close()

plt.plot(x_list, eigenvectors[:, 0])
plt.plot(x_list, eigenvectors[:, 1])
plt.plot(x_list, eigenvectors[:, 2])
plt.plot(x_list, eigenvectors[:, 3])
#plt.plot(x_list, eigenvectors[:, 4])
#plt.plot(x_list, eigenvectors[:, 5])
plt.title("perturbed")
plt.show()
plt.close()

plt.plot(x_list, eigenvectors[:, 0]**2, 'blue')
plt.plot(x_list, eigenvectors[:, 1]**2, 'blue')
plt.plot(x_list, eigenvectors[:, 2]**2, 'blue')
plt.plot(x_list, eigenvect0rs[:, 0]**2, 'orange')
plt.plot(x_list, eigenvect0rs[:, 1]**2, 'orange')
plt.plot(x_list, eigenvect0rs[:, 2]**2, 'orange')
plt.title("unperturbed and perturbed")
plt.show()
plt.close()

plt.imshow(H_prime_0, cmap=plt.get_cmap("Greys"))
plt.title("Effect of perturbing potential on eigenstates")
plt.show()

# Hurray!
