#! /bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, hbar
from scipy.signal import convolve2d
from numpy.polynomial.hermite import Hermite as H

# Set up constants
# Number of points (For now, a square domain)
N = 40
# bounds for 2D particle box
x_min, x_max = (0, 1e-9)
x_center = (x_min - x_max) / 2
y_min, y_max = (0, 1e-9)
y_center = (y_min - y_max) / 2
L_x = x_max - x_min
L_y = y_max - y_min
# "spring" constant for potential
k = 1
h_bar = hbar
# electron mass
m = 9.11e-31

# spacial discretization
x_list = np.linspace(x_min, x_max, N)
y_list = np.linspace(y_min, y_max, N)
x_grid, y_grid = np.meshgrid(x_list, y_list)
dx = x_list[1] - x_list[0]
dy = y_list[1] - y_list[0]

# Potential Energy - Set up potential and corresponding matrix
V_list = np.zeros(N * N)
V_grid = np.diag(V_list)

# Set up momentum_sq operator
# Done by convolving a kernel with an identity matrix in order to correctly
# construct the momentum_sq operator based on a 2nd finite difference
# derivative
p_sq_grid = np.identity(N)
p_sq_kernel = np.zeros((3, 3))
p_sq_kernel[1, 0] =  1
p_sq_kernel[1, 1] = -2
p_sq_kernel[1, 2] =  1
p_sq_grid = -h_bar**2 * convolve2d(p_sq_grid, p_sq_kernel, mode='same') / (2 * m * dx**2)

p_sq_x = np.identity(N**2)
p_sq_y = np.identity(N**2)

for big_i in range(N):
	i0 = big_i * N
	i1 = (big_i + 1) * N
	p_sq_x[i0:i1, i0:i1] = np.ones((N, N)) * p_sq_grid
	for big_j in range(N):
		j0 = big_j * N
		j1 = (big_j + 1) * N
		p_sq_y[i0:i1,j0:j1] = np.identity(N) * p_sq_grid[big_i, big_j]

K_grid = p_sq_x + p_sq_y

"""
for i in range(N**2):
	for j in range(N**2):
		print("%.1f" %(p_sq_x[i, j]) , end='\t')
	print()
print()
print()

for i in range(N**2):
	for j in range(N**2):
		print("%.1f" %(p_sq_y[i, j]) , end='\t')
	print()
"""


# construct the Hamiltonian!
H_grid = K_grid + V_grid

# Calculate Eigenvalues, they come out sorted via eigh
eigenval, eigenvec = np.linalg.eigh(H_grid)
eigengrid = np.array([eig.reshape((N, N)) for eig in eigenvec.T])


for k in np.arange(N):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x_grid, y_grid, eigengrid[k] * eigengrid[k])
	plt.show()
	plt.close()
exit()


# Compare Eigenvalues to analytic solutions
# Constructing Analytic Eigenvectors
N_list = np.array(np.arange(1, N+1))
N_grid, x_grid = np.meshgrid(N_list, x_list)
# E = np.sqrt(2 * dx / L) * np.sin(N_grid * pi * x_grid / L)
# Eigenvalues for finite square well are modeled considering the -1th and Nth (out of bounds) nodes are the first nodes with V->inf, thereby making the well L + 2*dx wide, with the origin at the left point
E = np.sqrt(2 * dx / (L + dx + dx)) * np.sin(N_grid * pi * (x_grid + dx) / (L + dx + dx))
w = np.sqrt(k / m)

# Calculation of errors
error           = E * E - eigenvec *  eigenvec
sq_err_grid     = error * error
avg_sq_err      = np.sum(sq_err_grid, 1) / N
std_dev_sq_err  = np.std(sq_err_grid, 1)
max_sq_err      = np.max(sq_err_grid, 1)
print("-----------------------------------------------")
print("n\tAvg.Err\t\tStd.Dev\t\tMax.Err")
n = 0
for avg_, std_, max_ in zip(avg_sq_err, std_dev_sq_err, max_sq_err):
    print("%d\t%e\t%e\t%e" %(n, avg_, std_, max_))
    n += 1
print("-----------------------------------------------")

# Plotting Eigenvalues
for i in range(5):
    plt.plot(x_list, error[:, i]**2, label="Error")
    plt.plot(x_list, eigenvec[:, i]**2, label="Numerical")
    plt.plot(x_list, E[:, i]**2, label="Analytical")
    plt.legend()
    plt.show()
plt.close()
