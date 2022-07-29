import numpy as np
from scipy.constants import hbar, pi
from scipy.signal import convolve2d

def Laplacian1D(N):
	K_grid = np.identity(N)
	K_kernel = np.zeros((3, 3))
	K_kernel[1, 0] =  1
	K_kernel[1, 1] = -2
	K_kernel[1, 2] =  1
	K_grid = convolve2d(K_grid, K_kernel, mode='same') / 2
	return K_grid


def Laplacian2D(N):
	p_sq_grid = Laplacian1D(N)
	p_sq_x 	  = np.identity(N**2)
	p_sq_y 	  = np.identity(N**2)
	for big_i in range(N):
		i0 = big_i * N
		i1 = (big_i + 1) * N
		p_sq_x[i0:i1, i0:i1] = np.ones((N, N)) * p_sq_grid
		for big_j in range(N):
			j0 = big_j * N
			j1 = (big_j + 1) * N
			p_sq_y[i0:i1,j0:j1] = np.identity(N) * p_sq_grid[big_i, big_j]
	K_grid = p_sq_x + p_sq_y
	return K_grid
