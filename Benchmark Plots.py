import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def CEC_4(solution, shift=0):
    """
    Rosenbrock Function
    f(x*) = 400
    """
    x = solution - shift
    dim = len(solution)
    res = 0
    for i in range(dim - 1):
        res += 100 * np.square(x[i]**2 - x[i+1]) + np.square(x[i] - 1)
    return res

# Create grid and multivariate normal
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = np.array([CEC_4(np.array([X[i, j], Y[i, j]])) for i in range(X.shape[0]) for j in range(X.shape[1])]).reshape(X.shape)

# Make the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('CEC_4 function value')
ax.set_title('3D Surface Plot of the Rosenbrock Function')

# Show the plot
plt.show()

