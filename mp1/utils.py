import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_func2D(function, path=None, output_path='output.png'):
    x0 = np.linspace(-5, 5, 100)
    x1 = np.linspace(-5, 5, 100)
    x0, x1 = np.meshgrid(x0, x1)

    # Apply function over the grid
    Y = np.array([[function(np.array([x0[i, j], x1[i, j]])) for j in range(x0.shape[1])] for i in range(x0.shape[0])])

    # Creating subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) # https://snip.mathpix.com/1390852413/notes/e66752a4-ca81-4c0e-8e33-395d3630c881/edit

    # 3D plot
    ax0 = fig.add_subplot(121, projection='3d')
    ax0.view_init(30, -30)
    ax0.plot_surface(x0, x1, Y, cmap='terrain')

    # 2D top-down view
    contour = ax[1].contourf(x0, x1, Y, cmap='terrain')

    # Plot the path if provided
    if path is not None:
        path = np.array(path)
        y_path = np.array([function(point) for point in path])
        ax0.plot(path[:, 0], path[:, 1], y_path, color='r', marker='o', markersize=5, linestyle='-', linewidth=2)
        ax[1].plot(path[:, 0], path[:, 1], color='r', marker='o', markersize=5, linestyle='-')

    ax0.set_xlabel('x0')
    ax0.set_ylabel('x1')
    ax0.set_zlabel('y')
    ax[1].set_xlabel('x0')
    ax[1].set_ylabel('x1')
    fig.colorbar(contour, ax=ax[1], orientation='vertical')

    plt.savefig(os.path.join('./visualize', output_path))
