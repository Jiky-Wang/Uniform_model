import numpy as np
import matplotlib.pyplot as plt

def laplacian(mat):
    """This function applies a discretized laplacian
    in periodic boundary conditions to a matrix
    """
    neigh_mat = -4 * mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [
        (1.0, (-1, 0)),
        (1.0, (0, -1)),
        (1.0, (0, 1)),
        (1.0, (1, 0)),
    ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0, 1))

    return neigh_mat

def update(u, v, Du, Dv, f, k, dt):
    """
    Apply the Gray-Scott update formula.
    :param u:
    :param v:
    :param Du: u的扩散系数
    :param Dv: v的扩散系数
    :param f: 添加率
    :param k: 死亡率
    :param dt:
    :return:
    """
    u += (Du * laplacian(u) - u * v ** 2 + f * (1 - u)) * dt
    v += (Dv * laplacian(v) + u * v ** 2 - (f + k) * v) * dt

    return u, v

def initialize(N, random_influence=0.2):
    """
    初始化 u & v
    :return:
    """
    # get initial homogeneous concentrations
    u = (1 - random_influence) * np.ones((N, N))
    v = np.zeros((N, N))

    # put some noise on there
    u += random_influence * np.random.random((N, N))
    v += random_influence * np.random.random((N, N))

    # get center and radius for initial disturbance
    N2, r = N // 2, 50

    # apply initial disturbance
    u[N2 - r:N2 + r, N2 - r:N2 + r] = 0.50
    v[N2 - r:N2 + r, N2 - r:N2 + r] = 0.25

    return u, v

def draw(u, v):
    """Return the matplotlib artists."""
    fig, ax = plt.subplots(1, 2, figsize=(5.65, 3))
    imu = ax[0].imshow(u, animated=True, vmin=0, cmap='Greys')
    imv = ax[1].imshow(v, animated=True, vmax=1, cmap='Greys')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('u')
    ax[1].set_title('v')

    return fig, imu, imv

dt = 1.0
Du, Dv, f, k, N = 0.16, 0.08, 0.060, 0.062, 200  # original
u, v = initialize(N)
steps = 10000
for step in range(steps):
    # u += (Du * laplacian(u) - u * v ** 2 + f * (1 - u)) * dt
    # v += (Dv * laplacian(u) + u * v ** 2 - (f + k) * v) * dt
    u, v = update(u, v, Du, Dv, f, k, dt)

draw(u, v)
plt.show()

