import numpy as np
from My_Sobel import sobel as gradient
from My_Laplacian import laplacian
import matplotlib.pyplot as plt

def update(b, n, Db, Dn, beta, mu, dt):
    """Update the data b and n"""
    grad_b = Db * gradient(b)
    b += (gradient(grad_b) + beta * b - mu * b) * dt

    # lap_n = Dn * laplacian(n)
    # c = 3 * beta * b
    n += (Dn * laplacian(n) - 3 * beta * b) * dt
    # n += (lap_n - c) * dt

    return b, n

def initialize(N, b_dens, n_dens):
    """Initialize b and n."""
    b = np.zeros((N, N))
    x = N // 2
    y = N // 2

    # 0.33 * b_dens 组
    b[x - 6: x + 6, y - 6: y + 6] = 0.33 * b_dens
    b[x - 7, y - 5: y + 5] = 0.33 * b_dens
    b[x + 6, y - 5:y + 5] = 0.33 * b_dens
    b[x - 5:x + 5, y - 7] = 0.33 * b_dens
    b[x - 5:x + 5, y + 6] = 0.33 * b_dens
    b[x - 8, y - 2: y + 2] = 0.33 * b_dens
    b[x + 7, y - 2: y + 2] = 0.33 * b_dens
    b[x - 2: x + 2, y - 8] = 0.33 * b_dens
    b[x - 2: x + 2, y + 7] = 0.33 * b_dens

    # 0.66 * b_dens 组
    b[x - 4: x + 4, y - 5: y + 5] = 0.66 * b_dens
    b[x - 5, y - 4: y + 4] = 0.66 * b_dens
    b[x + 4, y - 4: y + 4] = 0.66 * b_dens
    b[x - 6, y - 2: y + 2] = 0.66 * b_dens
    b[x + 5, y - 2: y + 2] = 0.66 * b_dens
    b[x - 2: x + 2, y - 6] = 0.66 * b_dens
    b[x - 2: x + 2, y + 5] = 0.66 * b_dens

    # 1.0 * b_dens 组
    b[x-2:x+2, y-2:y+2] = b_dens
    b[x-3, y-1:y+1] = 1.0 * b_dens
    b[x+2, y-1:y+1] = 1.0 * b_dens
    b[x-1:x+1, y-3] = 1.0 * b_dens
    b[x-1:x+1, y+2] = 1.0 * b_dens

    # n: 1-18
    n = n_dens * np.ones((N, N))

    return b, n

def draw(b, n):
    """return the matplotlib artists for animation"""
    fig, ax = plt.subplots(1, 2, figsize=(5.65, 3))
    imb = ax[0].imshow(b, animated=True, vmin=0, cmap='Greys')
    imn = ax[1].imshow(n, animated=True, vmax=1, cmap='Greys')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('b')
    ax[1].set_title('n')

    return fig, imb, imn

def main():
    N = 256
    D0 = 0.0125  # D0: 0.0125, 0.125, 0.625
    Db = D0 * np.ones((N, N))
    Dn = 0.25  # Dn 表示营养物质的扩散相关系数
    a = 1.0
    deta = 0.077
    bmax = 50
    bmin = 0.05 * bmax
    # k 确定扩散类型——线性或非线性。 k=0 表示琼脂是软的，k=1 表示琼脂是硬的
    k = np.zeros((N, N))
    mu = np.ones((N, N))

    # 初始化 b, n
    b, n = initialize(N, 1., 6.)

    # 设置模型 steps
    dt = 1.0
    steps = 10000
    for step in range(steps):
        # 确定 k
        B_count = 0
        n_sum = 0
        for i in range(int(N)):
            for j in range(int(N)):

                if a - deta * b[i, j] <= 0.6:
                    k[i, j] = 0
                elif a - deta * b[i, j] >= 0.63:
                    k[i, j] = 1

                Db[i, j] = D0 * b[i, j] ** k[i, j]
                mu[i, j] = 1 / (1 + 4 * n[i, j])

                if b[i, j] > bmin:
                    B_count += 1
                    n_sum += n[i, j]

                if n[i, j] < 0:
                    n[i, j] = 0

        if B_count == 0:
            N = 1
        else:
            N = n_sum / B_count

        if N > 0.9:
            beta = 0.3 * np.ones_like(n)
        else:
            beta = n

        b, n = update(b, n, Db, Dn, beta, mu, dt)

    # 绘图
    # draw(b, n)


    return


if __name__ == '__main__':
    main()