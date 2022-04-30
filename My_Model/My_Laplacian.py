import numpy as np


def laplacian(mat):
    """Calculate the laplacian operator."""
    res = -1 * mat.copy()

    neighbors = [
        (0.2, (-1, 0)),
        (0.2, (0, -1)),
        (0.2, (0, 1)),
        (0.2, (1, 0)),
        (0.05, (-1, 1)),
        (0.05, (1, -1)),
        (0.05, (-1, -1)),
        (0.05, (1, 1))
    ]

    # neighbors = [
    #     (1.0, (-1, 0)),
    #     (1.0, (0, -1)),
    #     (1.0, (0, 1)),
    #     (1.0, (1, 0)),
    # ]

    # calculate res
    for weight, neigh in neighbors:
        res += weight * np.roll(mat, neigh, (0, 1))

    return res


if __name__ == '__main__':
    mat = 6 * np.ones((10, 10))
    mat[4: 6, 4: 6] = 0
    res = laplacian(mat)
    print(res.shape)


