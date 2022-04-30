import cv2 as cv
import numpy as np

def sobel(mat):
    """Calculate the gradient operator."""
    res_x = np.zeros_like(mat)
    res_y = np.zeros_like(mat)
    x_neighbors = [
        (0.0, (-1, 0)),
        (1.0, (0, -1)),
        (-1.0, (0, 1)),
        (0.0, (1, 0)),
        (-0.5, (-1, 1)),
        (0.5, (1, -1)),
        (0.5, (-1, -1)),
        (-0.5, (1, 1))
    ]
    y_neighbors = [
        (-1.0, (-1, 0)),
        (0.0, (0, -1)),
        (0.0, (0, 1)),
        (1.0, (1, 0)),
        (-0.5, (-1, 1)),
        (0.5, (1, -1)),
        (-0.5, (-1, -1)),
        (0.5, (1, 1))
    ]

    # calculate Gx
    for weight, neigh in x_neighbors:
        res_x += weight * np.roll(mat, neigh, (0, 1))

    # calculate Gy
    for weight, neigh in y_neighbors:
        res_y += weight * np.roll(mat, neigh, (0, 1))

    # calculate G
    res = np.float32(np.abs(res_x) + np.abs(res_y))

    return res


if __name__ == '__main__':
    mat = np.ones((10, 10))
    mat[4:6, 4:6] = 0.5
    res = sobel(mat)
    print(res.shape)

