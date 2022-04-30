import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation  # this is needed for the animation API
from matplotlib.colors import Normalize  # this is needed to rescale the color during the simulation


class Uniform_GrayScott():
    """Class to model a uniform Gray-Scott Reaction-Diffusion equation"""

    def __init__(self, N, Dn, Db, f, k):
        self.N = N
        # self.Nt = Nt
        self.Dn = Dn
        self.Db = Db
        self.f = f
        self.k = k
        self.n = np.ones((N, N), dtype=np.float32)
        self.b = np.zeros((N, N), dtype=np.float32)

    def laplacian(self, mat):
        """Calculate the laplacian operator.
        This function applies a discretized Laplacian
        in periodic boundary conditions to a matrix
        For more information see
        https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
        """
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

    def gradient(self, mat):
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

    def initialize(self, b_dens, n_dens):
        """Setting up the initial condition"""
        self.b = np.zeros((self.N, self.N), dtype=np.float32)
        # self.n = np.ones((self.N, self.N), dtype=np.float32)
        N, N2 = self.N, int(self.N / 2)
        # 0.33 * b_dens 组
        self.b[N2 - 6: N2 + 6, N2 - 6: N2 + 6] = 0.33 * b_dens
        self.b[N2 - 7, N2 - 5: N2 + 5] = 0.33 * b_dens
        self.b[N2 + 6, N2 - 5: N2 + 5] = 0.33 * b_dens
        self.b[N2 - 5: N2 + 5, N2 - 7] = 0.33 * b_dens
        self.b[N2 - 5: N2 + 5, N2 + 6] = 0.33 * b_dens
        self.b[N2 - 8, N2 - 2: N2 + 2] = 0.33 * b_dens
        self.b[N2 + 7, N2 - 2: N2 + 2] = 0.33 * b_dens
        self.b[N2 - 2: N2 + 2, N2 - 8] = 0.33 * b_dens
        self.b[N2 - 2: N2 + 2, N2 + 7] = 0.33 * b_dens

        # 0.66 * b_dens 组
        self.b[N2 - 4: N2 + 4, N2 - 5: N2 + 5] = 0.66 * b_dens
        self.b[N2 - 5, N2 - 4: N2 + 4] = 0.66 * b_dens
        self.b[N2 + 4, N2 - 4: N2 + 4] = 0.66 * b_dens
        self.b[N2 - 6, N2 - 2: N2 + 2] = 0.66 * b_dens
        self.b[N2 + 5, N2 - 2: N2 + 2] = 0.66 * b_dens
        self.b[N2 - 2: N2 + 2, N2 - 6] = 0.66 * b_dens
        self.b[N2 - 2: N2 + 2, N2 + 5] = 0.66 * b_dens

        # 1.0 * b_dens 组
        self.b[N2 - 2: N2 + 2, N2 - 2: N2 + 2] = b_dens
        self.b[N2 - 3, N2 - 1: N2 + 1] = 1.0 * b_dens
        self.b[N2 + 2, N2 - 1: N2 + 1] = 1.0 * b_dens
        self.b[N2 - 1: N2 + 1, N2 - 3] = 1.0 * b_dens
        self.b[N2 - 1: N2 + 1, N2 + 2] = 1.0 * b_dens

        # n: 1-18
        self.n = n_dens * np.ones((self.N, self.N), dtype=np.float32)

        return

    def calculate(self):
        """Calculate the results"""
        self.n += Dn * self.laplacian(self.n) - self.n * self.b ** 2 + self.f * (1 - self.n)
        self.b += Db * self.laplacian(self.b) + self.n * self.b ** 2 - (self.f + self.k) * self.b

        return

    def plot_static(self, Nt, b_dens, n_dens):
        """Plot the static concentration of b & n"""
        # Initialize
        self.initialize(b_dens, n_dens)

        # Calculate
        for i in range(int(Nt)):
            self.calculate()

        f = plt.figure(figsize=(5.65, 3), dpi=400, facecolor='w', edgecolor='k')
        sp = f.add_subplot(1, 2, 1)
        plt.pcolor(self.b, cmap=plt.cm.RdBu)
        plt.axis('tight')
        plt.title('bacterial')

        sp = f.add_subplot(1, 2, 2)
        plt.pcolor(self.n, cmap=plt.cm.RdBu)
        plt.axis('tight')
        plt.title('nutrient')
        plt.show()

    def updatefig(self, frame_id, updates_per_frame, imb, imn):
        """Takes care of the matplotlib-artist update in the animation"""
        # update x times before updating the frame
        for i in range(updates_per_frame):
            self.calculate()

        # update the frame
        imb.set_array(self.b)
        imn.set_array(self.n)

        # renormalize the colors
        imn.set_norm(Normalize(vmin=np.amin(self.n), vmax=np.amax(self.n)))
        imb.set_norm(Normalize(vmin=np.amin(self.b), vmax=np.amax(self.b)))

        # return the updated matplotlib objects
        return imn, imb


    def get_artists(self):
        """Draw the concentrations."""
        fig, ax = plt.subplots(1, 2, figsize=(5.65, 3))
        imb = ax[0].imshow(self.b, animated=True, cmap=plt.cm.RdBu)
        imn = ax[1].imshow(self.n, animated=True, cmap=plt.cm.RdBu)
        ax[0].set_title('bacterial')
        ax[1].set_title('nutrition')
        ax[0].axis('off')
        ax[1].axis('off')

        return fig, imb, imn

    def plot_animation(self, b_initialdens, n_initialdens):
        """Takes care of the matplotlib-artist update in the animation"""
        # Initialize
        self.initialize(b_initialdens, n_initialdens)

        fig, imb, imn = self.get_artists()

        # how many updates should be computed before a new frame is drawn
        updates_per_frame = 10

        # these are the arguments which have to passed to the update function
        animation_arguments = (updates_per_frame, imb, imn)

        # start the animation
        ani = animation.FuncAnimation(fig,  # matplotlib figure
                                      self.updatefig,  # function that takes care of the update
                                      fargs=animation_arguments,  # arguments to pass to this function
                                      interval=1,  # update every `interval` milliseconds
                                      blit=True,  # optimize the drawing update
                                      )

        # show the animation
        plt.show()




if __name__ == "__main__":
    N = 256
    Nt, Dn, Db, F, K = 3.2e2, 0.12, 0.08, 0.020, 0.050
    b_dens, n_dens = 1.0, 2.0

    model = Uniform_GrayScott(N, Dn, Db, F, K)
    # model.plot_static(Nt, b_dens, n_dens)
    model.plot_animation(b_dens, n_dens)


