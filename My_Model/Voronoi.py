import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
from Circle import getCircle
import datetime


class Voronoi():
    """
    This class if for build a Voronoi lattice.
    """
    def __init__(self, row, col):
        """
        初始化Voronoi
        :param row: The number of boundary square in row.
        :param col:The number of boundary square in column
        """
        self.row = row
        self.col = col

    def random_sites(self, min_distance):
        """
        Return a dictionary of each lattice site and their coordinate.
        {(0, 0): (x0, y0), (0, 1): (x0, y1), (0, 2): (x0, y2, ...)}
        """
        random_sites = {}
        # for i in range(self.row):
        #     for j in range(self.col):
        #         random_sites[(i, j)] = (0, 0)

        for i in range(self.row):
            for j in range(self.col):
                distance_min = 0.0
                while distance_min < min_distance:
                    random_sites[(i, j)] = (np.random.rand()+j, np.random.rand()+i)
                    distance_list = []
                    for position_j in random_sites.keys():
                        if position_j != (i, j):
                            distance_list.append(np.sqrt((random_sites[position_j][0] -
                                                          random_sites[(i, j)][0]) ** 2 +
                                                         (random_sites[position_j][1] - random_sites[(i, j)][1]) ** 2))
                    if distance_list == []:
                        break
                    if distance_list:
                        distance_min = np.min(distance_list)
                        if distance_min > min_distance:
                            break
        return random_sites

    def direct_lattice(self, random_sites):
        """
        Return a dictionary of all the direct lattice's match-points (three points in a triangle)
        and the triangle's center of the circle.
        {[(x1, y1), (x2, y2), (x3, y3)]: (c_x1, c_y1), [(xi, yi), (xj, yj), (xk, yk)]: (c_xi, c_yi), ...}
        """
        indexs = []
        for index in random_sites.keys():
            indexs.append(index)

        potential_neighbors = list(itt.combinations(indexs, 3))
        # print(potential_neighbors[:3])
        nearest_neighbors = {}
        for triangle in potential_neighbors:
            # print(triangle[0])
            (xi, yi) = random_sites[triangle[0]]
            (xj, yj) = random_sites[triangle[1]]
            (xk, yk) = random_sites[triangle[2]]
            p1, p2, p3 = (xi, yi), (xj, yj), (xk, yk)
            (x0, y0), r = getCircle(p1, p2, p3)
            dist = []
            for site in random_sites.values():
                # print('site0: ', site[0])
                dist.append(np.sqrt((site[0] - x0) ** 2 + (site[1] - y0) ** 2))
            dist_min = np.min(dist)
            if dist_min < r:
                continue
            else:
                nearest_neighbors[(xi, yi), (xj, yj), (xk, yk)] = (x0, y0)

        return nearest_neighbors

    def voronoi_lattice(self):

        pass


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    row = 8
    col = 8
    vor = Voronoi(row, col)
    random_sites = vor.random_sites(0.2)
    print(f"The random sites are: {random_sites} \nThe number of random sites is: {len(random_sites)}")
    nearest_neighbors = vor.direct_lattice(random_sites)
    print(f"\nThe nearest neighbors: {nearest_neighbors}")
    # num = 0
    # for key in nearest_neighbors.keys():
    #     num += 1

    num = len(nearest_neighbors)
    print(f"The nearest number is: {num}")
    end_time = datetime.datetime.now()
    print(f"\nTime: {end_time - start_time}")
