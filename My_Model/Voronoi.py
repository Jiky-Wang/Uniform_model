import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
from Circle import getCircle
import time


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
        {(0, 0): (x0, y0), (0, 1): (x0, y1), (0, 2): (x0, y2), ...}
        :param min_distance Define the minor distance from neighbor sites.
        :return random_sites_2d With the format of {(0, 0): (x0, y0), (0, 1): (x0, y1), (0, 2): (x0, y2), ...}
        random_sites_1d With the format of {0: (x0, y0), 1: (x0, y1), 2:(x0, y2), ...}
        random_1d_2d With the format of {0: (0, 0), 1: (0, 1), 2: (0, 2), ...}
        """
        random_sites_2d = {}

        for i in range(self.row):
            for j in range(self.col):
                distance_min = 0.0
                while distance_min < min_distance:
                    random_sites_2d[(i, j)] = (np.random.rand()+j, np.random.rand()+i)
                    distance_list = []
                    for position_j in random_sites_2d.keys():
                        if position_j != (i, j):
                            distance_list.append(np.sqrt((random_sites_2d[position_j][0] -
                                                          random_sites_2d[(i, j)][0]) ** 2 +
                                                         (random_sites_2d[position_j][1] - random_sites_2d[(i, j)][1])
                                                         ** 2))
                    if distance_list == []:
                        break
                    if distance_list:
                        distance_min = np.min(distance_list)
                        if distance_min > min_distance:
                            break

        # 按照 random_sites_1d {0: (x0, y0), 1: (x0, y1), 2: (x0, y2), ...} 的格式将数据进行存储
        # 按照 random_1d_2d {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), ...} 的格式存储 1d 到 2d 索引的对应关系

        random_1d_index = []
        random_1d_key = []
        random_1d_val = []

        for i in range(len(random_sites_2d)):
            random_1d_index.extend([i])
        for key in random_sites_2d.keys():
            random_1d_key.extend([key])
        for val in random_sites_2d.values():
            random_1d_val.extend([val])

        random_1d_2d = dict(zip(random_1d_index, random_1d_key))
        random_sites_1d = dict(zip(random_1d_index, random_1d_val))

        return random_sites_2d, random_sites_1d, random_1d_2d

    def direct_lattice_2d(self, random_sites):
        """
        Return a dictionary of all the direct lattice's match-points (three points in a triangle) with 2-d index
        and the triangle's center of the circle.
        {[(x1, y1), (x2, y2), (x3, y3)]: (c_x1, c_y1), [(xi, yi), (xj, yj), (xk, yk)]: (c_xi, c_yi), ...}
        """
        indexs = []
        for index in random_sites.keys():
            indexs.append(index)

        potential_neighbors = list(itt.permutations(indexs, 3))  # 这里不能使用 itertools.combinations()，这里还必须得要能重复的才行
        # print(potential_neighbors[:3])
        nearest_neighbors_2d = {}
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
                nearest_neighbors_2d[(xi, yi), (xj, yj), (xk, yk)] = (x0, y0)

        return nearest_neighbors_2d

    def direct_lattice_1d(self, random_sites_1d):
        """
        Return a dictionary of all the direct lattice's match-points (three points in a triangle) with 1-d index
        and the triangle's center of the circle.
        {[1, 2, 3]: (c_x1, c_y1), [i, j, k]: (c_xi, c_yi), ...}
        """
        indexs = []
        for index in random_sites_1d.keys():
            indexs.append(index)

        potential_neighbors = list(itt.permutations(indexs, 3))  # 这里不能使用 itertools.combinations()，这里还必须得要能重复的才行
        # print('Length of potential_neighbors', len(potential_neighbors))
        nearest_neighbors_1d = {}
        for triangle in potential_neighbors:
            # print(triangle[0], triangle[1], triangle[2])
            p1 = random_sites_1d[triangle[0]]
            p2 = random_sites_1d[triangle[1]]
            p3 = random_sites_1d[triangle[2]]
            (x0, y0), r = getCircle(p1, p2, p3)
            dists = []
            for site in random_sites_1d.values():
                # print('site0: ', site[0])
                dists.extend([np.sqrt((x0 - site[0]) ** 2 + (y0 - site[1]) ** 2)])
            dist_min = np.min(dists)
            if dist_min >= r:
                nearest_neighbors_1d[(triangle[0], triangle[1], triangle[2])] = (x0, y0)
        # 这一段执行完之后仍然有问题，有一些非常离谱的圆心坐标，不知道是不是正常的，从数量上来说倒没有那么离谱的多，只是有的圆心甚至上千，很疑惑

        return nearest_neighbors_1d

    def voronoi_lattice(self):

        pass

    def calculate_lij(self, lattice_sites, nearest_neighbors):
        """
        Calculate the necessary part lij of the wij.
        :param lattice_sites: A dictionary with all lattice sites and their coordinates.
        :param nearest_neighbors: A dictionary with every point in a triangle and its circle center.
        :return: lij_dict A dictionary with all lij and their length with a format of
        {((0, 0), (0, 1)): dist1, ((0, 0), (1, 0)): dist2, ...}
        """
        lij_dict = {}
        indexss = []
        for indexs in nearest_neighbors.keys():
            indexss.append(indexs)

        # point = (0, 0)
        for triangle in indexss:
            l1 = np.sqrt((lattice_sites[triangle[0]][0] - lattice_sites[triangle[1]][0]) ** 2 +
                         (lattice_sites[triangle[0]][1] - lattice_sites[triangle[1]][1]) ** 2)
            lij_dict[(triangle[0], triangle[1])] = l1
            # lij[(triangle[1], triangle[0])] = l1
            l2 = np.sqrt((lattice_sites[triangle[0]][0] - lattice_sites[triangle[2]][0]) ** 2 +
                         (lattice_sites[triangle[0]][1] - lattice_sites[triangle[2]][1]) ** 2)
            lij_dict[(triangle[0], triangle[2])] = l2
            # lij[(triangle[2], triangle[0])] = l2

        return lij_dict

    def calculate_fij(self, nearest_neighbors):
        """
        Each line has a fij or fji
        :param nearest_neighbors: The nearest_neighbors from function direct_lattice with a format of
        {[(x1, y1), (x2, y2), (x3, y3)]: (c_x1, c_y1), [(xi, yi), (xj, yj), (xk, yk)]: (c_xi, c_yi), ...}
        :return: fij_dict triangle near the line with format of {((0, 0), (0, 1)): ((tri0), (tri1)), ...}
        """
        fij_dict = {}


        return fij_dict


if __name__ == '__main__':
    start_time = time.time()
    row = 8
    col = 8
    vor = Voronoi(row, col)
    random_sites_2d, random_sites_1d, random_1d_2d = vor.random_sites(0.2)
    print(f"The random sites are: {random_sites_2d} \nThe number of random sites is: {len(random_sites_2d)}")
    nearest_neighbors = vor. direct_lattice_1d(random_sites_1d)

    # nearest_neighbors = vor.direct_lattice_2d(random_sites_2d)
    print(f"\nThe nearest neighbors: {nearest_neighbors}")
    print('The length of nearest neighbors: ', len(nearest_neighbors))
    # num = len(nearest_neighbors)
    # print(f"The nearest number is: {num}")
    # end_time = time.time()
    # print(f"\nTime: {end_time - start_time}")
