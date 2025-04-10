# -*- coding: utf-8 -*-
# @author: Vincent Thibeault


def print_reccurence_relations(indices_list):
    for indices in indices_list:
        mu1, mu2 = indices
        print(f"{mu1 - mu2 + 2}c_({mu1+1},{mu2-1})"
              f" + {mu2 - mu1 + 2}c_({mu1-1},{mu2+1})"
              f" &= \lambda c_{(mu1, mu2)}")


if __name__ == "__main__":
    indices_degm2 = [(-3, 1), (-2, 0), (-1, -1), (0, -2), (1, -3)]
    indices_degm1 = [(-3, 2), (-2, 1), (-1, 0), (0, -1), (1, -2), (2, -3)]
    indices_deg0 = [(-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2)]
    indices_deg1 = [(-2, 3), (-1, 2), (0, 1), (1, 0), (2, -1), (3, -2)]
    indices_deg2 = [(-1, 3), (0, 2), (1, 1), (2, 0), (3, -1)]
    indices_deg3 = [(-1, 4), (0, 3), (1, 2), (2, 1), (3, 0), (4, -1)]
    indices_deg4 = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    indices_deg5 = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]

    print("\n\nDegree -2")
    print_reccurence_relations(indices_degm2)
    print("\n\n")

    print("\n\nDegree -1")
    print_reccurence_relations(indices_degm1)
    print("\n\n")

    print("\n\nDegree 0")
    print_reccurence_relations(indices_deg0)
    print("\n\n")

    print("Degree 1")
    print_reccurence_relations(indices_deg1)
    print("\n\n")

    print("Degree 2")
    print_reccurence_relations(indices_deg2)
    print("\n\n")

    print("Degree 3")
    print_reccurence_relations(indices_deg3)
    print("\n\n")

    print("Degree 4")
    print_reccurence_relations(indices_deg4)
    print("\n\n")

    print("Degree 5")
    print_reccurence_relations(indices_deg5)
    print("\n\n")
