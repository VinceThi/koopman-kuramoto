import pytest
from graphs.get_constants import *


# create a complete graph with 4 vertices
g1 = complete_graph(4)

# tests for delete_self_loops function


def test_oneloop():
    g2 = g1.copy()
    g2.add_edge(0, 0)
    assert list(g2.edges()) != list(g1.edges())
    g2 = delete_self_loops(g2)
    assert list(g2.edges()) == list(g1.edges())


def test_twoloops():
    g2 = g1.copy()
    g2.add_edge_list([(0, 0), (2, 2)])
    assert list(g2.edges()) != list(g1.edges())
    g2 = delete_self_loops(g2)
    assert list(g2.edges()) == list(g1.edges())


def test_fourloops():
    g2 = g1.copy()
    g2.add_edge_list([(0, 0), (1, 1), (2, 2), (3, 3)])
    assert list(g2.edges()) != list(g1.edges())
    g2 = delete_self_loops(g2)
    assert list(g2.edges()) == list(g1.edges())


if __name__ == "__main__":
    pytest.main()
