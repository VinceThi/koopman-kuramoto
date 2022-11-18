# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import numpy as np
import networkx as nx

N = 5
G = nx.star_graph(N-1)
A = nx.to_numpy_array(G)

print(A)
