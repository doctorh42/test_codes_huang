# traffic_tree_optimizer.py

import numpy as np
from graph_tool.all import Graph


def optimize_traffic_tree(graph, pos):
    N = graph.num_vertices()
    traffic = {e: 1 for e in graph.edges()}  # 初始交通量设为1

    def cost(graph, traffic):
        total_cost = 0
        for e in graph.edges():
            source, target = int(e.source()), int(e.target())
            distance = np.linalg.norm(np.array(pos[source]) - np.array(pos[target]))
            total_cost += distance / traffic[e]
        return total_cost / graph.num_edges()

    min_cost = cost(graph, traffic)

    for _ in range(10000):
        e1, e2 = np.random.choice(list(graph.edges()), 2, replace=False)
        if traffic[e1] > 1:
            traffic[e1] -= 1
            traffic[e2] += 1
            new_cost = cost(graph, traffic)
            if new_cost < min_cost:
                min_cost = new_cost
            else:
                traffic[e1] += 1
                traffic[e2] -= 1

    return traffic


def extract_optimal_tree(graph, traffic):
    optimal_tree = Graph(directed=False)
    optimal_tree.add_vertex(graph.num_vertices())

    for e in traffic:
        if traffic[e] > 0:
            source, target = int(e.source()), int(e.target())
            optimal_tree.add_edge(source, target)

    return optimal_tree
