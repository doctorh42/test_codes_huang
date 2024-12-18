# ga_util.py

import numpy as np
import random
from graph_tool.all import Graph, shortest_distance, min_spanning_tree
from deap import creator


# 初始化个体生成函数
def init_individual(read_graph, edge_weights, num_edges):
    num_vertices = read_graph.num_vertices()
    mst_edges = min_spanning_tree(read_graph, weights=edge_weights)

    individual = [0] * (num_vertices * num_vertices)
    for edge in read_graph.edges():
        if mst_edges[edge]:
            source, target = int(edge.source()), int(edge.target())
            individual[source * num_vertices + target] = 1
            individual[target * num_vertices + source] = 1

    mst_edge_count = sum(mst_edges.a)
    non_mst_edges = [(int(edge.source()), int(edge.target())) for edge in read_graph.edges() if not mst_edges[edge]]
    additional_edges_needed = num_edges - mst_edge_count
    additional_edges = random.sample(non_mst_edges, additional_edges_needed)

    for i, j in additional_edges:
        individual[i * num_vertices + j] = 1
        individual[j * num_vertices + i] = 1

    return creator.Individual(individual)


# 将个体解码为图并计算边权重属性
def individual_to_graph(individual, num_vertices, positions):
    graph = Graph(directed=False)
    graph.add_vertex(num_vertices)
    # 复制顶点属性
    graph.vertex_properties['number'] = graph.new_vertex_property("int")
    for i in range(num_vertices):
        graph.vertex_properties['number'][graph.vertex(i)] = i + 1  # 假设原始编号从 1 开始

    new_edge_weights = graph.new_edge_property("double")
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if individual[i * num_vertices + j] == 1:
                edge = graph.add_edge(graph.vertex(i), graph.vertex(j))
                source_pos = positions[graph.vertex(i)]
                target_pos = positions[graph.vertex(j)]
                euclidean_distance = np.sqrt(
                    (source_pos[0] - target_pos[0]) ** 2 + (source_pos[1] - target_pos[1]) ** 2)
                new_edge_weights[edge] = euclidean_distance
    return graph, new_edge_weights


# 定义适应度函数
def evaluate(individual, read_graph, read_pos, edge_weights):
    graph, new_edge_weights = individual_to_graph(individual, read_graph.num_vertices(), read_pos)
    num_edges = graph.num_edges()
    if num_edges != read_graph.num_edges():
        return float('inf'),
    dist_matrix = shortest_distance(graph, weights=new_edge_weights).get_2d_array(range(graph.num_vertices()))
    total_distance = np.sum(dist_matrix[dist_matrix != np.inf])
    return total_distance,


# 自定义交叉操作
def cxGraph(ind1, ind2, num_edges):
    size = len(ind1)
    point = random.randint(1, size - 1)
    new_ind1 = creator.Individual(np.concatenate((ind1[:point], ind2[point:])))
    new_ind2 = creator.Individual(np.concatenate((ind2[:point], ind1[point:])))
    new_ind1 = fix_edges(new_ind1, num_edges)
    new_ind2 = fix_edges(new_ind2, num_edges)
    return new_ind1, new_ind2


# 修正边数函数
def fix_edges(individual, num_edges):
    num_vertices = int(np.sqrt(len(individual)))
    edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices) if
             individual[i * num_vertices + j] == 1]
    non_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices) if
                 individual[i * num_vertices + j] == 0]

    if len(edges) > num_edges:
        extra_edges = random.sample(edges, len(edges) - num_edges)
        for i, j in extra_edges:
            individual[i * num_vertices + j] = 0
            individual[j * num_vertices + i] = 0
    elif len(edges) < num_edges:
        missing_edges = random.sample(non_edges, num_edges - len(edges))
        for i, j in missing_edges:
            individual[i * num_vertices + j] = 1
            individual[j * num_vertices + i] = 1
    return individual


# 自定义变异操作
def mutGraph(ind, num_edges):
    size = int(np.sqrt(len(ind)))
    edges = [(i, j) for i in range(size) for j in range(i + 1, size) if ind[i * size + j] == 1]
    non_edges = [(i, j) for i in range(size) for j in range(i + 1, size) if ind[i * size + j] == 0]

    if edges and non_edges:
        i, j = random.choice(edges)
        ind[i * size + j] = 0
        ind[j * size + i] = 0
        i, j = random.choice(non_edges)
        ind[i * size + j] = 1
        ind[j * size + i] = 1

    return ind,
