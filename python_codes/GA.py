import random
import numpy as np
import math
from graph_tool import topology, Graph
from deap import creator

def graph_to_individual(graph):
    num_vertices = graph.num_vertices()
    individual = [0] * (num_vertices * num_vertices)
    for edge in graph.edges():
        source, target = int(edge.source()), int(edge.target())
        individual[source * num_vertices + target] = 1
        individual[target * num_vertices + source] = 1
    return individual

def individual_to_graph(individual, num_vertices):
    graph = Graph(directed=False)
    graph.add_vertex(num_vertices)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if individual[i * num_vertices + j] == 1:
                graph.add_edge(graph.vertex(i), graph.vertex(j))
    return graph

def euclidean_distance(pos, i, j):
    x_i, y_i = pos[i]
    x_j, y_j = pos[j]
    return math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)

def total_euclidean_distance(graph, pos):
    total_distance = 0.0
    num_vertices = graph.num_vertices()
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            total_distance += euclidean_distance(pos, i, j)
    return total_distance

def fitness_function(individual, num_vertices, pos):
    graph = individual_to_graph(individual, num_vertices)
    return total_euclidean_distance(graph, pos),

def create_initial_population(graph, pos, pop_size, num_additional_edges):
    population = []
    for _ in range(pop_size):
        mst, added_edges = create_mst(graph, pos)
        add_random_edges(mst, added_edges, num_additional_edges)
        individual = graph_to_individual(mst)
        # 增加多样性：随机翻转一些边
        for _ in range(num_additional_edges):
            i, j = random.sample(range(graph.num_vertices()), 2)
            idx = i * graph.num_vertices() + j
            individual[idx] = 1 - individual[idx]
        population.append(creator.Individual(individual))
    return population

def create_mst(graph, pos):
    distance_matrix = calculate_all_distances(graph, pos)
    mst = Graph(directed=False)
    mst.add_vertex(graph.num_vertices())
    edge_list = [(i, j, distance_matrix[i][j]) for i in range(graph.num_vertices()) for j in range(i + 1, graph.num_vertices())]
    edge_list.sort(key=lambda x: x[2])
    added_edges = set()
    for edge in edge_list:
        if len(added_edges) >= graph.num_vertices() - 1:
            break
        u, v, _ = edge
        mst.add_edge(u, v)
        added_edges.add((u, v))
        if topology.shortest_distance(mst, source=u, target=v) != 1:
            mst.remove_edge(mst.edge(u, v))
            added_edges.remove((u, v))
    return mst, added_edges

def add_random_edges(graph, existing_edges, num_edges_to_add):
    num_vertices = graph.num_vertices()
    while len(existing_edges) < num_vertices - 1 + num_edges_to_add:
        u, v = random.sample(range(num_vertices), 2)
        if u > v:
            u, v = v, u
        if (u, v) not in existing_edges:
            graph.add_edge(u, v)
            existing_edges.add((u, v))

def calculate_all_distances(graph, pos):
    num_vertices = graph.num_vertices()
    distance_matrix = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            distance_matrix[i][j] = euclidean_distance(pos, i, j)
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

def ensure_edge_count(ind, num_edges):
    current_edges = sum(ind)
    num_vertices = int(math.sqrt(len(ind)))
    if current_edges > num_edges:
        indices = [i for i, x in enumerate(ind) if x == 1]
        indices_to_remove = random.sample(indices, current_edges - num_edges)
        for index in indices_to_remove:
            ind[index] = 0
    elif current_edges < num_edges:
        indices = [i for i, x in enumerate(ind) if x == 0]
        indices_to_add = random.sample(indices, num_edges - current_edges)
        for index in indices_to_add:
            ind[index] = 1
    return ind

def custom_crossover(ind1, ind2, num_edges=69):
    num_vertices = int(math.sqrt(len(ind1)))
    cx_point1 = random.randint(1, len(ind1) - 2)
    cx_point2 = random.randint(cx_point1, len(ind1) - 1)
    ind1[cx_point1:cx_point2], ind2[cx_point1:cx_point2] = ind2[cx_point1:cx_point2], ind1[cx_point1:cx_point2]
    ind1 = ensure_edge_count(ind1, num_edges)
    ind2 = ensure_edge_count(ind2, num_edges)
    return ind1, ind2

def custom_mutation(individual, indpb=0.2, num_edges=69):
    num_vertices = int(math.sqrt(len(individual)))
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]
    individual = ensure_edge_count(individual, num_edges)
    return individual,

