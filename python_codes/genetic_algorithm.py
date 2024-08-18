# python_codes/genetic_algorithm.py

import random
from graph_tool import Graph
from graph_tool.topology import min_spanning_tree, shortest_distance

def initialize_population(graph, pop_size):
    population = []
    for _ in range(pop_size):
        mst = min_spanning_tree(graph)
        mst_graph = Graph(directed=False)
        mst_graph.add_vertex(graph.num_vertices())
        for edge, in_mst in zip(graph.edges(), mst):
            if in_mst:
                mst_graph.add_edge(edge.source(), edge.target())
        # 复制顶点和边属性
        for prop_name, prop_map in graph.vertex_properties.items():
            mst_graph.vertex_properties[prop_name] = mst_graph.new_vertex_property(prop_map.value_type())
            for v in mst_graph.vertices():
                mst_graph.vertex_properties[prop_name][v] = graph.vertex_properties[prop_name][v]
        for prop_name, prop_map in graph.edge_properties.items():
            mst_graph.edge_properties[prop_name] = mst_graph.new_edge_property(prop_map.value_type())
            for e in mst_graph.edges():
                mst_graph.edge_properties[prop_name][e] = graph.edge_properties[prop_name][graph.edge(e.source(), e.target())]
        population.append(mst_graph)
    return population

# 适应度函数：计算总路径长度
def fitness(graph):
    total_length = 0
    for v in graph.vertices():
        dist_map = shortest_distance(graph, source=v)
        total_length += sum(dist_map.a)
    return total_length

# 选择父代
def tournament_selection(population, k):
    selected = random.sample(population, k)
    selected.sort(key=fitness)
    return selected[0]

# 交叉操作
def crossover(parent1, parent2):
    size = parent1.num_edges()
    edges1 = list(parent1.edges())
    edges2 = list(parent2.edges())
    child = Graph(directed=False)
    child.add_vertex(parent1.num_vertices())
    for i in range(size // 2):
        edge = edges1[i]
        child.add_edge(edge.source(), edge.target())
    for i in range(size // 2, size):
        edge = edges2[i]
        child.add_edge(edge.source(), edge.target())
    # 复制顶点和边属性
    for prop_name, prop_map in parent1.vertex_properties.items():
        child.vertex_properties[prop_name] = child.new_vertex_property(prop_map.value_type())
        for v in child.vertices():
            child.vertex_properties[prop_name][v] = parent1.vertex_properties[prop_name][v]
    for prop_name, prop_map in parent1.edge_properties.items():
        child.edge_properties[prop_name] = child.new_edge_property(prop_map.value_type())
        for e in child.edges():
            child.edge_properties[prop_name][e] = parent1.edge_properties[prop_name][parent1.edge(e.source(), e.target())]
    return child

# 变异操作
def mutate(graph, mutation_rate):
    if random.random() < mutation_rate:
        edges = list(graph.edges())
        graph.remove_edge(random.choice(edges))
        available_edges = list(read_graph.edges())
        new_edge = random.choice(available_edges)
        graph.add_edge(new_edge.source(), new_edge.target())

# 主遗传算法
def genetic_algorithm(graph, pop_size, num_generations, mutation_rate, tournament_size):
    population = initialize_population(graph, pop_size)
    best_individual = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

        # 评估新种群
        for individual in population:
            ind_fitness = fitness(individual)
            if ind_fitness < best_fitness:
                best_fitness = ind_fitness
                best_individual = individual

        print(f"Generation {generation}: Best fitness = {best_fitness}")

    return best_individual
