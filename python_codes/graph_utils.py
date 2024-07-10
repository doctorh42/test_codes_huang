from graph_tool.all import Graph, shortest_distance, random_spanning_tree
import random
from deap import base, creator, tools, algorithms


def add_random_edges(graph, num_edges, existing_edges, edge_color_map=None, seed=None):
    if seed is not None:
        random.seed(seed)
    vertices = list(graph.vertices())
    added_edges = 0
    while added_edges < num_edges:
        v1, v2 = random.sample(vertices, 2)
        edge = (min(v1, v2), max(v1, v2))
        if v1 != v2 and edge not in existing_edges:
            e = graph.add_edge(v1, v2)
            existing_edges.add(edge)
            if edge_color_map:
                edge_color_map[e] = [1, 0, 0, 1]  # 红色表示随机边
            added_edges += 1


def calculate_total_shortest_path_sum(graph):
    total_sum = 0
    for v in graph.vertices():
        distances = shortest_distance(graph, source=v)
        total_sum += sum(distances.a)
    return total_sum


def create_mst(graph, pos):
    mst = Graph(directed=False)
    mst.add_vertex(graph.num_vertices())
    mst.vertex_properties["number"] = mst.new_vertex_property("int")
    mst_pos = mst.new_vertex_property("vector<double>")
    edge_color_map = mst.new_edge_property("vector<double>")

    mst_edge_map = random_spanning_tree(graph)

    for v in graph.vertices():
        mst.vertex_properties["number"][mst.vertex(v)] = graph.vertex_properties["number"][v]
        mst_pos[mst.vertex(v)] = pos[v]

    existing_edges = set()
    for e in graph.edges():
        if mst_edge_map[e]:
            edge = mst.add_edge(e.source(), e.target())
            existing_edges.add((min(int(e.source()), int(e.target())), max(int(e.source()), int(e.target()))))
            edge_color_map[edge] = [0, 0, 1, 1]  # 蓝色表示MST边

    return mst, mst_pos, edge_color_map, existing_edges


def evalGraph(individual, existing_edges, seed, read_graph):
    graph = Graph(directed=False)
    graph.add_vertex(read_graph.num_vertices())
    local_existing_edges = set(existing_edges)
    random.seed(seed)  # 确保使用相同的随机种子
    for source, target in individual:
        graph.add_edge(source, target)
        local_existing_edges.add((min(source, target), max(source, target)))
    add_random_edges(graph, 20, local_existing_edges, seed=seed)  # 确保添加20条随机边
    return calculate_total_shortest_path_sum(graph),


def setup_ga_toolbox(edge_list, tournsize=3):
    if "FitnessMin" in dir(creator):
        del creator.FitnessMin
    if "Individual" in dir(creator):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_edge", random.sample, edge_list, len(edge_list))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_edge)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", cxRandomEdge)
    toolbox.register("mutate", mutRandomEdge, indpb=0.5, edge_list=edge_list)  # 增大变异概率
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    return toolbox


def cxRandomEdge(ind1, ind2):
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


def mutRandomEdge(individual, indpb, edge_list):
    if random.random() < indpb:
        edge = random.choice(edge_list)
        if edge not in individual:
            individual[random.randint(0, len(individual) - 1)] = edge
    return individual,


def run_genetic_algorithm(toolbox, population, existing_edges, read_graph, NGEN, CXPB, MUTPB):
    best_ind = None
    best_fitness = float('inf')
    best_seed = None

    for gen in range(NGEN):
        seeds = [random.randint(0, 10000) for _ in range(len(population))]
        fits = [evalGraph(ind, existing_edges, seed, read_graph)[0] for ind, seed in zip(population, seeds)]

        for fit, ind, seed in zip(fits, population, seeds):
            ind.fitness.values = (fit,)
            if fit < best_fitness:
                best_fitness = fit
                best_ind = ind
                best_seed = seed

        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=CXPB, mutpb=MUTPB)
        population[:] = offspring

        # 保留最优解
        if best_ind is not None:
            population[0] = toolbox.clone(best_ind)

        print(f"Generation {gen}: Best fitness = {best_fitness}")
        if gen % 10 == 0:
            print(f"Current population fitness: {fits}")

    return best_ind, best_fitness, best_seed


def create_optimized_graph(mst, best_ind, best_seed, existing_edges, read_graph):
    optimized_graph = Graph(directed=False)
    optimized_graph.add_vertex(read_graph.num_vertices())
    optimized_graph.vertex_properties["number"] = optimized_graph.new_vertex_property("int")
    for v in read_graph.vertices():
        optimized_graph.vertex_properties["number"][optimized_graph.vertex(v)] = read_graph.vertex_properties["number"][
            v]

    for e in mst.edges():
        optimized_graph.add_edge(e.source(), e.target())

    best_ind_edges = set((min(source, target), max(source, target)) for source, target in best_ind)
    for source, target in best_ind:
        edge = (min(source, target), max(source, target))
        if edge not in existing_edges:
            optimized_graph.add_edge(source, target)

    return optimized_graph


def create_initial_population(toolbox, population_size, edge_list, mst_edges, existing_edges):
    population = []
    for _ in range(population_size):
        ind = toolbox.individual()
        additional_edges = set()
        while len(additional_edges) < 20:
            edge = random.choice(edge_list)
            if edge not in mst_edges and edge not in existing_edges and edge not in additional_edges:
                additional_edges.add(edge)
        for edge in additional_edges:
            ind.append(edge)
        population.append(ind)
    return population
