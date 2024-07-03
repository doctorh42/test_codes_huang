from graph_tool.all import Graph, shortest_distance, random_spanning_tree
import random
from deap import base, creator, tools, algorithms

def add_random_edges(graph, num_edges, edge_list, edge_color_map=None, seed=None):
    if seed is not None:
        random.seed(seed)
    vertices = list(graph.vertices())
    added_edges = 0
    while added_edges < num_edges:
        v1, v2 = random.sample(vertices, 2)
        edge = (min(v1, v2), max(v1, v2))
        if v1 != v2 and (edge not in edge_list):
            e = graph.add_edge(v1, v2)
            edge_list.add(edge)
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

def evalGraph(individual, read_graph):
    graph = Graph(directed=False)
    graph.add_vertex(read_graph.num_vertices())

    for source, target in individual:
        graph.add_edge(source, target)

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
    toolbox.register("mutate", mutRandomEdge, indpb=0.5, edge_list=edge_list)
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

def run_genetic_algorithm(toolbox, population, read_graph, NGEN, CXPB, MUTPB):
    best_ind = None
    best_fitness = float('inf')

    for gen in range(NGEN):
        fits = [evalGraph(ind, read_graph)[0] for ind in population]

        for fit, ind in zip(fits, population):
            ind.fitness.values = (fit,)
            if fit < best_fitness:
                best_fitness = fit
                best_ind = ind

        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=CXPB, mutpb=MUTPB)
        population[:] = offspring

        # 保留最优解
        if best_ind is not None:
            population[0] = toolbox.clone(best_ind)

        print(f"Generation {gen}: Best fitness = {best_fitness}")
        if gen % 10 == 0:
            print(f"Current population fitness: {fits}")

    return best_ind, best_fitness

def create_optimized_graph(read_graph, best_ind):
    optimized_graph = Graph(directed=False)
    optimized_graph.add_vertex(read_graph.num_vertices())
    optimized_graph.vertex_properties["number"] = optimized_graph.new_vertex_property("int")
    for v in read_graph.vertices():
        optimized_graph.vertex_properties["number"][optimized_graph.vertex(v)] = read_graph.vertex_properties["number"][v]

    for source, target in best_ind:
        optimized_graph.add_edge(source, target)

    return optimized_graph

def create_initial_population(toolbox, population_size, edge_list):
    population = []
    for _ in range(population_size):
        ind = []
        while len(ind) < 69:
            edge = random.choice(edge_list)
            if edge not in ind:
                ind.append(edge)
        ind = creator.Individual(ind)
        population.append(ind)
    return population
