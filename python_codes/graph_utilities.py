import numpy as np
import random
from graph_tool.all import Graph, shortest_distance
from deap import creator

# 将图转换为个体编码　グラフを個体エンコードに変換する
def graph_to_individual(graph, num_vertices):
    individual = [0] * (num_vertices * num_vertices)
    for edge in graph.edges():
        source, target = int(edge.source()), int(edge.target())
        individual[source * num_vertices + target] = 1
        individual[target * num_vertices + source] = 1
    return individual

# 将个体解码为图并计算边权重属性　個体をグラフにデコードし、エッジの重みを計算する
def individual_to_graph(individual, num_vertices, read_pos):
    graph = Graph(directed=False)
    graph.add_vertex(num_vertices)
    new_edge_weights = graph.new_edge_property("double")
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if individual[i * num_vertices + j] == 1:
                edge = graph.add_edge(graph.vertex(i), graph.vertex(j))
                source_pos = read_pos[graph.vertex(i)]
                target_pos = read_pos[graph.vertex(j)]
                euclidean_distance = np.sqrt(
                    (source_pos[0] - target_pos[0]) ** 2 + (source_pos[1] - target_pos[1]) ** 2)
                new_edge_weights[edge] = euclidean_distance
    return graph, new_edge_weights

# 计算两条线段是否相交的函数
def is_intersect(p1, p2, q1, q2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

# 定义适应度函数　適応度関数を定義する
def evaluate(individual, num_vertices, read_pos, read_graph):
    graph, new_edge_weights = individual_to_graph(individual, num_vertices, read_pos)
    num_edges = graph.num_edges()
    if num_edges != read_graph.num_edges():  # 动态边数限制，确保边数与原图相同
        return float('inf'),  # 如果边数不相同，适应度设为无穷大

    dist_matrix = shortest_distance(graph, weights=new_edge_weights).get_2d_array(range(graph.num_vertices()))
    total_distance = np.sum(dist_matrix[dist_matrix != np.inf])

    # 计算边交叉的惩罚项
    edge_list = [(e.source(), e.target()) for e in graph.edges()]
    cross_count = 0
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            e1, e2 = edge_list[i], edge_list[j]
            p1, p2 = read_pos[e1[0]], read_pos[e1[1]]
            q1, q2 = read_pos[e2[0]], read_pos[e2[1]]
            if is_intersect(p1, p2, q1, q2):
                cross_count += 1

    # 将交叉的数量添加到总距离中作为惩罚
    total_distance += cross_count * 1000  # 乘以一个大数以增加惩罚力度

    return total_distance,

# 自定义交叉操作　交叉操作
def cxGraph(ind1, ind2, num_vertices, read_graph):
    size = len(ind1)
    point = random.randint(1, size - 1)
    new_ind1 = creator.Individual(np.concatenate((ind1[:point], ind2[point:])))
    new_ind2 = creator.Individual(np.concatenate((ind2[:point], ind1[point:])))

    # 修正边数
    def fix_edges(individual):
        edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices) if
                 individual[i * num_vertices + j] == 1]
        non_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices) if
                     individual[i * num_vertices + j] == 0]
        if len(edges) > read_graph.num_edges():
            # 移除多余的边
            extra_edges = random.sample(edges, len(edges) - read_graph.num_edges())
            for i, j in extra_edges:
                individual[i * num_vertices + j] = 0
                individual[j * num_vertices + i] = 0
        elif len(edges) < read_graph.num_edges():
            # 添加缺失的边
            missing_edges = random.sample(non_edges, read_graph.num_edges() - len(edges))
            for i, j in missing_edges:
                individual[i * num_vertices + j] = 1
                individual[j * num_vertices + i] = 1
        return individual

    new_ind1 = fix_edges(new_ind1)
    new_ind2 = fix_edges(new_ind2)

    return new_ind1, new_ind2

# 自定义变异操作　変異操作
def mutGraph(ind, num_vertices):
    edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices) if ind[i * num_vertices + j] == 1]
    non_edges = [(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices) if ind[i * num_vertices + j] == 0]

    if edges and non_edges:
        # 移除一条边　エッジを削除する
        i, j = random.choice(edges)
        ind[i * num_vertices + j] = 0
        ind[j * num_vertices + i] = 0

        # 添加一条边　エッジを追加する
        i, j = random.choice(non_edges)
        ind[i * num_vertices + j] = 1
        ind[j * num_vertices + i] = 1

    return ind,

# 初始化个体时确保边数与原图相同　初期個体を初期化する際にエッジ数を原図と同じに確保する
def initIndividual(read_graph):
    num_vertices = read_graph.num_vertices()
    individual = [0] * (num_vertices * num_vertices)
    edges = random.sample([(i, j) for i in range(num_vertices) for j in range(i + 1, num_vertices)], read_graph.num_edges())
    for i, j in edges:
        individual[i * num_vertices + j] = 1
        individual[j * num_vertices + i] = 1
    return creator.Individual(individual)
