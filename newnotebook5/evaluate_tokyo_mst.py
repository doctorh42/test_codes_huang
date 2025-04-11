from graph_tool.all import load_graph, Graph, graph_draw, shortest_path
import numpy as np
from collections import defaultdict
import pandas as pd
import os

# === 设置文件路径 ===
filename = "./MST_net/tokyo_pop_500_mst.net"
graph = load_graph(filename, fmt="pajek")

# === 获取节点坐标 ===
x = graph.vp.get("x")
y = graph.vp.get("y")
if x is not None and y is not None:
    pos = {v: (x[v], y[v]) for v in graph.vertices()}
else:
    raise ValueError("节点坐标未定义，请检查 .net 文件中是否包含 x/y 属性")

positions = np.array([pos[v] for v in graph.vertices()])
centroid = np.mean(positions, axis=0)
center_node = np.argmin([np.linalg.norm(p - centroid) for p in positions])

# === 获取叶节点 ===
def get_leaf_nodes(g):
    return [int(v) for v in g.vertices() if v.out_degree() == 1]

leaf_nodes = get_leaf_nodes(graph)

# === 分层函数 ===
def assign_layers(pos, leaf_nodes, center, num_layers):
    distances = {v: np.linalg.norm(pos[v] - center) for v in leaf_nodes}
    sorted_nodes = sorted(distances.items(), key=lambda x: x[1])
    layers = defaultdict(list)
    for idx, (node, dist) in enumerate(sorted_nodes):
        layer = int((idx / len(leaf_nodes)) * num_layers)
        layers[layer].append(node)
    return layers

# === 添加连接函数 ===
def add_layer_connections(base_graph, pos, layers, center_node, connect_ratio):
    new_edges = []
    for layer, nodes in layers.items():
        sorted_nodes = sorted(nodes, key=lambda v: np.arctan2(pos[v][1] - pos[center_node][1], pos[v][0] - pos[center_node][0]))
        for i in range(len(sorted_nodes)):
            u, v = sorted_nodes[i], sorted_nodes[(i + 1) % len(sorted_nodes)]
            new_edges.append((u, v))
        num_to_connect = max(1, int(connect_ratio * len(nodes)))
        chosen = np.random.choice(nodes, num_to_connect, replace=False)
        for node in chosen:
            new_edges.append((center_node, node))
    return new_edges

# === 拥堵与路径评估 ===
def evaluate_congestion(graph, edge_list):
    edge_usage = defaultdict(int)
    total_path_length = 0
    for v1 in graph.vertices():
        for v2 in graph.vertices():
            if int(v1) >= int(v2):
                continue
            path = shortest_path(graph, v1, v2)[1]
            total_path_length += len(path)
            for e in path:
                s, t = int(e.source()), int(e.target())
                edge = tuple(sorted((s, t)))
                edge_usage[edge] += 1
    congestion_cost = sum(edge_usage.values())  # 线性拥堵
    return total_path_length, congestion_cost

# === 实验参数 ===
layer_options = [2, 3, 4]
connect_ratios = [0.2, 0.5, 0.8]
results = []

os.makedirs("output", exist_ok=True)

for num_layers in layer_options:
    layers = assign_layers(pos, leaf_nodes, centroid, num_layers)
    for ratio in connect_ratios:
        new_g = Graph(graph)
        added_edges = add_layer_connections(new_g, pos, layers, center_node, ratio)
        for u, v in added_edges:
            new_g.add_edge(new_g.vertex(u), new_g.vertex(v))
        total_len, congestion = evaluate_congestion(new_g, added_edges)
        results.append((num_layers, ratio, total_len, congestion))

        # 可视化输出
        save_path = f"./output/tokyo_layer{num_layers}_ratio{int(ratio*100)}.png"
        graph_draw(new_g, pos=pos, output=save_path)

# === 结果保存 ===
df = pd.DataFrame(results, columns=["层数", "中心连接比例", "最短路径总和", "拥堵代价"])
df.to_csv("./output/tokyo_results.csv", index=False)
print("✅ 结果已输出至 ./output 文件夹中！")
