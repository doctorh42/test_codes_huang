
from graph_tool.all import *
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import numpy as np
import os

def load_net_to_graphtool(file_path):
    g = Graph(directed=False)
    v_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        reading_edges = False
        for line in lines:
            line = line.strip()
            if line.lower().startswith("*edges"):
                reading_edges = True
                continue
            if not reading_edges:
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                for node in [u, v]:
                    if node not in v_map:
                        v_map[node] = g.add_vertex()
                g.add_edge(v_map[u], v_map[v])
    return g

def analyze_graph(g, label=None):
    degs = g.get_total_degrees(g.get_vertices())
    print(f"📊 网络：{label}")
    print(f"- 节点数: {g.num_vertices()}")
    print(f"- 边数: {g.num_edges()}")
    print(f"- 平均度数: {np.mean(degs):.2f}")
    print(f"- 最大度数: {np.max(degs)}")
    print(f"- 最小度数: {np.min(degs)}")
    print("-" * 40)

    # 绘制度分布
    degree_count = np.bincount(degs)
    x = np.arange(len(degree_count))
    y = degree_count
    plt.plot(x, y, marker='o', label=label)

if __name__ == "__main__":
    print("请选择一个或多个 Pajek 格式 .net 文件")
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(filetypes=[("Pajek .net files", "*.net")])

    if not file_paths:
        print("未选择文件，程序结束。")
    else:
        for path in file_paths:
            g = load_net_to_graphtool(path)
            analyze_graph(g, label=os.path.basename(path))

        plt.xlabel("度数")
        plt.ylabel("节点数")
        plt.title("度分布图（Graph-tool 版本）")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
