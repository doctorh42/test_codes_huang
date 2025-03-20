import osmnx as ox
import networkx as nx

# 指定城市
place_name = [
    "Chiyoda, Tokyo, Japan", "Chuo, Tokyo, Japan", "Minato, Tokyo, Japan",
    "Shinjuku, Tokyo, Japan", "Bunkyo, Tokyo, Japan", "Taito, Tokyo, Japan",
    "Sumida, Tokyo, Japan", "Koto, Tokyo, Japan", "Shinagawa, Tokyo, Japan",
    "Meguro, Tokyo, Japan", "Ota, Tokyo, Japan", "Setagaya, Tokyo, Japan",
    "Shibuya, Tokyo, Japan", "Nakano, Tokyo, Japan", "Suginami, Tokyo, Japan",
    "Toshima, Tokyo, Japan", "Kita, Tokyo, Japan", "Arakawa, Tokyo, Japan",
    "Itabashi, Tokyo, Japan", "Nerima, Tokyo, Japan", "Adachi, Tokyo, Japan",
    "Katsushika, Tokyo, Japan", "Edogawa, Tokyo, Japan", "Chiba, Japan"
]

# 获取行车道路（drive），可以改成 'walk'（步行）、'bike'（自行车）等
G = ox.graph_from_place(place_name, network_type="drive")

# 创建一个新图，只保留车道数 >= 2 的道路
G_filtered = G.copy()
for u, v, k, data in list(G.edges(keys=True, data=True)):
    lanes = data.get("lanes", "1")  # 获取 lanes，默认值设为 "1"
    
    if isinstance(lanes, list):  # 处理 lanes 可能是列表的情况
        lanes = lanes[0]
    
    try:
        lanes = int(lanes)  # 转换为整数
    except (ValueError, TypeError):
        lanes = 1  # 失败则设为1

    if lanes < 2:  # 过滤掉单车道
        G_filtered.remove_edge(u, v, key=k)


# 转换为无向图
G_undirected = G_filtered.to_undirected()
# 清除孤立节点和小型断裂道路网
largest_cc = max(nx.connected_components(G_undirected), key=len)
G_cleaned = G_undirected.subgraph(largest_cc).copy()
# 显示路网
ox.plot_graph(G_cleaned)

# 保存为 GraphML 格式（适用于 NetworkX 解析）
ox.save_graphml(G_cleaned, "Tokyo_23_wards_road_network.graphml")

# 保存为 GeoJSON（适用于 GIS 可视化）
ox.save_graph_geopackage(G_cleaned, "Tokyo_23_wards_road_network.gpkg")