import graph_tool.all as gt
import os
from pyproj import Transformer

city = "Tokyo"
G= gt.load_graph(f"{city}_road_network.graphml")

print(G)

# 选择 Japan Plane Rectangular CS Zone 9 (EPSG:6677)
transformer = Transformer.from_crs("epsg:4326", "epsg:6677", always_xy=True)

# 创建坐标属性
pos = G.new_vertex_property("vector<double>")
for v in G.vertices():
    lon = float(G.vertex_properties["x"][v])
    lat = float(G.vertex_properties["y"][v])
    x, y = transformer.transform(lon, lat)
    pos[v] = [x, -y]  # 反转 y 轴以符合地理标准

gt.graph_draw(
    G, 
    pos, 
    output_size=(1000, 1000),
    vertex_size=2,
    edge_pen_width=1,
    align=True
)