import geopandas as gpd
from shapely.ops import linemerge, unary_union, polygonize

# === Step 1: 读取 C4RING 并转换为 Polygon ===
c4_gdf = gpd.read_file("C4RING.geojson")

# 如果是 LineString，需要 polygonize
if c4_gdf.geometry.iloc[0].geom_type == "LineString":
    merged = linemerge(unary_union(c4_gdf.geometry))
    polygon = max(list(polygonize(merged)), key=lambda p: p.area)
else:
    polygon = c4_gdf.geometry.unary_union

# === Step 2: 读取完整东京路网 ===
tokyo_gdf = gpd.read_file("tokyo.geojson")

# === Step 3: 仅保留位于 C4 Polygon 内的道路 ===
tokyo_inside = tokyo_gdf[tokyo_gdf.geometry.within(polygon)]

# === Step 4: 保存裁剪后的路网 ===
tokyo_inside.to_file("tokyo_inside_c4.geojson", driver="GeoJSON")
print("✅ 成功输出裁剪后的东京路网：tokyo_inside_c4.geojson")
