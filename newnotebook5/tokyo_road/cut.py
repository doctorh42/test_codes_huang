import json
from shapely.geometry import shape, mapping
from shapely.ops import linemerge, unary_union, polygonize

# Step 1: 手动读取 C4RING.geojson
with open("C4RING.geojson", "r", encoding="utf-8") as f:
    ring_geojson = json.load(f)

ring_geoms = [shape(feat["geometry"]) for feat in ring_geojson["features"]]
merged = linemerge(unary_union(ring_geoms))
polygon = max(list(polygonize(merged)), key=lambda p: p.area)

# Step 2: 手动读取 tokyo.geojson
with open("tokyo.geojson", "r", encoding="utf-8") as f:
    tokyo_geojson = json.load(f)

# Step 3: 筛选在 polygon 内部的道路
filtered = []
for feat in tokyo_geojson["features"]:
    geom = shape(feat["geometry"])
    if geom.within(polygon):
        filtered.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": feat["properties"]
        })

# Step 4: 保存为新 GeoJSON
with open("tokyo_inside_c4.geojson", "w", encoding="utf-8") as f:
    json.dump({
        "type": "FeatureCollection",
        "features": filtered
    }, f, ensure_ascii=False)

print("✅ 已完成裁剪并输出 tokyo_inside_c4.geojson")
