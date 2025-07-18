# prebuild_graph.py

import osmnx as ox

# — 1) Define your region by name, not by bounding-box floats
place_name = "Northern Ireland, United Kingdom"

# — 2) Download only the major roads in that place
G = ox.graph_from_place(
    place_name,
    network_type="drive",
    custom_filter='["highway"~"motorway|trunk|primary"]'
)

# — 3) Save for instant reuse
ox.save_graphml(G, "ni_major_roads.graphml")
print("✅ Saved ni_major_roads.graphml for Northern Ireland")
