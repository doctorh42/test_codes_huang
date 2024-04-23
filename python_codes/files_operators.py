from graph_tool import Graph


def save_files(filename, graph_obj, graph_pos, position_flag=False, jinkou_flag=False, community=False,
               community_labels=None):
    """
    Save graph data to a file with optional community information.

    :param filename: Desired name with file path, including the file extension.
    :param graph_obj: Graph-tool graph object.
    :param graph_pos: Graph-tool graph position object.
    :param position_flag: If True, save nodes' position.
    :param jinkou_flag: If True, save nodes' population value.
    :param community: If True, save nodes' community information.
    :param community_labels: Graph-tool VertexPropertyMap with community labels.
    :return: None
    """
    # Adjust column templates based on flags
    column_count = 2  # Always have at least vertex ID and number
    if position_flag:
        column_count += 2  # Add columns for x and y positions
    if jinkou_flag:
        column_count += 1  # Add column for population
    if community:
        column_count += 1  # Add column for community

    # Prepare to write to file
    vertices_count = graph_obj.num_vertices()
    with open(filename, 'w') as f:
        f.write(f"*Vertices {vertices_count}\n")
        for vertex in graph_obj.vertices():
            temp_v = [graph_obj.vertex_index[vertex] + 1,
                      int(graph_obj.vertex_properties['number'][graph_obj.vertex_index[vertex]] + 1)]
            if position_flag:
                vertex_pos = graph_pos[graph_obj.vertex(vertex)]
                temp_v.extend([vertex_pos[0], vertex_pos[1]])
            if jinkou_flag:
                temp_v.append(int(graph_obj.vertex_properties['population'][vertex]))
            if community:
                temp_v.append(int(community_labels[vertex]))
            temp_str = ' '.join(str(x) for x in temp_v)
            f.write(temp_str + '\n')
        f.write("*Edges\n")
        for s, t in graph_obj.iter_edges():
            f.write(f"{s + 1} {t + 1}\n")
        f.close()

def read_files(filename, jinkou_flag=False, community_flag=False):
    """
    :param jinkou_flag: Only for files that have population column
    :param community_flag: Set to True if the file includes community information
    :param filename: the file you want to open, with file path.
    :return: graph-tool's graph object, pos vector object, and optionally community labels
    """
    with open(filename, 'r') as f:
        raw_network_lines = f.readlines()

    raw_graph = Graph(directed=False)
    raw_graph.vertex_properties['number'] = raw_graph.new_vertex_property('int')
    if jinkou_flag:
        raw_graph.vertex_properties['population'] = raw_graph.new_vertex_property('int')
    if community_flag:
        raw_graph.vertex_properties['community'] = raw_graph.new_vertex_property('int')

    num_vertices = int(raw_network_lines[0].split()[1])
    raw_graph.add_vertex(num_vertices)

    edges_start_line = raw_network_lines.index('*Edges\n')
    raw_pos = raw_graph.new_vertex_property('vector<double>')
    for line in raw_network_lines[1:edges_start_line]:
        parts = line.split()
        vertex_id = int(parts[0]) - 1  # Convert 1-based index to 0-based
        vertex_number = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        if jinkou_flag:
            population = int(parts[4])
            community = int(parts[5]) if community_flag else None
        else:
            community = int(parts[4]) if community_flag else None

        raw_pos[vertex_id] = (x, y)
        raw_graph.vertex_properties['number'][raw_graph.vertex(vertex_id)] = vertex_number
        if jinkou_flag:
            raw_graph.vertex_properties['population'][raw_graph.vertex(vertex_id)] = population
        if community_flag:
            raw_graph.vertex_properties['community'][raw_graph.vertex(vertex_id)] = community

    for line in raw_network_lines[edges_start_line + 1:]:
        source, target = map(int, line.split())
        source_index = source - 1  # Convert 1-based index to 0-based
        target_index = target - 1
        raw_graph.add_edge(source_index, target_index)

    if community_flag:
        return raw_graph, raw_pos, raw_graph.vertex_properties['community']
    else:
        return raw_graph, raw_pos

print("2")
