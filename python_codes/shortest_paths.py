from graph_tool.all import shortest_distance
import csv


def calculate_shortest_paths(graph, node_indices, progress_bar=None):
    nodes = list(graph.vertices())
    distances = {}
    total_distance_sum = 0
    num_pairs = 0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):  # 确保每对节点只计算一次
            distance = shortest_distance(graph, source=nodes[i], target=nodes[j])
            distances[(node_indices[i], node_indices[j])] = distance
            distances[(node_indices[j], node_indices[i])] = distance  # because the graph is undirected
            total_distance_sum += distance
            num_pairs += 1
            if progress_bar:
                progress_bar.update(1)  # 更新进度条

    overall_average_distance = total_distance_sum / num_pairs if num_pairs > 0 else 0
    return distances, total_distance_sum, overall_average_distance


def save_to_csv(node_indices, distances, total_distance_sum, overall_average_distance, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        header = ["Node"] + [str(node) for node in node_indices] + ["Total Distance", "Average Distance"]
        writer.writerow(header)

        # 写入每个节点的距离
        for i in range(len(node_indices)):
            row = [str(node_indices[i])] + [
                distances[(node_indices[i], node_indices[j])] if node_indices[i] != node_indices[j] else 0 for j in
                range(len(node_indices))]
            total_distance = sum(distances[(node_indices[i], node_indices[j])] for j in range(len(node_indices)) if
                                 node_indices[i] != node_indices[j])
            average_distance = total_distance / (len(node_indices) - 1) if len(node_indices) > 1 else 0
            row += [total_distance, average_distance]
            writer.writerow(row)

        # 写入整体网络的总距离和平均距离
        writer.writerow([])
        writer.writerow(["Overall Total Distance", total_distance_sum])
        writer.writerow(["Overall Average Distance", overall_average_distance])

    print(f"Shortest paths have been saved to {filename}")
    print(f"Overall Total Distance: {total_distance_sum}")
    print(f"Overall Average Distance: {overall_average_distance}")
