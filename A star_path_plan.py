import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, box
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import heapq

def load_shapefile(file_path):
    print(f"Loading shapefile from {file_path}...")
    return gpd.read_file(file_path)

class AStar:
    def __init__(self, obstacles, resolution, robot_radius, show_animation=False):
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None
        self.show_animation = show_animation

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.obstacles = obstacles
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, g_cost, h_cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.g_cost = g_cost
            self.h_cost = h_cost
            self.f_cost = g_cost + h_cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return f"{self.x},{self.y},{self.g_cost},{self.h_cost},{self.f_cost},{self.parent_index}"

        def __lt__(self, other):
            return self.f_cost < other.f_cost

    def planning(self, sx, sy, gx, gy, boundary_polygon):
        print(f"Planning path from ({sx}, {sy}) to ({gx}, {gy}) using A* algorithm...")
        self.min_x, self.min_y, self.max_x, self.max_y = boundary_polygon.bounds
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, self.calc_heuristic(sx, sy, gx, gy), -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, 0.0, -1)

        open_set = []
        closed_set = dict()
        open_set_dict = {}
        heapq.heappush(open_set, start_node)
        open_set_dict[self.calc_index(start_node)] = start_node

        start_time = time.time()
        self.calc_obstacle_map(boundary_polygon)
        print(f"Time taken for calc_obstacle_map: {time.time() - start_time:.2f} seconds")

        clipped_obstacles = gpd.clip(self.obstacles, boundary_polygon)

        if self.show_animation:
            plt.figure()
            plt.plot(sx, sy, "og", label="Start")
            plt.plot(gx, gy, "xb", label="Goal")
            obstacles_plotted = False
            for geom in clipped_obstacles.geometry:
                if geom.geom_type == 'Polygon' or geom.geom_type == 'MultiPolygon':
                    x, y = geom.exterior.xy
                    if not obstacles_plotted:
                        plt.plot(x, y, "-k", label="Obstacle")
                        obstacles_plotted = True
                    else:
                        plt.plot(x, y, "-k")
                elif geom.geom_type == 'LineString':
                    x, y = geom.xy
                    if not obstacles_plotted:
                        plt.plot(x, y, "-k", label="Obstacle")
                        obstacles_plotted = True
                    else:
                        plt.plot(x, y, "-k")
            plt.xlim(boundary_polygon.bounds[0], boundary_polygon.bounds[2])
            plt.ylim(boundary_polygon.bounds[1], boundary_polygon.bounds[3])
            plt.grid(True)
            plt.axis("equal")
            # 畫出限制範圍的邊框
            boundary_x, boundary_y = boundary_polygon.exterior.xy
            plt.plot(boundary_x, boundary_y, "--r", label="Boundary")
            plt.legend()

        while open_set:
            current = heapq.heappop(open_set)
            del open_set_dict[self.calc_index(current)]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Goal found!")
                goal_node.parent_index = current.parent_index
                goal_node.g_cost = current.g_cost
                break

            closed_set[self.calc_index(current)] = current

            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.g_cost + move_cost,
                                 self.calc_heuristic(self.calc_position(current.x + move_x, self.min_x),
                                                     self.calc_position(current.y + move_y, self.min_y),
                                                     gx, gy),
                                 self.calc_index(current))
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set_dict:
                    heapq.heappush(open_set, node)
                    open_set_dict[n_id] = node
                else:
                    if open_set_dict[n_id].f_cost > node.f_cost:
                        open_set_dict[n_id] = node
                        heapq.heapify(open_set)

            if self.show_animation:
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), "xc")
                plt.pause(0.01)

        rx, ry = self.calc_final_path(goal_node, closed_set)
        if self.show_animation:
            plt.plot(rx, ry, "-r", label="Path")
            plt.xlim(boundary_polygon.bounds[0], boundary_polygon.bounds[2])
            plt.ylim(boundary_polygon.bounds[1], boundary_polygon.bounds[3])
            # 再次畫出限制範圍的邊框
            plt.plot(boundary_x, boundary_y, "--r")
            plt.legend()
            plt.pause(0.01)
            plt.show()
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    def calc_position(self, index, minp):
        return index * self.resolution + minp

    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    def calc_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False

        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, boundary_polygon):
        start_time = time.time()
        print("Calculating obstacle map...")
        self.min_x, self.min_y, self.max_x, self.max_y = boundary_polygon.bounds

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        clipped_obstacles = gpd.clip(self.obstacles, boundary_polygon)

        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]

        for geom in clipped_obstacles.geometry:
            if geom.geom_type == 'Polygon' or geom.geom_type == 'MultiPolygon':
                buffered = geom.buffer(self.robot_radius)
                minx, miny, maxx, maxy = buffered.bounds
                minx = max(self.min_x, minx)
                miny = max(self.min_y, miny)
                maxx = min(self.max_x, maxx)
                maxy = min(self.max_y, maxy)
                for x in range(self.calc_xy_index(minx, self.min_x), self.calc_xy_index(maxx, self.min_x)):
                    for y in range(self.calc_xy_index(miny, self.min_y), self.calc_xy_index(maxy, self.min_y)):
                        if (0 <= x < self.x_width) and (0 <= y < self.y_width):
                            px = self.calc_position(x, self.min_x)
                            py = self.calc_position(y, self.min_y)
                            if buffered.contains(Point(px, py)):
                                self.obstacle_map[x][y] = True
            elif geom.geom_type == 'LineString':
                buffered = geom.buffer(self.robot_radius)
                minx, miny, maxx, maxy = buffered.bounds
                minx = max(self.min_x, minx)
                miny = max(self.min_y, miny)
                maxx = min(self.max_x, maxx)
                maxy = min(self.max_y, maxy)
                for x in range(self.calc_xy_index(minx, self.min_x), self.calc_xy_index(maxx, self.min_x)):
                    for y in range(self.calc_xy_index(miny, self.min_y), self.calc_xy_index(maxy, self.min_y)):
                        if (0 <= x < self.x_width) and (0 <= y < self.y_width):
                            px = self.calc_position(x, self.min_x)
                            py = self.calc_position(y, self.min_y)
                            if buffered.contains(Point(px, py)):
                                self.obstacle_map[x][y] = True

        print(f"Time taken for map calculation: {time.time() - start_time:.2f} seconds")

    @staticmethod
    def get_motion_model():
        return [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
            [2, 0, 2],
            [0, 2, 2],
            [-2, 0, 2],
            [0, -2, 2],
            [2, 1, math.sqrt(5)],
            [2, -1, math.sqrt(5)],
            [-2, 1, math.sqrt(5)],
            [-2, -1, math.sqrt(5)],
            [1, 2, math.sqrt(5)],
            [1, -2, math.sqrt(5)],
            [-1, 2, math.sqrt(5)],
            [-1, -2, math.sqrt(5)]
        ]

    @staticmethod
    def calc_heuristic(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

def save_paths_to_shp(paths, crs, output_file):
    print(f"Saving paths to {output_file}...")
    data = []
    for path, start_name, end_name, distance in paths:
        data.append({
            'start_name': start_name,
            'end_name': end_name,
            'distance': distance,
            'geometry': LineString(path)
        })
    gdf = gpd.GeoDataFrame(data, crs=crs)
    gdf.to_file(output_file, driver='ESRI Shapefile')
    print("Save completed.")

def plot_path(graph, paths, obstacles, start_points, end_points, boundary_polygon):
    print("Plotting paths...")
    fig, ax = plt.subplots()
    pos = {point: (point.x, point.y) for path in paths for point in path}
    nx.draw(graph, pos, node_size=10, with_labels=False, ax=ax, edge_color='blue', alpha=0.5)
    for path in paths:
        path_line = LineString(path)
        ax.plot(*path_line.xy, color='red', linewidth=2, label='Shortest Path')
    # 僅顯示切割後的障礙物
    obstacles = gpd.clip(obstacles, boundary_polygon)
    obstacles.plot(ax=ax, facecolor='none', edgecolor='black', label='Obstacles')
    start_points.plot(ax=ax, color='green', marker='o', label='Start Points')
    end_points.plot(ax=ax, color='blue', marker='x', label='End Points')
    boundary_x, boundary_y = boundary_polygon.exterior.xy
    ax.plot(boundary_x, boundary_y, "--r", label="Boundary")  # 顯示邊界
    ax.set_xlim(boundary_polygon.bounds[0], boundary_polygon.bounds[2])
    ax.set_ylim(boundary_polygon.bounds[1], boundary_polygon.bounds[3])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()
    print("Plotting completed.")

def extract_obstacle_lines_and_polygons(obstacles):
    return obstacles

def create_boundary_polygon(sx, sy, gx, gy, buffer_distance=10):
    min_x = min(sx, gx) - buffer_distance
    min_y = min(sy, gy) - buffer_distance
    max_x = max(sx, gx) + buffer_distance
    max_y = max(sy, gy) + buffer_distance
    return box(min_x, min_y, max_x, max_y)

def main():
    print("Starting main process...")
    start_points = load_shapefile('start_points.shp')
    end_points = load_shapefile('end_points.shp')
    obstacles = load_shapefile('obstacles.shp')
    
    crs = start_points.crs
    resolution = 1.0  # grid resolution
    robot_radius = 1.0  # robot radius
    show_animation = True

    astar = AStar(obstacles, resolution, robot_radius, show_animation=show_animation)
    
    paths = []
    print("Finding paths...")
    for idx, start in start_points.iterrows():
        nearby_end_points = end_points[end_points.geometry.distance(start.geometry) <= 500]
        if nearby_end_points.empty:
            continue

        # 移除起始點或終點在障礙物內的障礙物
        obstacles_filtered = obstacles.copy()
        to_remove = obstacles_filtered.geometry.apply(lambda geom: geom.contains(start.geometry) or any(geom.contains(end) for end in nearby_end_points.geometry))
        obstacles_filtered = obstacles_filtered[~to_remove]
        
        astar.obstacles = obstacles_filtered
        for jdx, end in nearby_end_points.iterrows():
            boundary_polygon = create_boundary_polygon(start.geometry.x, start.geometry.y, end.geometry.x, end.geometry.y)
            print(f"Finding path from {start['Name']} to {end['NAME']}...")
            start_time = time.time()
            rx, ry = astar.planning(start.geometry.x, start.geometry.y, end.geometry.x, end.geometry.y, boundary_polygon)
            print(f"Time taken for planning: {time.time() - start_time:.2f} seconds")
            if rx is not None and ry is not None:
                path = [Point(x, y) for x, y in zip(rx, ry)]
                distance = len(path) * resolution
                paths.append((path, start['Name'], end['NAME'], distance))

    if paths:
        save_paths_to_shp(paths, crs, "shortest_paths.shp")
        graph = nx.Graph()
        for path, _, _, _ in paths:
            for i in range(len(path) - 1):
                graph.add_edge(path[i], path[i + 1], weight=resolution)
        plot_path(graph, [p[0] for p in paths], obstacles, start_points, end_points, boundary_polygon)
    print("Process completed.")

if __name__ == "__main__":
    main()