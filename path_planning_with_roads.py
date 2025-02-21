import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, box
import matplotlib.pyplot as plt
import math
import heapq
from scipy.spatial import KDTree  # 用於加速最近鄰搜尋

# ----------------------------
# 安全幾何判斷函式，遇到錯誤時先用 buffer(0) 修正
# ----------------------------
def safe_contains(g, geom, tol=1e-6):
    try:
        return g.contains(geom)
    except Exception:
        try:
            return g.buffer(0).contains(geom)
        except Exception:
            return False

# ----------------------------
# 工具函式
# ----------------------------
def remove_consecutive_duplicates(path, tol=1e-6):
    if not path:
        return []
    new_path = [path[0]]
    for pt in path[1:]:
        if math.hypot(pt[0] - new_path[-1][0], pt[1] - new_path[-1][1]) >= tol:
            new_path.append(pt)
    return new_path

def merge_segments_all(seg1, seg2, seg3, tol=1e-6):
    merged = seg1.copy()
    if seg2:
        if merged and seg2 and Point(*merged[-1]).distance(Point(*seg2[0])) < tol:
            merged.pop()
        merged.extend(seg2)
    if seg3:
        if merged and seg3 and Point(*merged[-1]).distance(Point(*seg3[0])) < tol:
            seg3 = seg3[1:]
        merged.extend(seg3)
    return remove_consecutive_duplicates(merged, tol)

def remove_overlap(road_segment, end_segment, tol=1e-6):
    if not road_segment or not end_segment:
        return road_segment
    if math.hypot(road_segment[-1][0] - end_segment[0][0],
                  road_segment[-1][1] - end_segment[0][1]) < tol:
        return road_segment[:-1]
    return road_segment

def remove_loops(path, tol=1e-6):
    # 利用四捨五入後的 tuple 作為 key 判斷是否重複（容差以 tol 為單位）
    new_path = []
    seen = {}
    for pt in path:
        key = (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)
        if key in seen:
            new_path = new_path[:seen[key] + 1]
            seen = { (round(p[0] / tol) * tol, round(p[1] / tol) * tol): idx for idx, p in enumerate(new_path)}
        else:
            seen[key] = len(new_path)
            new_path.append(pt)
    return new_path

def load_shapefile(file_path):
    print(f"Loading shapefile from {file_path}...")
    return gpd.read_file(file_path)

def points_equal(p1, p2, tol=1e-6):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < tol

def snap_to_graph(point, graph):
    # 利用 KDTree 加速搜尋
    nodes = list(graph.nodes)
    if not nodes:
        return None
    coords = [(n.x, n.y) for n in nodes]
    tree = KDTree(coords)
    dist, idx = tree.query((point.x, point.y))
    return nodes[idx]

def is_on_road(p, astar, tol):
    for road in astar.roads.geometry:
        if p.distance(road) <= tol:
            return True
    return False

def find_road_projection(astar, point):
    min_dist = float('inf')
    best_proj = None
    for road in astar.roads.geometry:
        if road.geom_type == 'LineString':
            proj = road.project(point)
            candidate = road.interpolate(proj)
            d = point.distance(candidate)
            if d < min_dist:
                min_dist = d
                best_proj = candidate
        elif road.geom_type == 'MultiLineString':
            for sub in road.geoms:
                proj = sub.project(point)
                candidate = sub.interpolate(proj)
                d = point.distance(candidate)
                if d < min_dist:
                    min_dist = d
                    best_proj = candidate
    return best_proj

def split_edge_at_node(graph, new_node, tol=1e-6):
    for u, v, data in list(graph.edges(data=True)):
        line = LineString([u, v])
        if new_node.distance(line) < tol:
            uv = (v.x - u.x, v.y - u.y)
            un = (new_node.x - u.x, new_node.y - u.y)
            dot = uv[0] * un[0] + uv[1] * un[1]
            len_sq = uv[0] ** 2 + uv[1] ** 2
            if len_sq == 0:
                continue
            t = dot / len_sq
            if t > tol and t < 1 - tol:
                graph.remove_edge(u, v)
                if new_node not in graph:
                    graph.add_node(new_node)
                graph.add_edge(u, new_node, weight=u.distance(new_node))
                graph.add_edge(new_node, v, weight=new_node.distance(v))
                return True
    return False

def find_node(graph, point, tol=1e-6):
    # 在 graph 中搜尋與 point 在容差內相同的節點
    for n in graph.nodes:
        if points_equal((n.x, n.y), (point.x, point.y), tol):
            return n
    return None

def get_candidate_road_nodes(point, roads, tol=1e-6):
    """
    將指定點投影到所有道路上，取得候選連接節點，並去除重複候選。
    """
    candidates = []
    for road in roads.geometry:
        if road.geom_type == 'LineString':
            proj = road.project(point)
            candidate = road.interpolate(proj)
            candidates.append(candidate)
        elif road.geom_type == 'MultiLineString':
            for sub in road.geoms:
                proj = sub.project(point)
                candidate = sub.interpolate(proj)
                candidates.append(candidate)
    # 去除重複（以 tol 為容差）
    unique = []
    for cand in candidates:
        if not any(points_equal((cand.x, cand.y), (u.x, u.y), tol) for u in unique):
            unique.append(cand)
    return unique

def compute_distance(path):
    """
    計算由 (x, y) 座標組成的路徑總距離
    """
    distance = 0.0
    for i in range(len(path) - 1):
        distance += math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
    return distance

# ----------------------------
# A* 演算法類別
# ----------------------------
class AStar:
    def __init__(self, obstacles, roads, resolution, robot_radius, show_animation=False):
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.obstacles = obstacles
        self.roads = roads
        self.show_animation = show_animation
        self.motion = self.get_motion_model()
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None

    class Node:
        def __init__(self, x, y, g_cost, h_cost, parent_index):
            self.x = x
            self.y = y
            self.g_cost = g_cost
            self.h_cost = h_cost
            self.f_cost = g_cost + h_cost
            self.parent_index = parent_index

        def __lt__(self, other):
            return self.f_cost < other.f_cost

    def planning(self, sx, sy, gx, gy, boundary_polygon, target_func=None, ax=None):
        self.min_x, self.min_y, self.max_x, self.max_y = boundary_polygon.bounds
        self.x_width = int((self.max_x - self.min_x) / self.resolution)
        self.y_width = int((self.max_y - self.min_y) / self.resolution)
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y),
                               0.0,
                               self.calc_heuristic(sx, sy, gx, gy),
                               -1)
        goal_node = None
        if target_func is None:
            goal_index_x = self.calc_xy_index(gx, self.min_x)
            goal_index_y = self.calc_xy_index(gy, self.min_y)
            goal_node = self.Node(goal_index_x, goal_index_y, 0.0, 0.0, -1)

        open_set = []
        closed_set = {}
        open_set_dict = {}
        heapq.heappush(open_set, start_node)
        open_set_dict[self.calc_index(start_node)] = start_node

        self.calc_obstacle_map(boundary_polygon)
        tol_val = self.resolution * 0.5
        max_iterations = 10000
        iteration = 0

        while open_set and iteration < max_iterations:
            iteration += 1
            current = heapq.heappop(open_set)
            del open_set_dict[self.calc_index(current)]
            current_pos = Point(self.calc_position(current.x, self.min_x),
                                self.calc_position(current.y, self.min_y))
            if target_func is not None:
                if target_func(current_pos):
                    goal_node = current
                    break
            else:
                if current.x == self.calc_xy_index(gx, self.min_x) and current.y == self.calc_xy_index(gy, self.min_y):
                    goal_node = current
                    break
                if current_pos.distance(Point(gx, gy)) <= tol_val:
                    goal_node = current
                    break

            closed_set[self.calc_index(current)] = current

            for move in self.motion:
                move_x, move_y, move_cost = move
                neighbor_x = current.x + move_x
                neighbor_y = current.y + move_y
                if neighbor_x < 0 or neighbor_x >= self.x_width or neighbor_y < 0 or neighbor_y >= self.y_width:
                    continue
                neighbor_px = self.calc_position(neighbor_x, self.min_x)
                neighbor_py = self.calc_position(neighbor_y, self.min_y)
                h = self.calc_heuristic(neighbor_px, neighbor_py, gx, gy)
                node = self.Node(neighbor_x, neighbor_y, current.g_cost + move_cost, h, self.calc_index(current))
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
        if iteration >= max_iterations:
            print("Max iterations reached!")
        if goal_node is None:
            return [], []
        path_x, path_y = self.calc_final_path(goal_node, closed_set)
        if ax is not None:
            ax.plot(path_x, path_y, "-r", linewidth=2, label="A* Path")
            ax.legend()
        return path_x, path_y

    def calc_final_path(self, goal_node, closed_set):
        rx = [self.calc_position(goal_node.x, self.min_x)]
        ry = [self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            if parent_index not in closed_set:
                print("Warning: Missing parent node in closed_set. Incomplete path.")
                break
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx[::-1], ry[::-1]

    def calc_position(self, index, minp):
        return index * self.resolution + minp

    def calc_xy_index(self, position, minp):
        return int((position - minp) / self.resolution)

    def calc_index(self, node):
        return node.y * self.x_width + node.x

    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        return not self.obstacle_map[node.x][node.y]

    def calc_obstacle_map(self, boundary_polygon):
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        clipped_obs = gpd.clip(self.obstacles, boundary_polygon)
        for geom in clipped_obs.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                buffered = geom.buffer(self.robot_radius)
                minx, miny, maxx, maxy = buffered.bounds
                for x in range(self.calc_xy_index(minx, self.min_x), self.calc_xy_index(maxx, self.min_x) + 1):
                    for y in range(self.calc_xy_index(miny, self.min_y), self.calc_xy_index(maxy, self.min_y) + 1):
                        px = self.calc_position(x, self.min_x)
                        py = self.calc_position(y, self.min_y)
                        if buffered.contains(Point(px, py)):
                            if 0 <= x < self.x_width and 0 <= y < self.y_width:
                                self.obstacle_map[x][y] = True
        clipped_roads = gpd.clip(self.roads, boundary_polygon)
        for geom in clipped_roads.geometry:
            if geom.geom_type == 'LineString':
                buffered = geom.buffer(self.robot_radius)
                minx, miny, maxx, maxy = buffered.bounds
                for x in range(self.calc_xy_index(minx, self.min_x), self.calc_xy_index(maxx, self.min_x) + 1):
                    for y in range(self.calc_xy_index(miny, self.min_y), self.calc_xy_index(maxy, self.min_y) + 1):
                        px = self.calc_position(x, self.min_x)
                        py = self.calc_position(y, self.min_y)
                        if buffered.contains(Point(px, py)):
                            if 0 <= x < self.x_width and 0 <= y < self.y_width:
                                self.obstacle_map[x][y] = True

    def get_motion_model(self):
        return [
            (1, 0, 1),
            (0, 1, 1),
            (-1, 0, 1),
            (0, -1, 1),
            (1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (-1, -1, math.sqrt(2))
        ]

    @staticmethod
    def calc_heuristic(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

# ----------------------------
# 儲存圖檔函式：依據起終點名稱存圖（尺寸20×10），不顯示圖
# ----------------------------
def plot_path_single(path, obstacles, roads, start_pt, end_pt, boundary_polygon, start_name, end_name, title="Path"):
    # 設定字型以解決中文或日文亂碼問題，依系統環境調整字型名稱
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows
    # plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]  # Linux
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(start_pt.x, start_pt.y, "og", markersize=8, label="Start")
    ax.plot(end_pt.x, end_pt.y, "xb", markersize=8, label="End")
    clipped_obs = gpd.clip(obstacles, boundary_polygon)
    clipped_obs.plot(ax=ax, facecolor="none", edgecolor="black", label="Obstacles")
    clipped_roads = gpd.clip(roads, boundary_polygon)
    clipped_roads.plot(ax=ax, color="green", label="Roads")
    bx, by = boundary_polygon.exterior.xy
    ax.plot(bx, by, "--r", label="Boundary")
    path_line = LineString(path)
    ax.plot(*path_line.xy, "-r", linewidth=2, label="Combined Path")
    ax.set_title(title)
    ax.legend()
    file_name = f"{start_name}_{end_name}.png"
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)

def create_boundary_polygon(sx, sy, gx, gy, buffer_distance=20):
    min_x = min(sx, gx) - buffer_distance
    min_y = min(sy, gy) - buffer_distance
    max_x = max(sx, gx) + buffer_distance
    max_y = max(sy, gy) + buffer_distance
    return box(min_x, min_y, max_x, max_y)

# ----------------------------
# 儲存所有路徑為 Shapefile（每條路徑都儲存）
# ----------------------------
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

# ----------------------------
# 主程式
# ----------------------------
def main():
    print("Starting main process...")
    start_points = load_shapefile('start_points.shp')
    end_points = load_shapefile('end_points.shp')
    obstacles = load_shapefile('obstacles.shp')
    roads = load_shapefile('roads.shp')

    # 過濾掉 geometry 為 None 的記錄
    start_points = start_points[start_points.geometry.notnull()]
    end_points = end_points[end_points.geometry.notnull()]

    crs = start_points.crs
    resolution = 1.0
    robot_radius = 1.0
    show_animation = False

    # 調整距離門檻（依資料而定）
    distance_threshold = 500  

    astar = AStar(obstacles, roads, resolution, robot_radius, show_animation=show_animation)
    paths = []
    tol_val = resolution * 0.5

    # 逐一處理每個起點
    for idx, start in start_points.iterrows():
        if start.geometry is None:
            continue
        # 過濾障礙物（利用 safe_contains 避免錯誤）
        obstacles_filtered = obstacles[~obstacles.geometry.apply(lambda g: safe_contains(g, start.geometry))]
        valid_end_points = end_points[end_points.geometry.notnull()]
        valid_end_points = valid_end_points[valid_end_points.geometry.distance(start.geometry) <= distance_threshold]
        if valid_end_points.empty:
            print(f"No valid end points within {distance_threshold} for start point {start['Name']}.")
            continue

        for jdx, end in valid_end_points.iterrows():
            # 同一組起終點只針對 boundary_polygon 內資料進行計算
            obstacles_filtered2 = obstacles_filtered[~obstacles_filtered.geometry.apply(lambda g: safe_contains(g, end.geometry))]
            astar.obstacles = obstacles_filtered2
            boundary_polygon = create_boundary_polygon(start.geometry.x, start.geometry.y, end.geometry.x, end.geometry.y, buffer_distance=20)
            print(f"Processing path from {start['Name']} to {end['NAME']}...")

            # 先取得起點與終點在道路上的候選連接節點
            start_candidates = get_candidate_road_nodes(start.geometry, roads)
            end_candidates = get_candidate_road_nodes(end.geometry, roads)
            if not start_candidates or not end_candidates:
                print("No candidate road nodes found for start or end.")
                continue

            # 過濾並排序候選點（僅保留與原始點距離較近的 top 3）
            start_candidates = sorted(start_candidates, key=lambda p: p.distance(start.geometry))[:3]
            end_candidates = sorted(end_candidates, key=lambda p: p.distance(end.geometry))[:3]

            # 建立 boundary_polygon 範圍內的道路網路圖，減少計算量
            road_graph = nx.Graph()
            for road in roads.geometry:
                if not road.intersects(boundary_polygon):
                    continue
                if road.geom_type == "LineString":
                    coords = list(road.coords)
                    for i in range(len(coords) - 1):
                        p1 = Point(coords[i])
                        p2 = Point(coords[i + 1])
                        road_graph.add_edge(p1, p2, weight=p1.distance(p2))
                elif road.geom_type == "MultiLineString":
                    for sub in road.geoms:
                        if not sub.intersects(boundary_polygon):
                            continue
                        coords = list(sub.coords)
                        for i in range(len(coords) - 1):
                            p1 = Point(coords[i])
                            p2 = Point(coords[i + 1])
                            road_graph.add_edge(p1, p2, weight=p1.distance(p2))

            best_total_distance = float('inf')
            best_full_path = None

            # 對所有候選組合進行規劃
            for s_candidate in start_candidates:
                # (1) 起始段：利用 A* 規劃從起點到候選起點
                start_seg_x, start_seg_y = astar.planning(
                    start.geometry.x, start.geometry.y,
                    gx=s_candidate.x, gy=s_candidate.y,
                    boundary_polygon=boundary_polygon,
                    target_func=None,
                    ax=None
                )
                if not start_seg_x:
                    continue
                start_segment = list(zip(start_seg_x, start_seg_y))
                start_seg_distance = compute_distance(start_segment)

                for e_candidate in end_candidates:
                    # (3) 終點段：以直線連線候選終點與真實終點
                    end_segment = [(e_candidate.x, e_candidate.y), (end.geometry.x, end.geometry.y)]
                    end_seg_distance = math.hypot(e_candidate.x - end.geometry.x, e_candidate.y - end.geometry.y)

                    # 將候選節點加入道路圖（避免重複加入）
                    existing_s = find_node(road_graph, s_candidate, tol=1e-6)
                    if existing_s is None:
                        if not split_edge_at_node(road_graph, s_candidate, tol=1e-6):
                            nearest = snap_to_graph(s_candidate, road_graph)
                            road_graph.add_node(s_candidate)
                            if nearest is not None:
                                road_graph.add_edge(s_candidate, nearest, weight=s_candidate.distance(nearest))
                        road_node_s = s_candidate
                    else:
                        road_node_s = existing_s

                    existing_e = find_node(road_graph, e_candidate, tol=1e-6)
                    if existing_e is None:
                        if not split_edge_at_node(road_graph, e_candidate, tol=1e-6):
                            nearest = snap_to_graph(e_candidate, road_graph)
                            road_graph.add_node(e_candidate)
                            if nearest is not None:
                                road_graph.add_edge(e_candidate, nearest, weight=e_candidate.distance(nearest))
                        road_node_e = e_candidate
                    else:
                        road_node_e = existing_e

                    # (2) 道路段：利用道路網路尋找最短路徑
                    try:
                        road_path_nodes = nx.shortest_path(road_graph, source=road_node_s, target=road_node_e, weight='weight')
                        road_segment = [(p.x, p.y) for p in road_path_nodes]
                        road_seg_distance = compute_distance(road_segment)
                    except nx.NetworkXNoPath:
                        continue

                    total_distance = start_seg_distance + road_seg_distance + end_seg_distance
                    if total_distance < best_total_distance:
                        best_total_distance = total_distance
                        best_full_path = merge_segments_all(start_segment, road_segment, end_segment, tol=1e-6)

            if best_full_path is None:
                print(f"Failed to compute candidate-based path for start {start['Name']} and end {end['NAME']}.")
                continue

            # 確保起點與終點精確納入，再進行後處理
            if not points_equal(best_full_path[0], (start.geometry.x, start.geometry.y)):
                best_full_path.insert(0, (start.geometry.x, start.geometry.y))
            if not points_equal(best_full_path[-1], (end.geometry.x, end.geometry.y)):
                best_full_path.append((end.geometry.x, end.geometry.y))
            best_full_path = remove_loops(best_full_path, tol=1e-6)
            best_full_path = remove_consecutive_duplicates(best_full_path, tol=1e-6)

            print(f"Path from {start['Name']} to {end['NAME']} found, total distance: {best_total_distance:.2f}")
            paths.append((best_full_path, start['Name'], end['NAME'], best_total_distance))
            title = f"Path: {start['Name']} -> {end['NAME']}"
            plot_path_single(best_full_path, obstacles, roads, start.geometry, end.geometry, boundary_polygon, start['Name'], end['NAME'], title=title)

    if paths:
        save_paths_to_shp(paths, crs, "shortest_paths.shp")
    else:
        print("No valid paths were computed.")

    print("Main process completed.")

if __name__ == "__main__":
    main()
