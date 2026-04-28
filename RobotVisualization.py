import numpy as np
import random
from scipy.spatial import ConvexHull

""" Camera """
FOV_H = 69.4
FOV_V = 42.5
MAX_DISTANCE = 1000


class RobotVisualization:
    def __init__(self, min_a, max_a, min_b, max_b, min_x, max_x, min_y, max_y, min_z, max_z):
        self.params = None
        self.T_base_ee = None
        self.T_ee_cam = None
        self.T_base_cam = None
        self.cam_pos_base = None
        self.R_ee = None

        # Lưu các tham số giới hạn
        self.MIN_A = min_a
        self.MAX_A = max_a
        self.MIN_B = min_b
        self.MAX_B = max_b
        self.MIN_X = min_x
        self.MAX_X = max_x
        self.MIN_Y = min_y
        self.MAX_Y = max_y
        self.MIN_Z = min_z
        self.MAX_Z = max_z

    def generate_random_values(self):
        """Tạo các giá trị ngẫu nhiên trong phạm vi cho phép"""
        z = random.uniform(self.MIN_Z, self.MAX_Z)
        a = random.uniform(self.MIN_A, self.MAX_A)
        b = random.uniform(self.MIN_B, self.MAX_B)
        c = 180  # Góc cố định
        return z, a, b, c

    def rotation_matrix_x(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

    def rotation_matrix_y(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    def rotation_matrix_z(self, angle_deg):
        angle_rad = np.radians(angle_deg)
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    def create_homogeneous_matrix(self, R, t):
        """Tạo ma trận đồng nhất 4x4 từ ma trận quay 3x3 và vector tịnh tiến"""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def calculate_intersection_with_oxy(self, cam_pos, R_ee):
        """Tính giao tuyến giữa tầm nhìn camera và mặt phẳng Oxy (z=0)"""
        fov_h = np.radians(FOV_H) / 2
        fov_v = np.radians(FOV_V) / 2

        intersections = []
        for y in [-np.tan(fov_h), np.tan(fov_h)]:
            for x in [-np.tan(fov_v), np.tan(fov_v)]:
                # Vector hướng trong hệ camera
                dir_cam = np.array([x, y, 1])
                # Chuyển sang hệ base
                dir_base = R_ee @ dir_cam

                # Tính giao điểm với mặt phẳng z=0
                if dir_base[2] != 0:  # Tránh chia cho 0
                    t = -cam_pos[2] / dir_base[2]
                    if t > 0:  # Chỉ lấy giao điểm phía trước camera
                        point = cam_pos + t * dir_base
                        intersections.append(point)

        # Sắp xếp các điểm giao để vẽ thành đa giác kín
        if len(intersections) >= 3:
            # Tính góc giữa các điểm và sắp xếp theo thứ tự
            angles = np.arctan2([p[1] - cam_pos[1] for p in intersections],
                               [p[0] - cam_pos[0] for p in intersections])
            sorted_points = [p for _, p in sorted(zip(angles, intersections))]
            return sorted_points
        return []

    def generate_disk_in_polygon(self, polygon_points, radius=62, max_attempts=1000):
        """
        Tạo đĩa bán kính cho trước nằm hoàn toàn trong đa giác với tâm random.
        Thêm điều kiện x phải nhỏ hơn MAX_X và y nằm trong khoảng MIN_Y đến MAX_Y.
        """
        # Chuyển sang numpy array và chỉ lấy x, y (bỏ z=0)
        points = np.array([p[:2] for p in polygon_points])

        # 1. Tính hình bao lồi và bounding box
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        min_coords = np.min(hull_points, axis=0)
        max_coords = np.max(hull_points, axis=0)

        # 2. Kiểm tra kích thước tối thiểu
        min_dimension = min(max_coords - min_coords)
        if min_dimension < 2 * radius:
            safe_radius = min_dimension / 2 * 0.9
            print(f"Đa giác quá nhỏ cho R={radius}, sử dụng R={safe_radius:.1f}")
            return np.mean(hull_points, axis=0)[0], np.mean(hull_points, axis=0)[1], safe_radius

        # 3. Hàm kiểm tra điểm có nằm trong đa giác không
        def point_in_polygon(point, polygon):
            x, y = point
            n = len(polygon)
            inside = False
            p1x, p1y = polygon[0]
            for i in range(n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        # 4. Hàm tính khoảng cách an toàn tới các cạnh
        def min_distance_to_edges(point, polygon):
            distances = []
            n = len(polygon)
            for i in range(n):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % n]
                edge = p2 - p1
                edge_length = np.linalg.norm(edge)
                normal = np.array([-edge[1], edge[0]]) / edge_length
                distance = np.abs(np.dot(point - p1, normal))
                distances.append(distance)
            return min(distances)

        # 5. Random điểm và kiểm tra với điều kiện x < MAX_X và y trong khoảng MIN_Y đến MAX_Y
        best_center = None
        best_radius = 0

        for _ in range(max_attempts):
            # Random điểm trong bounding box với điều kiện
            candidate = np.array([random.uniform(min_coords[0], max_coords[0]),
                                random.uniform(min_coords[1], max_coords[1])])

            if not (self.MIN_X <= candidate[0] <= self.MAX_X) or not (self.MIN_Y <= candidate[1] <= self.MAX_Y):
                continue

            # Kiểm tra điểm có trong đa giác không
            if not point_in_polygon(candidate, hull_points):
                continue

            # Tính khoảng cách an toàn
            safe_distance = min_distance_to_edges(candidate, hull_points)

            # Nếu tìm được vị trí tốt hơn
            if safe_distance > best_radius:
                best_radius = min(safe_distance, radius)
                best_center = candidate

                # Nếu đã đạt bán kính mong muốn thì dừng
                if best_radius >= radius:
                    break

        # Nếu không tìm được sau nhiều lần, lấy điểm tốt nhất đã tìm thấy
        if best_center is None:
            best_center = np.mean(hull_points, axis=0)
            best_radius = min_distance_to_edges(best_center, hull_points) * 0.9
            print(f"Không tìm được vị trí phù hợp sau {max_attempts} lần thử")

        return best_center[0], best_center[1], min(best_radius, radius)

    def calculate_transform_matrices(self):
        # 1. Lấy thông số ngẫu nhiên
        z, a, b, c = self.generate_random_values()
        x, y = 250, 0  # Vị trí ban đầu

        # 2. Tính các ma trận quay
        Rz = self.rotation_matrix_z(a)
        Ry = self.rotation_matrix_y(b)
        Rx = self.rotation_matrix_x(c)

        # 3. Ma trận quay tổng hợp (Z->Y->X)
        R_ee = Rz @ Ry @ Rx

        # 4. Ma trận base -> EE
        T_base_ee = self.create_homogeneous_matrix(R_ee, np.array([x, y, z]))

        # 5. Vị trí camera trong hệ EE (cố định)
        cam_pos_ee = np.array([-46, 32.5, 15]).reshape(3, 1)

        # 6. Ma trận EE -> Camera (chỉ tịnh tiến)
        T_ee_cam = self.create_homogeneous_matrix(np.eye(3), cam_pos_ee)

        # 7. Ma trận base -> Camera
        T_base_cam = T_base_ee @ T_ee_cam

        # 8. Tính vị trí camera trong hệ base
        cam_pos_base = T_base_ee[:3, :3] @ cam_pos_ee + T_base_ee[:3, [3]]

        self.params = {'x': x, 'y': y, 'z': z, 'a': a, 'b': b, 'c': c}
        self.T_base_ee = T_base_ee
        self.T_ee_cam = T_ee_cam
        self.T_base_cam = T_base_cam
        self.cam_pos_base = cam_pos_base
        self.R_ee = R_ee

    def visualize_with_disk(self):
        self.calculate_transform_matrices()

        # Lấy dữ liệu từ kết quả tính toán
        cam_pos_base = self.cam_pos_base.flatten()
        R_ee = self.R_ee

        # Tính và vẽ giao tuyến với Oxy
        intersection_points = self.calculate_intersection_with_oxy(cam_pos_base, R_ee)

        if len(intersection_points) >= 3:
            try:
                disk_x, disk_y, actual_radius = self.generate_disk_in_polygon(intersection_points, radius=62)

                # Thêm dòng in thông tin vị trí
                print(
                    f"{500 - disk_x:.6f}, {0 - disk_y:.6f}, {self.params['z']}, {self.params['a']}, {self.params['b']}, {self.params['c']}")

                return disk_x, disk_y, self.params['z'], self.params['a'], self.params['b'], self.params['c']
            except Exception as e:
                print(f"Không thể tạo đĩa: {str(e)}")
                return None
        else:
            print("Không tìm thấy giao điểm với mặt phẳng Oxy")
            return None