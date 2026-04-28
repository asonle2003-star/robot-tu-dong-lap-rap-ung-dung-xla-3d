import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
import os
import shutil
from cri_lib import CRIController
from RobotVisualization import RobotVisualization


class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.pipeline.start(self.config)
        self.running = True
        self.thread = threading.Thread(target=self.stream_data)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()

    def stream_data(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            with self.lock:
                self.frame = color_image

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def detect_checkerboard(self, image):
        if image is None:
            return None, False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern_size = (9, 6)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        mask = np.full_like(image, (255, 255, 255))
        if ret and len(corners) == pattern_size[0] * pattern_size[1]:
            cv2.drawChessboardCorners(mask, pattern_size, corners, ret)
        return mask, ret


class Robot:
    def __init__(self, params):
        self.robot_controller = CRIController()
        self.connected = False
        self.robot_visualization = RobotVisualization(
            params['min_a'], params['max_a'],
            params['min_b'], params['max_b'],
            params['min_x'], params['max_x'],
            params['min_y'], params['max_y'],
            params['min_z'], params['max_z']
        )
        self.camera = Camera()
        self.last_values = None
        self.should_stop = False

    def connect(self, ip: str, port: int = 3920):
        self.connected = self.robot_controller.connect(ip, port)
        return self.connected

    def stop(self):
        self.should_stop = True
        if self.connected:
            self.robot_controller.close()
            self.connected = False
        self.camera.stop()

    def get_status(self):
        if self.connected:
            return self.robot_controller.robot_state
        return None

    def visualize_robot_and_get_values(self):
        self.robot_visualization.calculate_transform_matrices()
        disk_x, disk_y, z, a, b, c = self.robot_visualization.visualize_with_disk()
        self.last_values = (disk_x, disk_y, z, a, b, c)
        return disk_x, disk_y, z, a, b, c

    def move_robot(self, disk_x, disk_y, z, a, b, c):
        x = 500 - disk_x
        y = 0 - disk_y
        success = self.robot_controller.move_cartesian(x, y, z, a, b, c, 0.0, 0.0, 0.0, 30.0,
                                                       wait_move_finished=True,
                                                       move_finished_timeout=1000)
        if not success:
            print("Di chuyển robot thất bại, đưa robot về vị trí gốc...")
            # Đưa robot về vị trí gốc nếu di chuyển thất bại
            self.move_robot_to_origin()
            return False
        return True

    def move_robot_to_origin(self):
        success = self.robot_controller.move_cartesian(250, 0, 250, 180, 0, 180, 0.0, 0.0, 0.0, 30.0,
                                                       wait_move_finished=True,
                                                       move_finished_timeout=1000)
        if not success:
            print("Không thể đưa robot về vị trí gốc!")
        else:
            print("Robot đã quay lại vị trí gốc!")

    def start_camera(self):
        self.camera.start()

    def get_camera_frame(self):
        return self.camera.get_frame()

    def capture_checkerboard_images(self, num_required=3):
        captured_images = []
        attempts = 0
        max_attempts = 20

        while len(captured_images) < num_required and attempts < max_attempts and not self.should_stop:
            attempts += 1

            # Random di chuyển robot
            disk_x, disk_y, z, a, b, c = self.visualize_robot_and_get_values()
            success = self.move_robot(disk_x, disk_y, z, a, b, c)

            if not success:
                print("Di chuyển robot thất bại, sẽ thử lại...")
                continue

            time.sleep(1)

            # Lấy vị trí thực tế của robot
            status = self.get_status()
            if not status:
                continue
            pos = status.position_robot

            # Chụp ảnh và kiểm tra
            frame = self.get_camera_frame()
            if frame is None:
                continue

            _, has_checkerboard = self.camera.detect_checkerboard(frame)

            if has_checkerboard:
                # Tạo tên file theo yêu cầu
                filename = f"_{pos.X:.2f}_{pos.Y:.2f}_{pos.Z:.2f}_{pos.A:.2f}_{pos.B:.2f}_{pos.C:.2f}.jpg"
                captured_images.append((frame, filename))
                print(f"Đã chụp được ảnh {len(captured_images)}/{num_required}: {filename}")
            else:
                print(f"Không phát hiện checkerboard (lần thử {attempts})")

        return captured_images


def camera_stream_thread(robot):

    while not robot.should_stop:
        frame = robot.get_camera_frame()
        if frame is not None:
            mask, ret = robot.camera.detect_checkerboard(frame)
            if ret:
                cv2.imshow("Checkerboard Mask", mask)
            else:
                # Hiển thị frame gốc nếu không tìm thấy checkerboard
                cv2.imshow("Checkerboard Mask", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            robot.should_stop = True
            break


def main():

    # Tạo thư mục lưu ảnh
    if os.path.exists("images"):
        shutil.rmtree("images")
    os.makedirs("images", exist_ok=True)

    # 9 bộ giá trị đầy đủ
    ab_sets = [
        # 1
        {'min_a': 140, 'max_a': 160, 'min_b': -21, 'max_b': -12, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 2
        {'min_a': 170, 'max_a': 190, 'min_b': -21, 'max_b': -12, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 3
        {'min_a': 200, 'max_a': 220, 'min_b': -21, 'max_b': -12, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 4
        {'min_a': 140, 'max_a': 160, 'min_b': -7, 'max_b': 7, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 5
        {'min_a': 170, 'max_a': 190, 'min_b': -7, 'max_b': 7, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 6
        {'min_a': 200, 'max_a': 220, 'min_b': -7, 'max_b': 7, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 7
        {'min_a': 140, 'max_a': 160, 'min_b': 12, 'max_b': 21, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 8
        {'min_a': 170, 'max_a': 190, 'min_b': 12, 'max_b': 21, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300},
        # 9
        {'min_a': 200, 'max_a': 220, 'min_b': 12, 'max_b': 21, 'min_x': -80, 'max_x': 400, 'min_y': -120,
         'max_y': 120, 'min_z': 180, 'max_z': 300}
    ]

    ip_address = "192.168.3.11"

    # Xử lý từng bộ tham số
    for set_idx, params in enumerate(ab_sets):
        print(f"\nBắt đầu xử lý bộ tham số {set_idx + 1}/9")


        # Khởi tạo robot với bộ tham số hiện tại
        robot = Robot(params)

        if not robot.connect(ip_address):
            print("Kết nối robot thất bại")
            continue

        robot.start_camera()
        time.sleep(3)  # Chờ camera khởi động

        # Bắt đầu luồng hiển thị camera
        stream_thread = threading.Thread(target=camera_stream_thread, args=(robot,))
        stream_thread.daemon = True
        stream_thread.start()

        # Thu thập 3 ảnh có checkerboard
        images = robot.capture_checkerboard_images(3)

        # Lưu ảnh
        for img, filename in images:
            cv2.imwrite(os.path.join("images", filename), img)
            print(f"Đã lưu ảnh: {filename}")

        robot.stop()
        stream_thread.join()
        cv2.destroyAllWindows()

    print("\nHoàn thành thu thập ảnh cho tất cả 9 bộ tham số")

    # Thực hiện calibration sau khi hoàn thành tất cả
    print("\nBắt đầu quá trình calibration...")
    try:
        import calibration
        print("Calibration đã hoàn thành!")
    except Exception as e:
        print(f"Lỗi khi thực hiện calibration: {str(e)}")

    print("\nChương trình kết thúc")


if __name__ == "__main__":
    main()