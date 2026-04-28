import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from datetime import datetime
import glob


def create_output_folder():
    """Tạo thư mục output với tên là ngày giờ hiện tại"""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"calib_results_{now}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def save_calibration_results(output_folder, H_tool2cam, K, dist,
                             base_to_ee_list, cam_to_checker_list):
    """Lưu tất cả kết quả calibration vào thư mục"""
    # Lưu các file chính
    np.savetxt(os.path.join(output_folder, "tool_to_camera.txt"), H_tool2cam)
    np.savetxt(os.path.join(output_folder, "camera_matrix.txt"), K)
    np.savetxt(os.path.join(output_folder, "dist_coeffs.txt"), dist)

    # Lưu các ma trận cho từng ảnh
    for i, (base_to_ee, cam_to_checker) in enumerate(zip(base_to_ee_list, cam_to_checker_list)):
        np.savetxt(os.path.join(output_folder, f"base_to_ee_{i:02d}.txt"), base_to_ee)
        np.savetxt(os.path.join(output_folder, f"camera_to_checker_{i:02d}.txt"), cam_to_checker)

    print(f"\nĐã lưu tất cả kết quả vào thư mục: {output_folder}")


def calibrate_eye_in_hand(image_folder):
    # 1. Tạo thư mục output
    output_folder = create_output_folder()

    # 2. Tải dữ liệu
    images, poses = load_images_and_poses(image_folder)
    if len(images) == 0:
        print("Không có ảnh nào được đọc thành công!")
        return None

    # 3. Phát hiện bàn cờ
    pattern_size = (9, 6)
    square_size = 0.01
    corners, valid_indices = detect_checkerboard(images, pattern_size)

    if len(corners) < 5:
        print(f"Không đủ ảnh phát hiện bàn cờ (cần ít nhất 5, hiện có {len(corners)})")
        return None

    # 4. Calib nội tại camera
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    img_size = (images[0].shape[1], images[0].shape[0])
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp] * len(corners), corners, img_size, None, None)

    print("\nKết quả calib nội tại:")
    print(f"Lỗi reprojection: {ret:.4f}")

    # 5. Tính toán các ma trận chuyển đổi
    H_target2cam = []
    base_to_ee_list = []
    cam_to_checker_list = []

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # Ma trận từ camera đến bàn cờ
        R, _ = cv2.Rodrigues(rvec)
        H_cam2checker = np.eye(4)
        H_cam2checker[:3, :3] = R
        H_cam2checker[:3, 3] = tvec.flatten()
        cam_to_checker_list.append(H_cam2checker)

        # Ma trận từ base đến end-effector
        pose = poses[valid_indices[i]]
        x, y, z, a, b, c = pose
        x, y, z = x / 1000, y / 1000, z / 1000  # mm -> m

        rot = Rotation.from_euler('zyx', [a, b, c], degrees=True)
        H_base2ee = np.eye(4)
        H_base2ee[:3, :3] = rot.as_matrix()
        H_base2ee[:3, 3] = [x, y, z]
        base_to_ee_list.append(H_base2ee)

        H_target2cam.append(H_cam2checker)

    # 6. Eye-in-Hand Calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=[np.linalg.inv(H[:3, :3]) for H in base_to_ee_list],
        t_gripper2base=[-H[:3, 3] for H in base_to_ee_list],
        R_target2cam=[H[:3, :3] for H in H_target2cam],
        t_target2cam=[H[:3, 3] for H in H_target2cam],
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    H_tool2cam = np.eye(4)
    H_tool2cam[:3, :3] = R_cam2gripper
    H_tool2cam[:3, 3] = t_cam2gripper.flatten()

    # 7. Lưu kết quả
    save_calibration_results(output_folder, H_tool2cam, K, dist,
                             base_to_ee_list, cam_to_checker_list)

    return H_tool2cam, K, dist


if __name__ == "__main__":
    image_folder = "images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Đã tạo thư mục {image_folder}. Vui lòng đặt ảnh calibration vào thư mục này.")
        exit()

    result = calibrate_eye_in_hand(image_folder)
    if result is None:
        print("\nCalibration thất bại. Vui lòng kiểm tra lại dữ liệu đầu vào.")