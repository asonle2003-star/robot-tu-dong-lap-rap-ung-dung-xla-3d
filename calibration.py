
import cv2
import numpy as np
import os
from datetime import datetime

pattern_size = (9, 6)
square_size = 10

image_directory = 'images'

if not os.path.exists(image_directory):
    print(f"The directory {image_directory} does not exist!")
else:
    images = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]
    print(f"Found {len(images)} images.")

    R_gripper_list = []
    T_gripper_list = []
    R_target_list = []
    T_target_list = []

    for image_name in images:
        try:
            parts = image_name.lstrip('_').split('_')
            if len(parts) == 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                a, b, c = float(parts[3]), float(parts[4]), float(parts[5].split('.')[0])

                image_path = os.path.join(image_directory, image_name)
                image = cv2.imread(image_path)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

                if ret:
                    obj_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
                    obj_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
                    obj_points *= square_size

                    img_points = corners.reshape(-1, 2)

                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([obj_points], [img_points], gray.shape[::-1], None, None)

                    R_cam_to_checker, _ = cv2.Rodrigues(rvecs[0])
                    T_cam_to_checker = tvecs[0]

                    Rz = np.array([
                        [np.cos(np.radians(a)), -np.sin(np.radians(a)), 0],
                        [np.sin(np.radians(a)), np.cos(np.radians(a)), 0],
                        [0, 0, 1]
                    ])
                    Ry = np.array([
                        [np.cos(np.radians(b)), 0, np.sin(np.radians(b))],
                        [0, 1, 0],
                        [-np.sin(np.radians(b)), 0, np.cos(np.radians(b))]
                    ])
                    Rx = np.array([
                        [1, 0, 0],
                        [0, np.cos(np.radians(c)), -np.sin(np.radians(c))],
                        [0, np.sin(np.radians(c)), np.cos(np.radians(c))]
                    ])

                    R_ee = Rz @ Ry @ Rx

                    T_base_ee = np.eye(4)
                    T_base_ee[:3, :3] = R_ee
                    T_base_ee[:3, 3] = [x, y, z]

                    R_gripper_list.append(R_ee)
                    T_gripper_list.append(np.array([x, y, z]))

                    R_target_list.append(R_cam_to_checker)
                    T_target_list.append(T_cam_to_checker)

        except ValueError as e:
            print(f"Error processing file {image_name}: {e}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f'calibration_{timestamp}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, '2_base_to_end-effector.txt'), 'w') as f:
        f.write("Base to EE Rotation Matrix and Translation Matrix (4x4)\n")
        f.write("===============================================\n")
        for i in range(len(R_gripper_list)):
            f.write(f"Base to End-Effector (Image {i+1}):\n")
            f.write("Rotation Matrix (4x4):\n")
            base_to_ee_4x4 = np.eye(4)
            base_to_ee_4x4[:3, :3] = R_gripper_list[i]
            base_to_ee_4x4[:3, 3] = T_gripper_list[i].flatten()
            np.savetxt(f, base_to_ee_4x4, fmt='%0.6f', delimiter=' ')
            f.write("\n")

    with open(os.path.join(output_dir, '1_camera_to_checkerboard.txt'), 'w') as f:
        f.write("Camera to Checkerboard Rotation Matrix and Translation Matrix (4x4)\n")
        f.write("===============================================================\n")
        for i in range(len(R_target_list)):
            f.write(f"Camera to Checkerboard (Image {i+1}):\n")
            f.write("Rotation Matrix (4x4):\n")
            cam_to_checker_4x4 = np.eye(4)
            cam_to_checker_4x4[:3, :3] = R_target_list[i]
            cam_to_checker_4x4[:3, 3] = T_target_list[i].flatten()
            np.savetxt(f, cam_to_checker_4x4, fmt='%0.6f', delimiter=' ')
            f.write("\n")

    with open(os.path.join(output_dir, '0_calibration_results.txt'), 'w') as f:
        f.write("Calibration Results: Camera to EE\n")
        f.write("=================================\n")
        method_dict = {
            "Tsai": cv2.CALIB_HAND_EYE_TSAI,
            "Park": cv2.CALIB_HAND_EYE_PARK,
            "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "Andreff": cv2.CALIB_HAND_EYE_ANDREFF
        }

        for method_name, method_flag in method_dict.items():
            f.write(f"Using method: {method_name}\n")
            R_cam_to_ee, T_cam_to_ee = cv2.calibrateHandEye(R_gripper_list, T_gripper_list, R_target_list, T_target_list, method=method_flag)
            f.write("Camera to End-Effector Rotation Matrix (4x4):\n")
            cam_to_ee_4x4 = np.eye(4)
            cam_to_ee_4x4[:3, :3] = R_cam_to_ee
            cam_to_ee_4x4[:3, 3] = T_cam_to_ee.flatten()
            np.savetxt(f, cam_to_ee_4x4, fmt='%0.6f', delimiter=' ')

            print(f"Method: {method_name}")
            print("Camera to End-Effector Rotation Matrix (4x4):")
            print(cam_to_ee_4x4[:3, :3])
            print("Camera to End-Effector Translation Matrix:")
            print(cam_to_ee_4x4[:3, 3])
            print("-" * 50)

            f.write("-" * 50 + "\n")

    print(f"Calibration matrices have been saved to the folder: {output_dir}")
