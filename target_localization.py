import os
import csv
import math
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

# Define Constants
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FOCAL_LENGTH_MM = 6.0      # 6.0 mm Wide-Angle Lens
SENSOR_WIDTH_MM = 6.287    # 6.287 mm Sensor Width
DEFAULT_FPS = 24           # 24 FPS
MAX_COASTING_FRAMES = 25   # Max missed frames before target marked LOST

def build_intrinsic_matrix(width, height, focal_length, sensor_width):
    fx = focal_length * (width / sensor_width)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K

def get_rotation_matrix(roll_deg, pitch_deg, yaw_deg, order='XYZ'):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(r), -math.sin(r)],
        [0.0, math.sin(r), math.cos(r)]
    ], dtype=np.float64)
    
    Ry = np.array([
        [math.cos(p), 0.0, math.sin(p)],
        [0.0, 1.0, 0.0],
        [-math.sin(p), 0.0, math.cos(p)]
    ], dtype=np.float64)
    
    Rz = np.array([
        [math.cos(y), -math.sin(y), 0.0],
        [math.sin(y), math.cos(y), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    if order == 'XYZ':
        return Rx @ Ry @ Rz
    elif order == 'ZYX':
        return Rz @ Ry @ Rx
    elif order == 'YXZ':
        return Ry @ Rx @ Rz
    elif order == 'ZXY':
        return Rz @ Rx @ Ry
    else:
        return Rx @ Ry @ Rz

def detect_target_yolo_frame(frame, model, conf_threshold=0.1):
    results = model(frame, verbose=False)
    best_box = None
    best_conf = -1.0
    
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > conf_threshold and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()
                
    if best_box is not None:
        u_min, v_min, u_max, v_max = best_box[0], best_box[1], best_box[2], best_box[3]
        u_c = (u_min + u_max) / 2.0
        v_c = (v_min + v_max) / 2.0
        return int(u_min), int(v_min), int(u_max), int(v_max), u_c, v_c, best_conf
    return None

def detect_target_color_frame(frame):
    if frame is None:
        return None
        
    target_bgr = np.array([106, 111, 214], dtype=np.float64)
    diff = np.linalg.norm(frame.astype(np.float64) - target_bgr, axis=2)
    
    mask = (diff < 40.0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    max_area = -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10.0 and area > max_area:
            max_area = area
            best_contour = c
            
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            u_c = M["m10"] / M["m00"]
            v_c = M["m01"] / M["m00"]
            return x, y, x + w, y + h, u_c, v_c
    return None

def project_target_mathematically(P_cam, R_world, P_target_gt, K):
    v_world = P_target_gt - P_cam
    v_cam = R_world.T @ v_world
    
    if abs(v_cam[2]) < 1e-6:
        return None
        
    x_cv = v_cam[0]
    y_cv = -v_cam[1]
    z_cv = -v_cam[2]
    
    if z_cv <= 0:
        return None
        
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * (x_cv / z_cv) + cx
    v = fy * (y_cv / z_cv) + cy
    return u, v

def unproject_pixel_to_ray(u, v, R_world, K_inv):
    d_cam_cv = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    d_cam_blender = np.array([d_cam_cv[0], -d_cam_cv[1], -d_cam_cv[2]], dtype=np.float64)
    d_world = R_world @ d_cam_blender
    v_unit = d_world / np.linalg.norm(d_world)
    return v_unit

def ray_ground_intersection(P_cam, v_ray, z_ground):
    if abs(v_ray[2]) < 1e-6:
        return None
    t = (z_ground - P_cam[2]) / v_ray[2]
    if t <= 0:
        return None
    P_intersect = P_cam + t * v_ray
    return P_intersect[0:2]

class SlidingWindowQueue:
    def __init__(self, size=5):
        self.size = size
        self.queue = []

    def add(self, P, v):
        self.queue.append((P, v))
        if len(self.queue) > self.size:
            self.queue.pop(0)

    def get_data(self):
        P_list = [item[0] for item in self.queue]
        v_list = [item[1] for item in self.queue]
        return P_list, v_list

class TargetKalmanFilter2D:
    def __init__(self, init_pos_2d, init_cov=1.0, q_pos=1e-3, q_vel=1e-6, r_val=1.0):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.x[0:2, 0] = init_pos_2d
        
        self.P = np.eye(4, dtype=np.float64) * init_cov
        self.P[2:4, 2:4] *= 0.1
        
        self.F = np.eye(4, dtype=np.float64)
        self.H = np.zeros((2, 4), dtype=np.float64)
        self.H[0:2, 0:2] = np.eye(2)
        
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.R = np.eye(2, dtype=np.float64) * r_val
        
    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0:2, 0:2] = np.eye(2) * self.q_pos * dt
        Q[2:4, 2:4] = np.eye(2) * self.q_vel * dt
        
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def update(self, z_2d, R=None):
        z_col = z_2d.reshape(2, 1)
        y = z_col - self.H @ self.x
        R_curr = R if R is not None else self.R
        S = self.H @ self.P @ self.H.T + R_curr
        
        try:
            K_gain = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K_gain @ y
            self.P = (np.eye(4) - K_gain @ self.H) @ self.P
        except np.linalg.LinAlgError:
            pass

def draw_hud_overlay(frame, frame_idx, total_frames, fps, u_c, v_c, P_est_2d, track_status, missed_count=0, is_paused=False, altitude=50.0, preset_name="normal"):
    overlay = frame.copy()
    
    box_x1, box_y1 = 20, 20
    box_x2, box_y2 = 580, 230
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
    
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    if track_status == "TRACKING":
        border_color = (0, 255, 0)
        status_text = f"TRACKING 2D [{preset_name.upper()} MODE]"
    elif track_status == "COASTING":
        border_color = (0, 255, 255)
        status_text = f"COASTING (MISS {missed_count}/{MAX_COASTING_FRAMES})"
    elif track_status == "LOST":
        border_color = (0, 0, 255)
        status_text = "TARGET LOST (RE-LOCKING)"
    else:
        border_color = (255, 255, 0)
        status_text = "WARMUP (BUFFERING)"
        
    if is_paused:
        status_text = "PAUSED (PRESS SPACE)"
        border_color = (0, 255, 255)
        
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), border_color, 2)
    
    time_sec = (frame_idx - 1) / fps if fps > 0 else 0.0
    centroid_str = f"({u_c:.1f}, {v_c:.1f}) px" if u_c is not None else "N/A (Occluded)"
    
    lines = [
        f"2D GROUND LOCALIZATION HUD [{status_text}]",
        f"Frame: {frame_idx:04d} / {total_frames:04d}  |  Time: {time_sec:.2f}s (@{fps}FPS)",
        f"2D Centroid: {centroid_str}",
        f"EST Ground X: {P_est_2d[0]:.4f} m" if P_est_2d is not None else "EST Ground X: Initializing...",
        f"EST Ground Y: {P_est_2d[1]:.4f} m" if P_est_2d is not None else "EST Ground Y: Initializing...",
        f"Altitude    : {altitude:.2f} meters"
    ]
    
    y_offset = 45
    for i, line in enumerate(lines):
        color = (255, 255, 255)
        scale = 0.55
        thickness = 1
        if i == 0:
            color = border_color
            scale = 0.60
            thickness = 2
        elif i in (3, 4):
            color = (0, 255, 0) if P_est_2d is not None else (0, 255, 255)
            thickness = 2
            
        cv2.putText(frame, line, (box_x1 + 15, box_y1 + y_offset + i * 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def print_mission_summary(preset, processed_count, total_frames, fps, altitude, z_ground, kf2d, tracked_estimates, valid_detection_count):
    time_total = (processed_count) / fps if fps > 0 else 0.0
    
    if kf2d is not None:
        best_x = kf2d.x[0, 0]
        best_y = kf2d.x[1, 0]
        vx = kf2d.x[2, 0]
        vy = kf2d.x[3, 0]
        
        var_x = kf2d.P[0, 0]
        var_y = kf2d.P[1, 1]
        std_pos = math.sqrt(max((var_x + var_y) / 2.0, 0.0))
        conf_radius = 1.96 * std_pos
        status_final = "ACTIVE TRACKING" if (processed_count - valid_detection_count) <= MAX_COASTING_FRAMES else "COASTING"
    elif len(tracked_estimates) > 0:
        best_x, best_y = np.mean(tracked_estimates, axis=0)
        vx, vy = 0.0, 0.0
        conf_radius = 0.50
        status_final = "RAW ESTIMATED"
    else:
        best_x, best_y = 0.0, 0.0
        vx, vy = 0.0, 0.0
        conf_radius = float('nan')
        status_final = "UNINITIALIZED"
        
    coverage_pct = (valid_detection_count / processed_count * 100.0) if processed_count > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("         REAL-TIME UAV TARGET LOCALIZATION MISSION SUMMARY")
    print("=" * 70)
    print(f"  Mission Flight Mode        : {preset.upper()} MODE")
    print(f"  Processed Video Stream     : {processed_count} / {total_frames} frames ({time_total:.2f} seconds @ {fps} FPS)")
    print(f"  Camera Operating Altitude  : {altitude:.2f} meters")
    print(f"  Final Target Track Status  : {status_final}")
    print("-" * 70)
    print("  BEST ESTIMATED TARGET 2D GROUND COORDINATES:")
    print(f"    X Ground (East-West)     : {best_x:10.4f} meters")
    print(f"    Y Ground (North-South)   : {best_y:10.4f} meters")
    print(f"    Z Ground Plane (Altitude): {z_ground:10.4f} meters")
    print(f"    Estimated Target Velocity: [{vx:.3f}, {vy:.3f}] m/s")
    print("-" * 70)
    print("  ESTIMATOR CONFIDENCE & QUALITY METRICS:")
    print(f"    95% Confidence Radius    : +/- {conf_radius:.4f} meters")
    print(f"    Valid Measurement Coverage: {valid_detection_count} / {processed_count} frames ({coverage_pct:.1f}%)")
    print(f"    Sliding Window Buffer N  : 5 frames")
    print("=" * 70 + "\n")

def process_video_pipeline(video_path, telemetry_path, output_video_path, mode='color',
                          preset='normal', seed=42, window_size=5,
                          q_pos=1e-3, q_vel=1e-6, r_val=1.0, init_cov=1.0, 
                          show_preview=False, loop_infinitely=False, custom_fps=DEFAULT_FPS):
    if seed is not None:
        np.random.seed(seed)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video file: {video_path}")
        
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = custom_fps if (cap_fps <= 0 or custom_fps is not None) else cap_fps
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if preset == 'normal':
        pos_noise_std = 0.2
        rot_noise_std = 0.1
        pixel_noise_std = 1.0
        missed_frames = set()
    else:  # 'pressure'
        pos_noise_std = 1.5
        rot_noise_std = 0.8
        pixel_noise_std = 8.0
        missed_frames = set(range(7, 11)).union(set(range(22, 27))).union(set(range(42, 50)))

    print(f"\nProcessing Video Stream [{preset.upper()} MODE]: {video_path}")
    print(f"Properties: {width}x{height} @ {fps} FPS, Total Frames: {total_frames}")
    
    window_name = f"2D Target Localization [{preset.upper()} MODE]"
    if show_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    with open(telemetry_path, 'r') as f:
        reader = csv.DictReader(f)
        telemetry_rows = list(reader)
        
    K = build_intrinsic_matrix(width, height, FOCAL_LENGTH_MM, SENSOR_WIDTH_MM)
    K_inv = np.linalg.inv(K)
    
    P_target_gt_3d = np.array([float(telemetry_rows[0]['target_x']), float(telemetry_rows[0]['target_y']), float(telemetry_rows[0]['target_z'])], dtype=np.float64)
    z_ground = P_target_gt_3d[2]
    altitude = float(telemetry_rows[0]['cam_z'])
    
    noisy_positions = []
    noisy_orientations = []
    noisy_pixels = []
    
    for row in telemetry_rows:
        P_tr = np.array([float(row['cam_x']), float(row['cam_y']), float(row['cam_z'])], dtype=np.float64)
        roll_tr = float(row['roll_deg'])
        pitch_tr = float(row['pitch_deg'])
        yaw_tr = float(row['yaw_deg'])
        
        P_n = P_tr + np.random.normal(0.0, pos_noise_std, 3)
        roll_n = roll_tr + np.random.normal(0.0, rot_noise_std)
        pitch_n = pitch_tr + np.random.normal(0.0, rot_noise_std)
        yaw_n = yaw_tr + np.random.normal(0.0, rot_noise_std)
        pixel_n = np.random.normal(0.0, pixel_noise_std, 2)
            
        noisy_positions.append(P_n)
        noisy_orientations.append((roll_n, pitch_n, yaw_n))
        noisy_pixels.append(pixel_n)

    yolo_model = None
    if mode == 'yolo':
        yolo_model = YOLO("yolo11n.pt", verbose=False)

    sliding_window = SlidingWindowQueue(size=window_size)
    kf2d = None
    prev_frame_id = None
    missed_count = 0
    valid_detection_count = 0
    tracked_estimates = []
    
    processed_count = 0
    writer_written = False
    
    frame_delay_ms = int(1000.0 / fps) if fps > 0 else 41
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            if show_preview or loop_infinitely:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                processed_count = 0
                sliding_window = SlidingWindowQueue(size=window_size)
                kf2d = None
                prev_frame_id = None
                missed_count = 0
                valid_detection_count = 0
                tracked_estimates = []
                writer_written = True
                continue
            else:
                break
            
        processed_count += 1
        idx = processed_count - 1
        
        if idx >= len(telemetry_rows):
            if show_preview or loop_infinitely:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                processed_count = 0
                sliding_window = SlidingWindowQueue(size=window_size)
                kf2d = None
                prev_frame_id = None
                missed_count = 0
                valid_detection_count = 0
                tracked_estimates = []
                writer_written = True
                continue
            else:
                break
            
        row = telemetry_rows[idx]
        frame_id = int(row['frame_id'])
        
        is_simulated_miss = (preset == 'pressure' and frame_id in missed_frames)
        
        u_min, v_min, u_max, v_max = None, None, None, None
        u_c, v_c = None, None
        detection_found = False
        
        if not is_simulated_miss:
            if mode == 'yolo':
                res = detect_target_yolo_frame(frame, yolo_model)
                if res is not None:
                    u_min, v_min, u_max, v_max, u_c, v_c, _ = res
                    detection_found = True
                else:
                    res_col = detect_target_color_frame(frame)
                    if res_col is not None:
                        u_min, v_min, u_max, v_max, u_c, v_c = res_col
                        detection_found = True
            elif mode == 'color':
                res = detect_target_color_frame(frame)
                if res is not None:
                    u_min, v_min, u_max, v_max, u_c, v_c = res
                    detection_found = True
            elif mode == 'projection':
                R_tr = get_rotation_matrix(float(row['roll_deg']), float(row['pitch_deg']), float(row['yaw_deg']))
                u_tr, v_tr = project_target_mathematically(
                    np.array([float(row['cam_x']), float(row['cam_y']), float(row['cam_z'])]),
                    R_tr, P_target_gt_3d, K
                )
                u_c, v_c = u_tr, v_tr
                detection_found = True
            
        roll_n, pitch_n, yaw_n = noisy_orientations[idx]
        R_world_noisy = get_rotation_matrix(roll_n, pitch_n, yaw_n)
        
        P_est_2d = None
        track_status = "WARMUP"
        
        if detection_found:
            missed_count = 0
            valid_detection_count += 1
            u_noisy = u_c + noisy_pixels[idx][0]
            v_noisy = v_c + noisy_pixels[idx][1]
            
            v_ray = unproject_pixel_to_ray(u_noisy, v_noisy, R_world_noisy, K_inv)
            
            # --- 1. BOX 1: RAW ESTIMATE (NO KF COASTING) ---
            xy_raw = ray_ground_intersection(noisy_positions[idx], v_ray, z_ground)
            if xy_raw is not None:
                tracked_estimates.append(xy_raw)
                P_raw_3d = np.array([xy_raw[0], xy_raw[1], z_ground])
                uv_raw = project_target_mathematically(noisy_positions[idx], R_world_noisy, P_raw_3d, K)
                if uv_raw is not None:
                    u_rw, v_rw = int(uv_raw[0]), int(uv_raw[1])
                    cv2.rectangle(frame, (u_rw - 40, v_rw - 40), (u_rw + 40, v_rw + 40), (0, 0, 255), 2)
                    cv2.putText(frame, "NO KF COASTING", (u_rw - 40, max(20, v_rw - 48)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (u_rw, v_rw), 4, (0, 0, 255), -1)

            # --- 2. BOX 2: WITH KALMAN FILTER COASTING ---
            sliding_window.add(noisy_positions[idx], v_ray)
            P_list, v_list = sliding_window.get_data()
            
            xy_estimates = []
            for P_cam_i, v_ray_i in zip(P_list, v_list):
                xy = ray_ground_intersection(P_cam_i, v_ray_i, z_ground)
                if xy is not None:
                    xy_estimates.append(xy)
                    
            if len(xy_estimates) > 0:
                xy_meas = np.mean(xy_estimates, axis=0)
                if kf2d is None:
                    if len(P_list) >= window_size:
                        kf2d = TargetKalmanFilter2D(xy_meas, init_cov=init_cov, q_pos=q_pos, q_vel=q_vel, r_val=r_val)
                else:
                    dt = float(frame_id - prev_frame_id) if prev_frame_id is not None else 1.0
                    kf2d.predict(dt)
                    kf2d.update(xy_meas)
                    
            if kf2d is not None:
                P_est_2d = kf2d.x[0:2, 0]
                track_status = "TRACKING"
                
                P_est_3d = np.array([P_est_2d[0], P_est_2d[1], z_ground])
                uv_corr = project_target_mathematically(noisy_positions[idx], R_world_noisy, P_est_3d, K)
                if uv_corr is not None:
                    u_cr, v_cr = int(uv_corr[0]), int(uv_corr[1])
                    cv2.rectangle(frame, (u_cr - 45, v_cr - 45), (u_cr + 45, v_cr + 45), (0, 255, 0), 3)
                    cv2.putText(frame, "WITH KF COASTING", (u_cr - 45, max(20, v_cr + 62)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, (u_cr, v_cr), 4, (0, 255, 0), -1)
        else:
            # MISSED FRAME: Raw Box Disappears (No KF Coasting), KF Coasting Box Extrapolates
            missed_count += 1
            if kf2d is not None:
                dt = float(frame_id - prev_frame_id) if prev_frame_id is not None else 1.0
                kf2d.predict(dt)
                P_est_2d = kf2d.x[0:2, 0]
                    
                if missed_count <= MAX_COASTING_FRAMES:
                    track_status = "COASTING"
                    
                    P_est_3d = np.array([P_est_2d[0], P_est_2d[1], z_ground])
                    uv_pred = project_target_mathematically(noisy_positions[idx], R_world_noisy, P_est_3d, K)
                    if uv_pred is not None:
                        u_p, v_p = int(uv_pred[0]), int(uv_pred[1])
                        cv2.rectangle(frame, (u_p - 45, v_p - 45), (u_p + 45, v_p + 45), (0, 255, 255), 3)
                        cv2.putText(frame, "WITH KF COASTING (COASTING)", (u_p - 45, max(20, v_p + 62)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(frame, (u_p, v_p), 4, (0, 255, 255), -1)
                else:
                    track_status = "LOST"
                    
        prev_frame_id = frame_id
        
        draw_hud_overlay(frame, processed_count, total_frames, fps, u_c, v_c, P_est_2d, track_status, missed_count=missed_count, altitude=altitude, preset_name=preset)
        
        if not writer_written:
            writer.write(frame)
        
        if show_preview:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nPreview window closed by user.")
                break
                
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            
            if key == ord('q'):
                print("\nEarly preview termination by user.")
                break
            elif key == ord(' ') or key == 32:
                print("\nPlayback PAUSED. Press [SPACE] to resume or [Q] to quit.")
                
                paused_frame = frame.copy()
                draw_hud_overlay(paused_frame, processed_count, total_frames, fps, u_c, v_c, P_est_2d, track_status, missed_count=missed_count, is_paused=True, altitude=altitude, preset_name=preset)
                cv2.imshow(window_name, paused_frame)
                
                while True:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("Preview window closed by user during pause.")
                        cap.release()
                        writer.release()
                        cv2.destroyAllWindows()
                        print_mission_summary(preset, processed_count, total_frames, fps, altitude, z_ground, kf2d, tracked_estimates, valid_detection_count)
                        return
                        
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key == ord(' ') or pause_key == 32:
                        print("Playback RESUMED.")
                        break
                    elif pause_key == ord('q'):
                        print("Early preview termination during pause.")
                        cap.release()
                        writer.release()
                        cv2.destroyAllWindows()
                        print_mission_summary(preset, processed_count, total_frames, fps, altitude, z_ground, kf2d, tracked_estimates, valid_detection_count)
                        return
                
    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()
        
    print_mission_summary(preset, processed_count, total_frames, fps, altitude, z_ground, kf2d, tracked_estimates, valid_detection_count)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.join(script_dir, "input_rendered.mp4")
    
    # Path Fallback: Check local folder then parent folder for telemetry.csv
    local_telemetry = os.path.join(script_dir, "telemetry.csv")
    parent_telemetry = os.path.abspath(os.path.join(script_dir, "..", "telemetry.csv"))
    default_telemetry = local_telemetry if os.path.exists(local_telemetry) else parent_telemetry
    
    default_output = os.path.join(script_dir, "output_annotated.mp4")

    parser = argparse.ArgumentParser(description="2D Ground Target Localization Pipeline")
    parser.add_argument('--preset', type=str, default='normal', choices=['normal', 'pressure'], help='Execution preset: "normal" (stable flight) or "pressure" (stress test dropouts & noise)')
    parser.add_argument('--video', type=str, default=default_video, help='Path to input video file')
    parser.add_argument('--telemetry', type=str, default=default_telemetry, help='Path to telemetry CSV')
    parser.add_argument('--output-video', type=str, default=default_output, help='Path to save annotated video')
    parser.add_argument('--mode', type=str, default='color', choices=['yolo', 'color', 'projection'], help='Target detection mode')
    parser.add_argument('--show', action='store_true', help='Toggle live resizable preview window')
    parser.add_argument('--loop', action='store_true', help='Loop video infinitely until closed')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate in FPS')
    parser.add_argument('--window-size', type=int, default=5, help='Sliding window buffer size N')
    args = parser.parse_args()
    
    process_video_pipeline(
        video_path=args.video,
        telemetry_path=args.telemetry,
        output_video_path=args.output_video,
        mode=args.mode,
        preset=args.preset,
        show_preview=args.show,
        loop_infinitely=args.loop,
        custom_fps=args.fps,
        window_size=args.window_size
    )
