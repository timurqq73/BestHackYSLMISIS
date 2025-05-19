import os
import cv2
import mediapipe as mp
import numpy as np
import scipy.signal
import pywt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error

def min_max_scaling(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-9)

def remove_trend(signal, order=2):
    t = np.arange(len(signal))
    poly_coeff = np.polyfit(t, signal, order)
    trend = np.polyval(poly_coeff, t)
    return signal - trend

def extract_rppg_signal_chrom(video_path, low_pass=0.75, high_pass=2.5):
  
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1 or fps > 1000:
        fps = 30.0
    nyq = 0.5 * fps

    roi_indices = list({10, 338, 297, 332, 284, 251, 389, 356, 234, 93, 132, 58, 172, 162, 127})
    
    r_values, g_values, b_values = [], [], []
    times = []
    frame_index = 0

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            h, w, _ = frame.shape

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in roi_indices])
                min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
                min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
                dx = int(0.1 * (max_x - min_x))
                dy = int(0.1 * (max_y - min_y))
                x1, y1 = max(min_x - dx, 0), max(min_y - dy, 0)
                x2, y2 = min(max_x + dx, w), min(max_y + dy, h)
                if x1 < x2 and y1 < y2:
                    roi = frame_rgb[y1:y2, x1:x2]
                    mean_rgb = np.mean(roi, axis=(0, 1))
                else:
                    mean_rgb = [0, 0, 0]
                r_values.append(mean_rgb[0])
                g_values.append(mean_rgb[1])
                b_values.append(mean_rgb[2])

            times.append(frame_index / fps)
            frame_index += 1

    cap.release()
    times = np.array(times)
    r_values = np.array(r_values)
    g_values = np.array(g_values)
    b_values = np.array(b_values)

    X = (2.0 * r_values - 1.5 * g_values) * 2
    Y = (1.5 * g_values - 1.5 * b_values) * 1.5

    X = remove_trend(X, order=2)
    Y = remove_trend(Y, order=2)

    X_norm = X / (np.std(X) + 1e-9)
    Y_norm = Y / (np.std(Y) + 1e-9)
    chrom_signal = X_norm - 0.5 * Y_norm

    b_filt, a_filt = scipy.signal.butter(2, [low_pass/nyq, high_pass/nyq], btype='bandpass')
    filtered = scipy.signal.filtfilt(b_filt, a_filt, chrom_signal)

    window_size = int(fps * 0.75)
    if window_size % 2 == 0:
        window_size += 1
    filtered = scipy.signal.savgol_filter(filtered, window_size, 3)

    return times, min_max_scaling(filtered)
def process_all_videos(dataset_path):
    
    total_rmse, total_mae = [], []
    pairs_processed = 0

    for root, _, files in os.walk(dataset_path):
        video_files = sorted([f for f in files if f.endswith('.avi')])
        gt_files = sorted([f for f in files if f.endswith('.txt')])

        for video_file, gt_file in zip(video_files, gt_files):
            video_path = os.path.join(root, video_file)
            gt_path = os.path.join(root, gt_file)

            try:
                print(f"\nОбрабатываем: {video_file} / {gt_file}")
                times, extracted_signal = extract_rppg_signal_chrom(video_path)

                with open(gt_path, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) < 3:
                    print(f"Ошибка: {gt_file} содержит менее 3 строк! Пропускаем файл.")
                    continue
                gt_signal = np.array([float(x) for x in lines[0].split()])
                gt_time = np.array([float(x) for x in lines[2].split()])

                
                f_gt = interp1d(gt_time, gt_signal, kind='linear', fill_value="extrapolate")
                gt_aligned = f_gt(times)

                gt_aligned = min_max_scaling(gt_aligned)
                extracted_signal = min_max_scaling(extracted_signal)

                rmse = np.sqrt(mean_squared_error(gt_aligned, extracted_signal))
                mae = mean_absolute_error(gt_aligned, extracted_signal)

                print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

                total_rmse.append(rmse)
                total_mae.append(mae)
                pairs_processed += 1

            except Exception as e:
                print(f"Ошибка с {video_file}: {e}")

    if pairs_processed > 0:
        print(f"\nСредний RMSE: {np.mean(total_rmse):.4f}")
        print(f"Средний MAE: {np.mean(total_mae):.4f}")
    else:
        print("Не обработано ни одной пары.")
if __name__ == "__main__":
    dataset_path = "test"
    process_all_videos(dataset_path)
