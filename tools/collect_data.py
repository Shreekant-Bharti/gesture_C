"""
collect_data.py
Data collection tool for hand gesture landmarks.
Saves one CSV per gesture in gesture_data/ folder.

Controls:
- SPACE: start/stop collecting samples
- n: next gesture
- q: quit
"""

import os
import csv
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
import mediapipe as mp

from config import GESTURES, CAMERA, MP, DATA, UI


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(CAMERA.index)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed. Check webcam connection and permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)
    cap.set(cv2.CAP_PROP_FPS, CAMERA.fps)
    return cap


def landmarks_to_feature_vector(hand_landmarks) -> Optional[np.ndarray]:
    """
    Convert 21 landmarks to 63 features (x,y,z) normalized relative to hand bounding box.
    Returns shape (63,).
    """
    if hand_landmarks is None:
        return None

    pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)  # (21,3)
    xs, ys = pts[:, 0], pts[:, 1]

    # Bounding box normalize (scale + translate)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    w = max(max_x - min_x, 1e-6)
    h = max(max_y - min_y, 1e-6)

    pts_norm = pts.copy()
    pts_norm[:, 0] = (pts[:, 0] - min_x) / w
    pts_norm[:, 1] = (pts[:, 1] - min_y) / h
    # z values in MediaPipe are relative already, still normalize roughly by box size
    pts_norm[:, 2] = pts[:, 2] / max(w, h)

    return pts_norm.flatten()  # 63


def draw_ui(frame: np.ndarray, gesture: str, idx: int, total: int,
            collecting: bool, sample_count: int, target: int) -> None:
    h, w = frame.shape[:2]
    panel_h = 140
    cv2.rectangle(frame, (0, 0), (w, panel_h), UI.bg_panel, -1)

    title = "Data Collection | ISL Gesture Dataset Builder"
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, UI.white, 2)

    status = "COLLECTING" if collecting else "PAUSED"
    status_color = UI.green if collecting else UI.yellow
    cv2.putText(frame, f"Status: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    cv2.putText(frame, f"Gesture [{idx+1}/{total}]: {gesture}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)

    cv2.putText(frame, f"Samples: {sample_count}/{target}", (520, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, UI.white, 2)

    controls = "SPACE start/stop | n next gesture | q quit"
    cv2.putText(frame, controls, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI.white, 2)


def append_to_csv(csv_path: str, row: np.ndarray) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=DATA.csv_delimiter)
        # Optional header only once
        if not file_exists:
            header = [f"f{i}" for i in range(63)]
            writer.writerow(header)
        writer.writerow(row.tolist())


def count_existing_samples(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        # subtract header
        return max(lines - 1, 0)
    except Exception:
        return 0


def main() -> None:
    ensure_dir(DATA.data_dir)

    try:
        cap = open_camera()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    gesture_index = 0
    collecting = False
    last_save_time = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MP.max_num_hands,
        min_detection_confidence=MP.min_detection_confidence,
        min_tracking_confidence=MP.min_tracking_confidence,
    ) as hands:

        print("[INFO] Data collection started.")
        print("[INFO] Controls: SPACE start/stop, n next gesture, q quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read from camera.")
                break

            if CAMERA.mirror:
                frame = cv2.flip(frame, 1)

            gesture = GESTURES[gesture_index]
            csv_path = os.path.join(DATA.data_dir, f"{gesture}.csv")
            sample_count = count_existing_samples(csv_path)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feature_vec = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                feature_vec = landmarks_to_feature_vector(hand_landmarks)

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )

            # Save sample if collecting and hand detected and delay passed
            now = time.time()
            if collecting and feature_vec is not None:
                if now - last_save_time >= DATA.sample_delay_sec:
                    append_to_csv(csv_path, feature_vec)
                    last_save_time = now
                    sample_count += 1
                    print(f"[SAVED] {gesture}: sample {sample_count}")

            draw_ui(frame, gesture, gesture_index, len(GESTURES),
                    collecting, sample_count, DATA.samples_per_gesture_target)

            cv2.imshow(UI.window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                collecting = not collecting
                state = "ON" if collecting else "OFF"
                print(f"[INFO] Collecting toggled {state} for gesture: {gesture}")
                # reset timer so you do not instantly save same pose frame
                last_save_time = time.time()
            if key == ord("n"):
                collecting = False
                gesture_index = (gesture_index + 1) % len(GESTURES)
                print(f"[INFO] Switched to next gesture: {GESTURES[gesture_index]}")
                last_save_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Data collection closed.")


if __name__ == "__main__":
    main()
