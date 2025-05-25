import numpy as np
import cv2


def calculate_region_of_interest(motion_boxes, frame_shape):
    """
    Calculate the weighted average center of motion boxes.
    Returns (center_x, center_y, width, height)
    """
    if not motion_boxes:
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)

    total_area = 0
    weighted_sum_x = 0
    weighted_sum_y = 0

    for box in motion_boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        total_area += area
        weighted_sum_x += center_x * area
        weighted_sum_y += center_y * area

    if total_area == 0:
        height, width = frame_shape[:2]
        return (width // 2, height // 2, 0, 0)

    avg_x = weighted_sum_x / total_area
    avg_y = weighted_sum_y / total_area

    return (int(avg_x), int(avg_y), 0, 0)


class KalmanTracker:
    def __init__(self, initial_x, initial_y):
        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kalman.statePre = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measured)
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])


def track_viewport(frames, motion_results, viewport_size, smoothing_factor=0.3):
    """
    Track viewport with Kalman Filter.
    """
    if not frames:
        return []

    height, width = frames[0].shape[:2]
    vpw, vph = viewport_size
    initial_x, initial_y = width // 2, height // 2
    kalman_tracker = KalmanTracker(initial_x, initial_y)

    viewport_positions = []

    for i, motion_boxes in enumerate(motion_results):
        roi_x, roi_y, _, _ = calculate_region_of_interest(motion_boxes, (height, width))
        predicted_x, predicted_y = kalman_tracker.update(roi_x, roi_y)

        # Enforce viewport boundaries
        predicted_x = max(vpw // 2, min(width - vpw // 2, predicted_x))
        predicted_y = max(vph // 2, min(height - vph // 2, predicted_y))

        viewport_positions.append((predicted_x, predicted_y))

    return viewport_positions