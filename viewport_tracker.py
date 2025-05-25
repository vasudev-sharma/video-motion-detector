import numpy as np
import cv2

def calculate_region_of_interest(motion_boxes, frame_shape, prev_centroid=None):
    """
    Enhanced ROI calculation with dynamic weighting and motion consistency checks
    """
    height, width = frame_shape[:2]
    
    if not motion_boxes:
        return (width // 2, height // 2, 0, 0) if prev_centroid is None else prev_centroid

    total_weight = 0
    weighted_sum = np.zeros(2)
    
    for box in motion_boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        center = np.array([(x1 + x2)/2, (y1 + y2)/2])
        
        # Dynamic weighting based on box size and distance from previous centroid [7]
        size_weight = np.sqrt(area)
        distance_weight = 1.0
        if prev_centroid is not None:
            distance = np.linalg.norm(center - prev_centroid[:2])
            distance_weight = 1 / (1 + distance)  # Inverse distance weighting
            
        weight = size_weight * distance_weight
        weighted_sum += center * weight
        total_weight += weight

    if total_weight == 0:
        return (width // 2, height // 2, 0, 0)

    avg_center = (weighted_sum / total_weight).astype(int)
    return (*avg_center, 0, 0)

class AdaptiveKalmanTracker:
    """
    Enhanced Kalman filter with adaptive noise covariance and acceleration modeling [2][4][8]
    """
    def __init__(self, initial_x, initial_y):
        self.kalman = cv2.KalmanFilter(6, 2)  # Extended state (x,y,vx,vy,ax,ay)
        
        # State transition matrix with acceleration terms [4]
        self.kalman.transitionMatrix = np.array([
            [1,0,1,0,0.5,0],
            [0,1,0,1,0,0.5],
            [0,0,1,0,1,0],
            [0,0,0,1,0,1],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ], np.float32)

        self.kalman.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0]
        ], np.float32)

        # Adaptive noise parameters [3][5]
        self.base_process_noise = 0.03
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * self.base_process_noise
        self.measurement_noise_base = 10.0
        
        # Initialize state
        self.kalman.statePre = np.array(
            [[initial_x], [initial_y], [0], [0], [0], [0]], dtype=np.float32
        )

    def update(self, x, y, measurement_confidence=1.0):
        """
        Update with adaptive noise based on measurement confidence [8]
        """
        # Adjust measurement noise covariance dynamically [5]
        self.kalman.measurementNoiseCov = np.eye(2) * self.measurement_noise_base / measurement_confidence
        
        # Adaptive process noise [3]
        velocity = np.linalg.norm(self.kalman.statePre[2:4])
        process_noise_scale = 1 + 0.1 * velocity
        self.kalman.processNoiseCov = np.eye(6) * self.base_process_noise * process_noise_scale

        measured = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measured)
        prediction = self.kalman.predict()
        
        return int(prediction[0]), int(prediction[1])

def track_viewport(frames, motion_results, viewport_size):
    """
    Enhanced tracking with motion model adaptation and boundary handling
    """
    # ... (initialization similar to original)
    
    tracker = AdaptiveKalmanTracker(initial_x, initial_y)
    prev_centroid = None
    
    for i, motion_boxes in enumerate(motion_results):
        # Calculate ROI with motion consistency [1]
        roi = calculate_region_of_interest(motion_boxes, (height, width), prev_centroid)
        prev_centroid = roi
        
        # Calculate measurement confidence based on motion box consistency [6]
        confidence = min(1.0, len(motion_boxes) * 0.1) if motion_boxes else 0.01
        
        predicted = tracker.update(roi[0], roi[1], confidence)
        
        # Smooth boundary handling with velocity damping [4]
        x, y = predicted
        vx = tracker.kalman.statePost[2][0]
        vy = tracker.kalman.statePost[3][0]
        
        # Apply soft boundary constraints with velocity damping
        x = _apply_boundary_constraint(x, vpw//2, width - vpw//2, vx)
        y = _apply_boundary_constraint(y, vph//2, height - vph//2, vy)
        
        viewport_positions.append((x, y))
    
    return viewport_positions

def _apply_boundary_constraint(value, min_val, max_val, velocity):
    """
    Smooth boundary constraint with velocity damping [4]
    """
    if value < min_val:
        return min_val + int(0.1 * velocity)
    if value > max_val:
        return max_val - int(0.1 * velocity)
    return value
