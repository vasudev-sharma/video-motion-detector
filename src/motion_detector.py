"""
Motion detection functions for the sports video analysis project.
"""

import cv2


def detect_motion(frames, frame_idx, threshold=25, min_area=350):
    """
    Detect motion in the current frame by comparing with previous frame.

    Args:
        frames: List of video frames
        frame_idx: Index of the current frame
        threshold: Threshold for frame difference detection
        min_area: Minimum contour area to consider

    Returns:
        List of bounding boxes for detected motion regions
    """
    # We need at least 2 frames to detect motion
    if frame_idx < 1 or frame_idx >= len(frames):
        return []

    # Get current and previous frame
    current_frame = frames[frame_idx]
    prev_frame = frames[frame_idx - 1]

    # Example starter code:
    motion_boxes = []
    kernel_size = (21, 21)
    current_frame = cv2.GaussianBlur(
        cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), kernel_size, 0
    )
    prev_frame = cv2.GaussianBlur(
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), kernel_size, 0
    )

    absolute_diff = cv2.absdiff(current_frame, prev_frame)

    # 4. Apply threshold
    _, absolute_diff = cv2.threshold(absolute_diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate image
    dilated_image = cv2.dilate(absolute_diff, None, iterations=5)

    # Countours

    countours_image, _ = cv2.findContours(
        dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contor in countours_image:
        if cv2.contourArea(contor) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contor)
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_boxes.append((x, y, x + w, y + h))

    return motion_boxes
