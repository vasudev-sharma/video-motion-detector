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

    # TODO: Implement motion detection
    # 1. Convert frames to grayscale
    # 2. Apply Gaussian blur to reduce noise (hint: cv2.GaussianBlur)
    # 3. Calculate absolute difference between frames (hint: cv2.absdiff)
    # 4. Apply threshold to highlight differences (hint: cv2.threshold)
    # 5. Dilate the thresholded image to fill in holes (hint: cv2.dilate)
    # 6. Find contours in the thresholded image (hint: cv2.findContours)
    # 7. Filter contours by area and extract bounding boxes

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
    cv2.imshow("current frame", current_frame)
    # cv2.waitKey(0)
    cv2.imshow("Prev frame", prev_frame)
    # cv2.waitKey(0)

    # 4. Apply threshold
    _, absolute_diff = cv2.threshold(absolute_diff, threshold, 255, cv2.THRESH_BINARY)

    # Dilate image
    dilated_image = cv2.dilate(absolute_diff, None, iterations=5)
    cv2.imshow("Dilated image", dilated_image)
    # cv2.waitKey(0)

    # Countours

    countours_image, _ = cv2.findContours(
        dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for contor in countours_image:
        if cv2.contourArea(contor) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contor)
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("contour image", cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR))
        # cv2.waitKey(0)
        motion_boxes.append((x, y, x + w, y + h))

    return motion_boxes
